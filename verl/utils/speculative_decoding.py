import torch
import math

def _rand_like_compat(t: torch.Tensor, seed: int | None = None):
    if seed is None:
        return torch.rand_like(t)
    g = torch.Generator(device=t.device if t.device.type == "cpu" else "cpu")
    g.manual_seed(seed)
    try:
        return torch.rand(t.shape, device=t.device, dtype=t.dtype, generator=g)
    except TypeError:
        torch.manual_seed(seed)
        return torch.rand_like(t)

@torch.no_grad()
def spec_cut(
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    response_mask: torch.Tensor,
    p_abs_thresh: float | None = None,
    seed: int | None = None,
):

    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim() == 2
    B, R = new_logp.shape
    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    # Δlogp = new - old
    log_ratio = (new_logp - old_logp).masked_fill(~valid, 0.0)

    U = _rand_like_compat(new_logp, seed=seed).clamp_min(1e-12)
    logU = torch.log(U)

    # accept condition: (Δ>=0) or (logU <= Δ)
    accept_mask = (~valid) | (log_ratio >= 0) | (logU <= log_ratio)

    # optional absolute threshold: require new_logp >= log(p_abs_thresh)
    if p_abs_thresh is not None:
        import math
        log_th = math.log(p_abs_thresh)
        accept_mask &= ((new_logp >= log_th) | (~valid))

    bad = valid & (~accept_mask)
    has_bad   = bad.any(dim=1)
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)  # no bad = 0
    cut_idx   = torch.where(has_bad, first_bad, resp_len)

    reuse_mask = (cut_idx == resp_len)
    need_mask  = ~reuse_mask
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)

    saved_tokens = (resp_len - cut_idx).clamp(min=0)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
    }
    return {
        "cut_idx": cut_idx,
        "resp_len": resp_len,
        "idx_reuse": idx_reuse, "idx_need": idx_need,
        "per_request_max_new_tokens": per_request_max_new_tokens,
        "metrics": metrics,
    }


@torch.no_grad()
def spec_cut_with_knobs(
    old_logp: torch.Tensor,         # [B,R]
    new_logp: torch.Tensor,         # [B,R]
    response_mask: torch.Tensor,    # [B,R] 1=valid response token
    *,
    bias: float = 0.0,              # b:  >0 more strict; <0 more lenient
    scale: float = 1.0,             # s:  <1 more lenient; >1 more strict
    p_abs_thresh: float | None = None,  # optional: new >= 0.3
    seed: int | None = None,
):
    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim()==2
    B, R   = new_logp.shape
    valid  = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    delta  = (new_logp - old_logp).masked_fill(~valid, 0.0)  # Δ

    delta2 = scale * (delta - bias)

    logU = torch.log(_rand_like_compat(new_logp, seed=seed).clamp_min(1e-12))
    accept = (~valid) | (delta2 >= 0) | (logU <= delta2)

    if p_abs_thresh is not None:
        log_th = math.log(p_abs_thresh)
        accept &= ((new_logp >= log_th) | (~valid))

    bad = valid & (~accept)
    has_bad   = bad.any(dim=1)
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)       # 无 bad -> 0
    cut_idx   = torch.where(has_bad, first_bad, resp_len)     # [B]

    reuse_mask = (cut_idx == resp_len)
    need_mask  = ~reuse_mask
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)

    # note: saved tokens ≈ accepted prefix = cut_idx
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/bias": float(bias), "spec/scale": float(scale),

    }
    return {
        "cut_idx": cut_idx,
        "idx_reuse": idx_reuse, "idx_need": idx_need,
        "resp_len": resp_len,
        "per_request_max_new_tokens": (R - cut_idx).to(torch.long),
        "metrics": metrics
        
    }


@torch.no_grad()
def rand_reuse_cut(
    old_logp: torch.Tensor,          # [B,R] 
    new_logp: torch.Tensor,          # [B,R]
    response_mask: torch.Tensor,     # [B,R]
    *,
    reuse_prob: float,               # ∈[0,1]
    seed: int | None = None,
):
    # ---- basic shape check ----
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))  # clamp to [0,1]

    valid    = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)          # [B]

    # ---- randomly select rows with probability reuse_prob ----
    # only need per-row random number: reuse -> cut_idx=resp_len; otherwise -> cut_idx=0
    # use new_logp's dtype/device to produce random number, ensure device/precision consistency
    row_rand  = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)  # [B]
    reuse_mask = (row_rand < reuse_prob)                                    # [B] bool
    need_mask  = ~reuse_mask

    # ---- calculate cut_idx / idx_* / per_request_max_new_tokens（与 spec_*\* 对齐）----
    zero_like = torch.zeros_like(resp_len)
    cut_idx   = torch.where(reuse_mask, resp_len, zero_like)                # [B]

    # ---- additional check: ensure cut_idx is in valid range ----
    cut_idx = cut_idx.clamp(min=torch.tensor([0] * resp_len.shape[0]), max=resp_len)  # ensure cut_idx is in valid range

    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)       # [Nr]
    idx_need  = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)       # [Nn]
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)               # [B]

    # ---- calculate metrics (keep spec/* names, avoid downstream changes)----
    # "saved tokens" use the same definition as spec_cut_with_knobs: saved ≈ accepted prefix = cut_idx
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_p":   float(reuse_prob),
    }

    return {
        "cut_idx": cut_idx,                           # [B] long
        "resp_len": resp_len,                         # [B] long
        "idx_reuse": idx_reuse, "idx_need": idx_need, # 1D long
        "per_request_max_new_tokens": per_request_max_new_tokens,  # [B] long
        "metrics": metrics,
    }

@torch.no_grad()
def rand_reuse_all_cut(
    old_logp: torch.Tensor,          # [B,R]
    new_logp: torch.Tensor,          # [B,R]
    response_mask: torch.Tensor,     # [B,R]
    *,
    reuse_prob: float,               # probability to randomly select truncation for all data
    seed: int | None = None,
):
    # ---- basic shape check ----
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))  # clamp to [0,1]

    valid    = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)          # [B]

    # ---- randomly select truncation point with probability reuse_prob ----
    # use new_logp's dtype/device to produce random number, ensure device/precision consistency
    row_rand  = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)  # [B]
    cut_idx   = torch.floor(row_rand * resp_len.to(torch.float32)).to(torch.long)  # [B]
    
    # ensure cut_idx is in valid range

    cut_idx = cut_idx.clamp(min=torch.tensor([0] * resp_len.shape[0]), max=resp_len)  # ensure cut_idx is in valid range

    # ---- calculate idx_* / per_request_max_new_tokens (align with spec_*)----
    reuse_mask = (cut_idx == resp_len)                       # [B] bool
    need_mask  = ~reuse_mask                                 # [B] bool
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)  # [Nr]
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)  # [Nn]
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)  # [B]

    # ---- calculate metrics (keep spec/* names, avoid downstream changes)----
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_all_p":   float(reuse_prob),  # new: random truncation probability
    }

    return {
        "cut_idx": cut_idx,                           # [B] long
        "resp_len": resp_len,                         # [B] long
        "idx_reuse": idx_reuse, "idx_need": idx_need, # 1D long
        "per_request_max_new_tokens": per_request_max_new_tokens,  # [B] long
        "metrics": metrics,
    }
