# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual build_ctx helper with CPU tensors and no trainer runtime."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def build_ctx():
    # This helper module needs only torch; avoid verl's distributed package imports.
    source = Path(__file__).resolve().parents[2] / "verl/utils/speculative_decoding.py"
    spec = importlib.util.spec_from_file_location("speculative_build_ctx_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_ctx


@pytest.mark.parametrize("left_padded", [False, True], ids=["unpadded", "left-padded"])
@pytest.mark.parametrize(
    "cuts",
    [[0, 0, 0], [0, 2, 0], [1, 2, 3], [-1, 20, 0]],
    ids=["all-zero", "mixed-zero-positive", "positive", "clamped"],
)
def test_build_ctx(build_ctx, left_padded, cuts):
    if left_padded:
        prompt_ids = torch.tensor([[0, 0, 11, 12], [0, 21, 22, 23], [31, 32, 33, 34]])
    else:
        prompt_ids = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
    prompt_mask = (prompt_ids != 0).long()
    prompt_positions = (prompt_mask.cumsum(1) - 1).clamp_min(0)
    responses = torch.tensor([[41, 42, 43], [51, 52, 53], [61, 62, 63]])
    originals = [tensor.clone() for tensor in (prompt_ids, prompt_mask, prompt_positions, responses)]

    actual_ids, actual_mask, actual_positions, max_k = build_ctx(
        prompt_ids, prompt_mask, prompt_positions, responses, torch.tensor(cuts), pad_id=0
    )

    lengths = [min(max(cut, 0), responses.shape[1]) for cut in cuts]
    assert max_k == max(lengths)
    width = prompt_ids.shape[1] + max_k
    expected_ids = torch.zeros((3, width), dtype=prompt_ids.dtype)
    expected_mask = torch.zeros_like(expected_ids)
    expected_positions = torch.zeros_like(expected_ids)
    for row, length in enumerate(lengths):
        valid = prompt_mask[row].bool()
        ids = torch.cat([prompt_ids[row, valid], responses[row, :length]])
        positions = torch.cat([prompt_positions[row, valid], prompt_positions[row, -1] + torch.arange(1, length + 1)])
        expected_ids[row, -len(ids) :] = ids
        expected_mask[row, -len(ids) :] = 1
        expected_positions[row, -len(ids) :] = positions

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_mask, expected_mask)
    torch.testing.assert_close(actual_positions, expected_positions)
    for actual, original in zip((prompt_ids, prompt_mask, prompt_positions, responses), originals, strict=True):
        torch.testing.assert_close(actual, original)


def test_build_ctx_empty_sub_batch(build_ctx):
    prompts = torch.empty((0, 4), dtype=torch.long)
    responses = torch.empty((0, 3), dtype=torch.long)
    ids, mask, positions, max_k = build_ctx(
        prompts, prompts.clone(), prompts.clone(), responses, torch.empty(0, dtype=torch.long), pad_id=0
    )
    assert max_k == 0
    assert ids.shape == mask.shape == positions.shape == (0, 4)
