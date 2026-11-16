# SPDX-License-Identifier: Apache-2.0
"""Run the trainer's post-rollout tensor block without importing Ray/vLLM.

The block is extracted unchanged from fit(), including its control flow. These
are CPU tensor regression tests, not a distributed trainer integration test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def assembly_code():
    source = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text())
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and any(isinstance(arg, ast.Constant) and arg.value == "post-rollout" for arg in item.context_expr.args)
            for item in node.items
        )
    ]
    assert len(blocks) == 1
    body = blocks[0].body
    start = next(
        i for i, node in enumerate(body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "(B, P)"
    )
    end = next(
        i for i, node in enumerate(body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "merged"
    )
    block = ast.Module(body=body[start : end + 1], type_ignores=[])
    return compile(block, str(source), "exec")


@pytest.mark.parametrize(
    "cut_indices,reuse_indices,continuations,expected_responses,expected_masks",
    [
        pytest.param(
            [3, 4, 5],
            [0, 1, 2],
            [],
            [[41, 42, 0, 0, 0], [51, 52, 53, 0, 0], [61, 62, 63, 64, 65]],
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
            id="all-reuse",
        ),
        pytest.param(
            [3, 1, 0],
            [0],
            [[71, 0, 0, 0, 0], [81, 82, 83, 84, 85]],
            [[41, 42, 0, 0, 0], [51, 71, 0, 0, 0], [81, 82, 83, 84, 85]],
            [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
            id="mixed",
        ),
        pytest.param(
            [0, 2, 1],
            [],
            [[71, 0, 0, 0, 0], [81, 82, 0, 0, 0], [91, 92, 93, 94, 95]],
            [[71, 0, 0, 0, 0], [51, 52, 81, 82, 0], [61, 91, 92, 93, 94]],
            [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            id="all-continuation",
        ),
    ],
)
def test_speculative_assembly(
    assembly_code, cut_indices, reuse_indices, continuations, expected_responses, expected_masks
):
    prompt_ids = torch.tensor([[0, 0, 11, 12], [0, 21, 22, 23], [31, 32, 33, 34]])
    prompt_mask = (prompt_ids != 0).long()
    prompt_positions = (prompt_mask.cumsum(1) - 1).clamp_min(0)
    need_indices = [row for row in range(3) if row not in reuse_indices]
    env = {
        "torch": torch,
        "pad_id": 0,
        "prompt_ids": prompt_ids,
        "prompt_attn_mask": prompt_mask,
        "prompt_pos_ids": prompt_positions,
        "aligned_old_responses": torch.tensor([[41, 42, 0, 0, 0], [51, 52, 53, 0, 0], [61, 62, 63, 64, 65]]),
        "aligned_old_response_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
        "idx_reuse": torch.tensor(reuse_indices, dtype=torch.long),
        "idx_need": torch.tensor(need_indices, dtype=torch.long),
        "cut_idx": torch.tensor(cut_indices),
        "gen_batch_output": SimpleNamespace(batch={"responses": torch.tensor(continuations, dtype=torch.long)}),
    }
    exec(assembly_code, env)
    merged = env["merged"]
    expected_responses = torch.tensor(expected_responses)
    torch.testing.assert_close(merged["responses"], expected_responses)
    torch.testing.assert_close(merged["input_ids"], torch.cat([prompt_ids, expected_responses], dim=1))
    # Preserve the existing convention: include the first pad/EOT in the mask.
    torch.testing.assert_close(merged["attention_mask"], torch.cat([prompt_mask, torch.tensor(expected_masks)], dim=1))
    response_positions = prompt_positions[:, -1:] + torch.arange(1, 6)
    torch.testing.assert_close(merged["position_ids"], torch.cat([prompt_positions, response_positions], dim=1))
    torch.testing.assert_close(merged["prompts"], prompt_ids)
