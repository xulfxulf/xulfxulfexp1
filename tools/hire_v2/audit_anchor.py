#!/usr/bin/env python
"""Static and mathematical audit for the HIRE-v2 anchor overlay."""

from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys

import torch
import torch.nn.functional as F


REQUIRED_FILES = [
    "HIRE_V2_ANCHOR_DESIGN.md",
    "model/hire_v2_anchor_components.py",
    "model/hire_v2_anchor_model.py",
    "model/__init__.py",
    "utils/options.py",
    "solver/build.py",
    "processor/processor.py",
    "run_hire_v2_anchor_4090_tag.sh",
    "run_hire_v2_anchor_smoke.sh",
    "tools/hire_v2/eval_anchor_components.py",
    "tests/test_hire_v2_anchor_components.py",
]



class _FakeResidualBlock(torch.nn.Module):
    def __init__(self, width=8, heads=2):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, heads, dropout=0.0)
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, width * 2),
            torch.nn.GELU(),
            torch.nn.Linear(width * 2, width),
        )
        self.attn_mask = None

    def forward(self, x):
        normalized = self.ln_1(x)
        attended = self.attn(
            normalized,
            normalized,
            normalized,
            need_weights=False,
            attn_mask=self.attn_mask,
        )[0]
        x = x + attended
        return x + self.mlp(self.ln_2(x))


class _FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = torch.nn.ModuleList([_FakeResidualBlock() for _ in range(3)])

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x

def _load_components(root: pathlib.Path):
    component_path = root / "model" / "hire_v2_anchor_components.py"
    spec = importlib.util.spec_from_file_location(
        "hire_v2_anchor_components_audit", component_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    if missing:
        raise RuntimeError("missing delivered files: {}".format(missing))

    module = _load_components(root)
    ResidualObservationFusion = module.ResidualObservationFusion
    RDETextTokenSelection = module.RDETextTokenSelection
    CLIPAttentionAdapter = module.CLIPAttentionAdapter
    aggregate_anchor_objectives = module.aggregate_anchor_objectives

    torch.manual_seed(7)
    fusion = ResidualObservationFusion(global_dim=8, local_dim=12)
    global_feature = torch.randn(5, 8)
    local_feature = torch.randn(5, 12)
    observation, residual = fusion(global_feature, local_feature)
    expected = F.normalize(global_feature, dim=-1)
    if not torch.allclose(observation, expected, atol=1e-6, rtol=1e-6):
        raise RuntimeError("zero-initialized fusion does not preserve CLIP geometry")
    if not torch.equal(residual, torch.zeros_like(residual)):
        raise RuntimeError("zero-initialized local residual is not exactly zero")

    target = F.normalize(torch.randn_like(observation), dim=-1)
    loss = 1.0 - (observation * target).sum(dim=-1).mean()
    loss.backward()
    gradient = fusion.local_adapter.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("local residual adapter gradient is missing or non-finite")
    if gradient.abs().sum().item() <= 0:
        raise RuntimeError("local residual adapter gradient is exactly zero")

    # Verify the documented loss aggregation independently of the model class.
    values = [torch.tensor(float(index)) for index in range(1, 7)]
    aggregated = aggregate_anchor_objectives(*values)
    if not torch.equal(aggregated["sdm_loss"], 0.5 * (values[0] + values[2]) + values[4]):
        raise RuntimeError("SDM aggregation differs from the design document")
    if not torch.equal(aggregated["itc_loss"], 0.5 * (values[1] + values[3]) + values[5]):
        raise RuntimeError("ITC aggregation differs from the design document")

    # Padding and special-token values must not contaminate the text MLP's
    # BatchNorm statistics after they have been masked out.
    torch.manual_seed(11)
    module_a = RDETextTokenSelection(input_dim=8, output_dim=16, ratio=0.8)
    module_b = copy.deepcopy(module_a)
    module_a.train()
    module_b.train()
    token_ids = torch.tensor([
        [49406, 12, 13, 49407, 0, 0, 0, 0, 0, 0],
        [49406, 21, 22, 23, 49407, 0, 0, 0, 0, 0],
    ])
    tokens_a = torch.randn(2, 10, 8)
    tokens_b = tokens_a.clone()
    invalid = token_ids.eq(0)
    tokens_b[invalid] = torch.randn_like(tokens_b[invalid]) * 1000.0
    attention = torch.softmax(torch.randn(2, 10, 10), dim=-1)
    output_a = module_a(tokens_a, token_ids, attention)
    output_b = module_b(tokens_b, token_ids, attention)
    if not torch.allclose(output_a, output_b, atol=1e-5, rtol=1e-5):
        raise RuntimeError("padding values influence text token-selection output")

    torch.manual_seed(13)
    standard_transformer = _FakeTransformer().eval()
    adapted_transformer = copy.deepcopy(standard_transformer).eval()
    transformer_input = torch.randn(7, 3, 8)
    expected_transformer_output = standard_transformer(transformer_input.clone())
    adapted_output, attention = CLIPAttentionAdapter._run_transformer_with_last_attention(
        adapted_transformer, transformer_input.clone()
    )
    if not torch.allclose(
        adapted_output, expected_transformer_output, atol=1e-6, rtol=1e-5
    ):
        raise RuntimeError("attention adapter changes the transformer output")
    if attention.shape != (3, 7, 7) or not torch.isfinite(attention).all():
        raise RuntimeError("attention adapter returned an invalid attention matrix")

    model_init = (root / "model/__init__.py").read_text(encoding="utf-8")
    options = (root / "utils/options.py").read_text(encoding="utf-8")
    design = (root / "HIRE_V2_ANCHOR_DESIGN.md").read_text(encoding="utf-8")
    if "build_hire_v2_anchor_model" not in model_init:
        raise RuntimeError("model factory does not dispatch HIRE-v2 anchor")
    if "--hire_v2" not in options or "hire_v2_anchor" not in options:
        raise RuntimeError("HIRE-v2 anchor options are incomplete")
    if "\x0c" in design:
        raise RuntimeError("design document contains a form-feed/LaTeX corruption")

    print("HIRE-v2 anchor audit passed")
    print("- required files: present")
    print("- zero-initialized fusion: exact")
    print("- local adapter gradient: finite and non-zero")
    print("- documented loss aggregation: exact")
    print("- padded text tokens: excluded from token MLP and BatchNorm")
    print("- attention adapter: output-equivalent and attention-valid")
    print("- model dispatch, options, and document encoding: valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
