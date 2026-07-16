import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_COMPONENT_PATH = Path(__file__).resolve().parents[1] / "model" / "hire_v2_anchor_components.py"
_SPEC = importlib.util.spec_from_file_location("hire_v2_anchor_components_test", _COMPONENT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RDETextTokenSelection = _MODULE.RDETextTokenSelection
RDEVisualTokenSelection = _MODULE.RDEVisualTokenSelection
ResidualObservationFusion = _MODULE.ResidualObservationFusion
CLIPAttentionAdapter = _MODULE.CLIPAttentionAdapter
aggregate_anchor_objectives = _MODULE.aggregate_anchor_objectives


def test_residual_fusion_initially_equals_normalized_global():
    torch.manual_seed(1)
    module = ResidualObservationFusion(global_dim=8, local_dim=12)
    global_feature = torch.randn(4, 8)
    local_feature = torch.randn(4, 12)
    observation, residual = module(global_feature, local_feature)
    assert torch.allclose(observation, F.normalize(global_feature, dim=-1), atol=1e-6)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_residual_fusion_adapter_receives_nonzero_gradient():
    torch.manual_seed(2)
    module = ResidualObservationFusion(global_dim=8, local_dim=12)
    global_feature = torch.randn(4, 8)
    local_feature = torch.randn(4, 12)
    target = F.normalize(torch.randn(4, 8), dim=-1)
    observation, _ = module(global_feature, local_feature)
    loss = 1.0 - (observation * target).sum(dim=-1).mean()
    loss.backward()
    gradient = module.local_adapter.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_visual_token_selection_shape_and_finiteness():
    torch.manual_seed(3)
    module = RDEVisualTokenSelection(input_dim=8, output_dim=16, ratio=0.3)
    module.eval()
    tokens = torch.randn(4, 11, 8)
    attention = torch.softmax(torch.randn(4, 11, 11), dim=-1)
    output = module(tokens, attention)
    assert output.shape == (4, 16)
    assert torch.isfinite(output).all()


def test_text_token_selection_ignores_padding_and_special_tokens():
    torch.manual_seed(4)
    module = RDETextTokenSelection(input_dim=8, output_dim=16, ratio=0.3)
    module.eval()
    tokens = torch.randn(3, 10, 8)
    token_ids = torch.tensor([
        [49406, 12, 13, 14, 49407, 0, 0, 0, 0, 0],
        [49406, 21, 22, 23, 24, 49407, 0, 0, 0, 0],
        [49406, 31, 32, 49407, 0, 0, 0, 0, 0, 0],
    ])
    attention = torch.softmax(torch.randn(3, 10, 10), dim=-1)
    output = module(tokens, token_ids, attention)
    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()


def test_padding_values_do_not_change_text_output_or_batchnorm_statistics():
    torch.manual_seed(40)
    original = RDETextTokenSelection(input_dim=8, output_dim=16, ratio=0.8)
    altered = copy.deepcopy(original)
    original.train()
    altered.train()

    token_ids = torch.tensor([
        [49406, 12, 13, 49407, 0, 0, 0, 0, 0, 0],
        [49406, 21, 22, 23, 49407, 0, 0, 0, 0, 0],
        [49406, 31, 32, 33, 34, 49407, 0, 0, 0, 0],
    ])
    tokens_a = torch.randn(3, 10, 8)
    tokens_b = tokens_a.clone()
    invalid = token_ids.eq(0)
    tokens_b[invalid] = torch.randn_like(tokens_b[invalid]) * 1000.0
    attention = torch.softmax(torch.randn(3, 10, 10), dim=-1)

    output_a = original(tokens_a, token_ids, attention)
    output_b = altered(tokens_b, token_ids, attention)
    assert torch.allclose(output_a, output_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(original.mlp.bn1.running_mean, altered.mlp.bn1.running_mean)
    assert torch.allclose(original.mlp.bn1.running_var, altered.mlp.bn1.running_var)
    assert torch.allclose(original.mlp.bn2.running_mean, altered.mlp.bn2.running_mean)
    assert torch.allclose(original.mlp.bn2.running_var, altered.mlp.bn2.running_var)


def test_local_residual_can_change_observation_after_update():
    torch.manual_seed(5)
    module = ResidualObservationFusion(global_dim=8, local_dim=12)
    with torch.no_grad():
        module.local_adapter.weight.normal_(std=0.05)
    global_feature = torch.randn(4, 8)
    local_feature = torch.randn(4, 12)
    observation, residual = module(global_feature, local_feature)
    assert residual.abs().sum() > 0
    assert not torch.allclose(observation, F.normalize(global_feature, dim=-1))
    assert torch.allclose(observation.norm(dim=-1), torch.ones(4), atol=1e-6)



class _FakeResidualBlock(torch.nn.Module):
    def __init__(self, width: int, heads: int):
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
    def __init__(self, width: int = 8, heads: int = 2, depth: int = 3):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [_FakeResidualBlock(width, heads) for _ in range(depth)]
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


def test_attention_adapter_preserves_transformer_output():
    torch.manual_seed(41)
    expected_transformer = _FakeTransformer()
    adapter_transformer = copy.deepcopy(expected_transformer)
    expected_transformer.eval()
    adapter_transformer.eval()
    x = torch.randn(7, 3, 8)
    expected = expected_transformer(x.clone())
    actual, attention = CLIPAttentionAdapter._run_transformer_with_last_attention(
        adapter_transformer, x.clone()
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert attention.shape == (3, 7, 7)
    assert torch.isfinite(attention).all()

def test_anchor_objective_aggregation_matches_documented_formula():
    values = [torch.tensor(float(index), requires_grad=True) for index in range(1, 7)]
    aggregated = aggregate_anchor_objectives(*values)
    expected_sdm = 0.5 * (values[0] + values[2]) + values[4]
    expected_itc = 0.5 * (values[1] + values[3]) + values[5]
    expected_anchor = 0.5 * sum(values[:4])
    expected_observation = values[4] + values[5]
    assert torch.equal(aggregated["sdm_loss"], expected_sdm)
    assert torch.equal(aggregated["itc_loss"], expected_itc)
    assert torch.equal(aggregated["anchor_objective"], expected_anchor)
    assert torch.equal(aggregated["observation_objective"], expected_observation)
    (aggregated["sdm_loss"] + aggregated["itc_loss"]).backward()
    assert all(value.grad is not None for value in values)
