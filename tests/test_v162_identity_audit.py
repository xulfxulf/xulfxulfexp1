import torch

# Import through repository path when the overlay is unpacked.
from tools.hire_v2.audit_v162_identity import (
    masked_mean,
    masked_std,
    row_metrics_from_scores,
    topk_overlap,
)


def test_masked_stats():
    values = torch.tensor([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    mean = masked_mean(values, mask, dim=1)
    std = masked_std(values, mask, dim=1)
    assert torch.allclose(mean, torch.tensor([2.0, 4.0]))
    assert torch.allclose(std, torch.tensor([1.0, (8.0 / 3.0) ** 0.5]), atol=1e-5)


def test_row_metrics():
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.2, 0.1]])
    qpid = torch.tensor([2, 1])
    gpid = torch.tensor([1, 2, 3])
    result = row_metrics_from_scores(scores, qpid, gpid)
    assert result["correct"].tolist() == [True, True]
    assert result["best_positive_rank"].tolist() == [1, 1]


def test_topk_overlap():
    left = torch.tensor([[1, 2, 3], [3, 4, 5]])
    right = torch.tensor([[3, 2, 9], [5, 4, 3]])
    overlap = topk_overlap(left, right)
    assert torch.allclose(overlap, torch.tensor([2.0 / 3.0, 1.0]))
