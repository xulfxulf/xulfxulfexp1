import unittest

import numpy as np
import torch
from torch import nn

from research.data import aggregate_probabilities, view3, support_candidates, SupportSampler
from research.model import confidence, SoftViewCLIP
from view4.data import PlanBatchSampler


class Tiny(SoftViewCLIP):
    def __init__(self):
        nn.Module.__init__(self)
        self.experts = nn.ModuleList([nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 2), nn.GELU(), nn.Linear(2, 8))
                                     for _ in range(3)])
        self.residual_alpha = .2


class SoftThreeTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual([view3(v) for v in (0, 30, 35, 145, 150, 210, 215, 325, 330, 360)],
                         [2, 2, 1, 1, 0, 0, 1, 1, 2, 2])

    def test_flip_probability_invariance(self):
        p = np.random.RandomState(3).dirichlet(np.ones(72))
        np.testing.assert_allclose(aggregate_probabilities(p), aggregate_probabilities(p[(-np.arange(72)) % 72]))

    def test_probability_rejects_bad_inputs(self):
        for p in ([0.] * 72, [1.] * 71, [float("nan")] * 72):
            with self.assertRaises(ValueError):
                aggregate_probabilities(p)

    def test_confidence_is_uncalibrated_entropy(self):
        p = torch.tensor([[1., 0., 0.], [1/3, 1/3, 1/3]])
        torch.testing.assert_close(confidence(p), torch.tensor([1., 0.]), atol=1e-6, rtol=0)

    def test_zero_heads_equal_shared_embedding(self):
        model = Tiny()
        for h in model.experts:
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)
        image = torch.randn(6, 8)
        p = torch.softmax(torch.randn(6, 3), -1)
        v, r = model.fuse(image, p)
        self.assertEqual(int(torch.count_nonzero(r)), 0)
        torch.testing.assert_close(v, torch.nn.functional.normalize(image, dim=-1, eps=1e-12), rtol=0, atol=0)

    def test_shared_gradient_and_head_only_decorrelation(self):
        model = Tiny()
        h, support = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
        p = torch.eye(3)[torch.tensor([0, 1, 2, 0])]
        batch = {"view_prob": p, "support_prob": p.roll(1, 1)}
        shared, dec, _ = model.auxiliary(h, support, batch, torch.ones(4, dtype=torch.bool))
        dec.backward()
        self.assertIsNone(h.grad)
        self.assertIsNone(support.grad)
        self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0 for x in model.experts.parameters()))
        shared.backward()
        self.assertGreater(float(h.grad.norm()), 0)
        self.assertGreater(float(support.grad.norm()), 0)

    def test_zero_pairs_connected_backward(self):
        model = Tiny()
        h = torch.randn(4, 8, requires_grad=True)
        batch = {"view_prob": torch.ones(4, 3) / 3, "support_prob": torch.ones(4, 3) / 3}
        shared, dec, stats = model.auxiliary(h, h[:0], batch, torch.zeros(4, dtype=torch.bool))
        (shared + dec).backward()
        self.assertEqual(float(shared + dec), 0)
        self.assertIsNotNone(h.grad)
        self.assertEqual(int(stats["support_pairs"]), 0)

    def test_support_identity_distinct_image_and_view(self):
        rows = [dict(person_id=p, image_id=i, view3_raw=view3(t), theta_raw=t)
                for p, i, t in [(0, 0, 180), (0, 1, 0), (0, 2, 190), (1, 3, 0)]]
        self.assertEqual(support_candidates(rows), [[1], [0, 2], [1], []])

    def test_main_plan_and_rng_unchanged(self):
        class Dataset:
            pairs = [(0, 0), (0, 1), (1, 0), (2, 0)]
            candidates = [[1], [0], []]
        indices = np.array([[[0, 1, 2, 3]], [[3, 2, 1, 0]]])
        ordinary = list(PlanBatchSampler(indices, 1, 1, 0))
        augmented = list(SupportSampler(indices, 1, 1, 0, Dataset(), 2))
        self.assertEqual([[x[:2] for x in batch] for batch in augmented], ordinary)
        self.assertTrue(all(sum(x[2] >= 0 for x in batch) <= 2 for batch in augmented))
        self.assertEqual(augmented, list(SupportSampler(indices, 1, 1, 0, Dataset(), 2)))


if __name__ == "__main__":
    unittest.main()
