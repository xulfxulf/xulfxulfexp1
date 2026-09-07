import json
from pathlib import Path
import unittest

import numpy as np
import torch

from run_flip_evaluation import average_embeddings, guard_plain, validate_spec
from research.data import aggregate_probabilities


class FlipTests(unittest.TestCase):
    def test_unit_length_and_symmetry(self):
        a = torch.tensor([[1., 0.], [0., 1.]])
        b = torch.tensor([[0., 1.], [1., 0.]])
        output = average_embeddings(a, b)
        self.assertTrue(torch.equal(output, average_embeddings(b, a)))
        self.assertTrue(torch.allclose(output.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.equal(average_embeddings(a, a), a))

    def test_reject_degenerate_or_changed_plain(self):
        with self.assertRaises(FloatingPointError):
            average_embeddings(torch.ones(2, 3), -torch.ones(2, 3))
        with self.assertRaises(RuntimeError):
            guard_plain({'r1': 70., 'mAP': 60.}, {'r1': 71., 'mAP': 60.}, .0001)

    def test_soft_three_views_are_mirror_invariant(self):
        for angle_bin in range(72):
            original, flipped = np.zeros(72), np.zeros(72)
            original[angle_bin] = 1.
            flipped[(-angle_bin) % 72] = 1.
            np.testing.assert_array_equal(aggregate_probabilities(original), aggregate_probabilities(flipped))

    def test_no_test_switch(self):
        with open(Path(__file__).resolve().parents[1]/'configs/evaluation_v008.json') as stream:
            spec = json.load(stream)
        validate_spec(spec)
        spec['split'] = 'test'
        with self.assertRaises(ValueError):
            validate_spec(spec)


if __name__ == '__main__':
    unittest.main()
