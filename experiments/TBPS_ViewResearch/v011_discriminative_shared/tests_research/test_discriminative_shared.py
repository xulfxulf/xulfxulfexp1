import unittest
import torch
import torch.nn.functional as F
from research.model import discriminative_shared_values


class DiscriminativeSharedTests(unittest.TestCase):
    def test_matches_explicit_symmetric_formula_and_ignores_same_pid(self):
        torch.manual_seed(51)
        a, b, bank = torch.randn(3, 8), torch.randn(3, 8), torch.randn(6, 8)
        ids, bank_ids = torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1, 2, 3, 4])
        actual = discriminative_shared_values(a, b, bank, ids, bank_ids, .07)
        an, bn, nn = [F.normalize(x, dim=-1, eps=1e-8) for x in (a, b, bank)]
        expected = []
        for i in range(3):
            pos = an[i].dot(bn[i])
            neg = nn[bank_ids != ids[i]]
            expected.append(.035 * sum(torch.logsumexp(torch.cat((torch.zeros(1),
                (neg @ query - pos)/.07)), 0) for query in (an[i], bn[i])))
        torch.testing.assert_close(actual, torch.stack(expected), atol=1e-7, rtol=1e-5)
        one = discriminative_shared_values(a[:1], b[:1], bank, ids[:1], bank_ids, .07)
        extra = discriminative_shared_values(a[:1], b[:1], torch.cat((bank, a[:1])),
                                             ids[:1], torch.cat((bank_ids, ids[:1])), .07)
        torch.testing.assert_close(one, extra, rtol=0, atol=0)

    def test_harder_different_pid_has_larger_penalty(self):
        a, b = torch.tensor([[1., 0.]]), torch.tensor([[.9, .1]])
        easy = discriminative_shared_values(a, b, torch.tensor([[0., 1.]]),
                                            torch.tensor([0]), torch.tensor([1]), .07)
        hard = discriminative_shared_values(a, b, a, torch.tensor([0]), torch.tensor([1]), .07)
        self.assertGreater(float(hard), float(easy))

    def test_all_same_pid_and_empty_pairs_have_connected_zero_gradients(self):
        a, b, bank = [torch.randn(n, 8, requires_grad=True) for n in (2, 2, 3)]
        value = discriminative_shared_values(a, b, bank, torch.zeros(2), torch.zeros(3), .07)
        torch.testing.assert_close(value, torch.zeros(2), rtol=0, atol=0)
        value.sum().backward()
        for x in (a, b, bank):
            self.assertIsNotNone(x.grad)
            self.assertEqual(float(x.grad.abs().sum()), 0.)
        empty = discriminative_shared_values(a[:0], b[:0], bank,
                                             torch.zeros(0), torch.zeros(3), .07)
        empty.sum().backward()
        self.assertEqual(tuple(empty.shape), (0,))

    def test_gradients_reach_anchors_supports_and_negative_bank(self):
        torch.manual_seed(52)
        a, b, bank = [torch.randn(n, 8, requires_grad=True) for n in (2, 2, 5)]
        value = discriminative_shared_values(a, b, bank, torch.tensor([0, 1]),
                                             torch.tensor([0, 1, 2, 3, 4]), .07).mean()
        value.backward()
        for x in (a, b, bank):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.norm()), 0.)

    def test_invalid_temperature_and_ids_fail(self):
        a = torch.randn(2, 8)
        for tau in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                discriminative_shared_values(a, a, a, torch.arange(2), torch.arange(2), tau)
        with self.assertRaises(ValueError):
            discriminative_shared_values(a, a, a, torch.arange(3), torch.arange(2), .07)


if __name__ == '__main__':
    unittest.main()
