import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.name == 'posix', 'Pinned training paths and fcntl are Linux-specific')
class AllocationTests(unittest.TestCase):
    def test_peak_group_rates_and_unchanged_auxiliary_schedule(self):
        from research.runner import schedule
        from view4.common import read_json
        cfg = read_json(Path(__file__).resolve().parents[1] / 'configs/research_v010.json')
        lr, _, _, dec = schedule(cfg, 1, 211)
        self.assertAlmostEqual(lr, 1e-6)
        self.assertAlmostEqual(lr * cfg['expert_lr_peak'] / cfg['base_lr_peak'], 1e-3)
        self.assertEqual(dec, 0.)
        _, _, shared, dec = schedule(cfg, 2, 211)
        self.assertEqual(shared, .4)
        self.assertAlmostEqual(dec, .0003)
        self.assertEqual(cfg['seed'], 1)
        self.assertEqual(cfg['local_batch'] * cfg['world_size'], 320)

    def test_beta_retained_and_auxiliary_warmups_unchanged(self):
        from research.runner import schedule
        from view4.common import read_json
        cfg = read_json(Path(__file__).resolve().parents[1] / 'configs/research_v010.json')
        for epoch in range(1, 6):
            for step in range(212):
                _, beta, shared, decor = schedule(cfg, epoch, step)
                self.assertEqual(beta, .5)
                self.assertAlmostEqual(shared, .4 * min(1., ((epoch-1)*212+step)/212))
                expected = 0. if epoch == 1 else .0003 * (step/211 if epoch == 2 else 1.)
                self.assertAlmostEqual(decor, expected)
        with self.assertRaises(ValueError):
            schedule(cfg, 6, 0)
