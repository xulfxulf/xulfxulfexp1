import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.name == 'posix', 'Pinned training paths and fcntl are Linux-specific')
class AllocationTests(unittest.TestCase):
    def test_peak_group_rates_and_unchanged_auxiliary_schedule(self):
        from research.runner import load_config, schedule
        cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/research_v003.json')
        lr, _, _, dec = schedule(cfg, 1, 211)
        self.assertAlmostEqual(lr, 1e-6)
        self.assertAlmostEqual(lr * cfg['expert_lr_peak'] / cfg['base_lr_peak'], 1e-3)
        self.assertEqual(dec, 0.)
        _, _, shared, dec = schedule(cfg, 2, 211)
        self.assertEqual(shared, .4)
        self.assertAlmostEqual(dec, .0003)
        self.assertEqual(cfg['seed'], 1)
        self.assertEqual(cfg['local_batch'] * cfg['world_size'], 320)
