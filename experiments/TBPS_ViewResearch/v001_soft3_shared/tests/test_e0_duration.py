"""Duration-only extension: retain the five-epoch schedule and E0 semantics."""

import math
import tempfile
import unittest
from pathlib import Path

import yaml

from view4.common import ROOT
from view4.config import load_config, lr_value, official_config


class E0DurationTests(unittest.TestCase):
    def test_legacy_schedule_is_unchanged(self):
        for step in range(1060):
            expected = (1e-6 + (1e-4 - 1e-6) * step / 211 if step < 212 else
                        5e-6 + .5 * (1e-4 - 5e-6) * (1 + math.cos(math.pi * (step - 212) / 848)))
            self.assertEqual(lr_value(step, 'E0'), expected)

    def test_ten_epoch_schedule_never_restarts_after_warmup(self):
        values = [lr_value(step, 'E0', 10) for step in range(2120)]
        self.assertEqual(values[0], 1e-6)
        self.assertAlmostEqual(values[211], 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(values[212:], values[213:])))
        self.assertGreater(values[1059], 5e-5)
        self.assertAlmostEqual(values[-1], 5e-6, delta=1e-9)
        with self.assertRaises(ValueError):
            lr_value(2120, 'E0', 10)

    def test_ten_epoch_profile_keeps_s_losses(self):
        cfg = load_config(ROOT / 'configs/cuhk_e0_10e.yaml')
        self.assertEqual(cfg['epochs'], 10)
        official = official_config(cfg)
        self.assertEqual(official.schedule.epoch, 10)
        self.assertEqual(official.experiment.nitc_ratio, 1.)
        self.assertTrue(official.experiment.ritc)
        self.assertEqual(official.experiment.ritc_ratio, 1.)
        for name in ('ss', 'citc', 'mlm', 'id', 'mvs_image', 'mixgen'):
            self.assertFalse(official.experiment[name], name)

    def test_unsupported_duration_is_rejected(self):
        cfg = yaml.safe_load((ROOT / 'configs/cuhk_e0_10e.yaml').read_text())
        cfg['epochs'] = 7
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == '__main__':
    unittest.main()
