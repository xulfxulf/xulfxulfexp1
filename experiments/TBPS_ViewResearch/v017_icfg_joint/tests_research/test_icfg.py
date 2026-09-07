import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from run_icfg_e0 import verify
from view4.config import lr_value, loss_weights
from view4.prepare import EXPECTED
from view4.sampling import e0_plan
from view4.audit_status import require_optimizer_updates


class ICFGTests(unittest.TestCase):
    def test_audit_rejects_all_skipped_updates(self):
        with self.assertRaises(RuntimeError):
            require_optimizer_updates([{'skipped': True}] * 8)
        with self.assertRaises(RuntimeError):
            require_optimizer_updates([])
        self.assertEqual(require_optimizer_updates([{'skipped': True}] * 4 + [{'skipped': False}] * 4), 4)

    def test_counts_and_main_plan(self):
        self.assertEqual(EXPECTED, {'train': (31289,2792,31289), 'val': (3383,310,3383), 'test': (19848,1000,19848)})
        plan = e0_plan(31289, 1, steps=97)
        self.assertEqual(plan.shape, (97,4,80))
        self.assertEqual(np.unique(plan).size, 31040)
        self.assertGreaterEqual(int(plan.min()), 0)
        self.assertLess(int(plan.max()), 31289)
        np.testing.assert_array_equal(plan, e0_plan(31289, 1, steps=97))

    def test_e0_epoch_scaled_schedule(self):
        self.assertAlmostEqual(lr_value(96, 'E0', 5, 97), 1e-4)
        self.assertAlmostEqual(loss_weights(2, 0, 'E0', 97)[0], .5)
        for e in range(1,6):
            for s in range(97):
                index = (e-1)*97+s
                self.assertGreater(lr_value(index,'E0',5,97), 0)
                self.assertEqual(loss_weights(e,s,'E0',97)[1], 0)
        with self.assertRaises(ValueError):
            lr_value(485,'E0',5,97)

    def test_baseline_verification_rejects_test_and_bad_steps(self):
        rows = [{'epoch':e,'batches':97,'validation':{'r1':50.+e,'mAP':40.+e}} for e in range(1,6)]
        result = dict(experiment='E0',seed=1,epochs=5,status='complete',config={'dataset':'ICFG-PEDES','input_resolution':[384,128]},
                      test_at_best_validation=None,history=rows,best_validation=dict(rows[-1]['validation'],epoch=5))
        self.assertEqual(verify(result)['guard'], 'passed')
        bad = copy.deepcopy(result)
        bad['test_at_best_validation'] = {'r1':99.,'mAP':99.}
        with self.assertRaises(RuntimeError):
            verify(bad)
        bad = copy.deepcopy(result)
        bad['history'][1]['batches'] = 115
        with self.assertRaises(RuntimeError):
            verify(bad)

    @unittest.skipUnless(os.name == 'posix', 'Linux runtime and absolute paths')
    def test_joint_training_requires_measured_reference_but_openai_initializer(self):
        from research.runner import load_config
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root/'configs/research_v017.json').read_text())
        baseline = dict(status='complete',experiment='E0',seed=1,epochs=5,
                        config={'dataset':'ICFG-PEDES','input_resolution':[384,128]},test_at_best_validation=None,
                        best_validation={'r1':61.,'mAP':48.})
        with patch('research.runner.read_json', side_effect=[copy.deepcopy(cfg),baseline]):
            self.assertEqual(load_config('unused')['baseline_validation'], {'r1':61.,'mAP':48.})
        bad = copy.deepcopy(cfg)
        bad['initializer'] = bad['baseline_checkpoint']
        with patch('research.runner.read_json', return_value=bad):
            with self.assertRaises(ValueError):
                load_config('unused')
        baseline['config']['dataset'] = 'RSTPReid'
        with patch('research.runner.read_json', side_effect=[copy.deepcopy(cfg),baseline]):
            with self.assertRaises(ValueError):
                load_config('unused')


if __name__ == '__main__':
    unittest.main()
