import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from run_cuhk_e0 import verify
from view4.config import lr_value, loss_weights
from view4.prepare import EXPECTED
from view4.sampling import e0_plan
from view4.audit_status import require_optimizer_updates


class CUHKTests(unittest.TestCase):
    def test_audit_rejects_all_skipped_updates(self):
        with self.assertRaises(RuntimeError):
            require_optimizer_updates([{'skipped': True}] * 8)
        with self.assertRaises(RuntimeError):
            require_optimizer_updates([])
        self.assertEqual(require_optimizer_updates([{'skipped': True}] * 4 + [{'skipped': False}] * 4), 4)

    def test_counts_and_main_plan(self):
        self.assertEqual(EXPECTED, {'train': (34054,11003,68126), 'val': (3078,1000,6158), 'test': (3074,1000,6156)})
        plan = e0_plan(68126, 1, steps=212)
        self.assertEqual(plan.shape, (212,4,80))
        self.assertEqual(np.unique(plan).size, 67840)
        self.assertGreaterEqual(int(plan.min()), 0)
        self.assertLess(int(plan.max()), 68126)
        np.testing.assert_array_equal(plan, e0_plan(68126, 1, steps=212))

    def test_e0_epoch_scaled_schedule(self):
        self.assertAlmostEqual(lr_value(211, 'E0', 5, 212), 1e-4)
        self.assertAlmostEqual(loss_weights(2, 0, 'E0', 212)[0], .5)
        for e in range(1,6):
            for s in range(212):
                index = (e-1)*212+s
                self.assertGreater(lr_value(index,'E0',5,212), 0)
                self.assertEqual(loss_weights(e,s,'E0',212)[1], 0)
        with self.assertRaises(ValueError):
            lr_value(1060,'E0',5,212)

    def test_baseline_verification_rejects_test_and_bad_steps(self):
        rows = [{'epoch':e,'batches':212,'validation':{'r1':50.+e,'mAP':40.+e}} for e in range(1,6)]
        result = dict(experiment='E0',seed=1,epochs=5,status='complete',config={'dataset':'CUHK-PEDES','input_resolution':[384,128]},
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
    def test_continuation_requires_own_measured_e0(self):
        from research.runner import load_config
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root/'configs/research_v011.json').read_text())
        baseline = dict(status='complete',experiment='E0',seed=1,epochs=5,
                        config={'dataset':'CUHK-PEDES','input_resolution':[384,128]},test_at_best_validation=None,
                        best_validation={'r1':61.,'mAP':48.})
        with patch('research.runner.read_json', side_effect=[copy.deepcopy(cfg),baseline]):
            self.assertEqual(load_config('unused')['baseline_validation'], {'r1':61.,'mAP':48.})
        baseline['config']['dataset'] = 'RSTPReid'
        with patch('research.runner.read_json', side_effect=[copy.deepcopy(cfg),baseline]):
            with self.assertRaises(ValueError):
                load_config('unused')


if __name__ == '__main__':
    unittest.main()
