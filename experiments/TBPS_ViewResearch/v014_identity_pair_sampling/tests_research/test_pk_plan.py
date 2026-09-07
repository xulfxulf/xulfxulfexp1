import unittest
from types import SimpleNamespace

import numpy as np

from research.data import validate_pk_plan
from view4.sampling import pk_plan


class PKPlanTests(unittest.TestCase):
    def setUp(self):
        self.images = [dict(person_id=pid, image_id=2*pid+i, theta_raw=180*i,
                            view_raw=2 if i == 0 else 0, captions=['one', 'two'])
                       for pid in range(4) for i in range(2)]
        pairs = [(i,j) for i in range(len(self.images)) for j in range(2)]
        self.dataset = SimpleNamespace(images=self.images, pairs=pairs)
        self.plan, _ = pk_plan(self.images, 1, 1, steps=3, world_size=2, p=4, k=2)

    def verify(self, plan=None):
        return validate_pk_plan(self.plan if plan is None else plan, self.dataset, 3, 2, 4, 4)

    def test_generated_plan_and_repeatability(self):
        result = self.verify()
        self.assertEqual(result['repeated_single_image_pairs'], 0)
        self.assertEqual(result['identities_per_global_batch'], 4)
        again, _ = pk_plan(self.images, 1, 1, steps=3, world_size=2, p=4, k=2)
        np.testing.assert_array_equal(self.plan, again)

    def test_duplicate_caption_does_not_substitute_for_distinct_image(self):
        bad = self.plan.copy()
        bad[0,0,1] = bad[0,0,0] ^ 1
        with self.assertRaisesRegex(ValueError, 'Repeated image'):
            self.verify(bad)

    def test_rank_splitting_rejected(self):
        bad = self.plan.copy()
        bad[0,0,1], bad[0,1,1] = bad[0,1,1], bad[0,0,1]
        with self.assertRaisesRegex(ValueError, 'split across ranks'):
            self.verify(bad)

    def test_invalid_indices_and_wrong_pid_counts(self):
        for change in ('float', 'range', 'identity'):
            bad = self.plan.copy()
            if change == 'float':
                bad = bad.astype(float)
            elif change == 'range':
                bad[0,0,0] = len(self.dataset.pairs)
            else:
                bad[0,0,0] = bad[0,0,2]
            with self.assertRaises(ValueError):
                self.verify(bad)

    def test_single_image_pid_stays_eligible(self):
        images = self.images[1:]
        dataset = SimpleNamespace(images=images,
            pairs=[(i,j) for i in range(len(images)) for j in range(2)])
        plan, _ = pk_plan(images, 1, 1, steps=3, world_size=2, p=4, k=2)
        result = validate_pk_plan(plan, dataset, 3, 2, 4, 4)
        self.assertEqual(result['repeated_single_image_pairs'], 3)


if __name__ == '__main__':
    unittest.main()
