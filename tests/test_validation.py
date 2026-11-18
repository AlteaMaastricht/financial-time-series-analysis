import unittest

import numpy as np

from tests import _path  # noqa: F401
from fints.validation import (
    Split, chronological_split, inner_cv, purge_gap, regression_metrics, walk_forward_splits,
)


class TestSplits(unittest.TestCase):
    def assert_chronological(self, s: Split, gap: int):
        for part in (s.train, s.val, s.test):
            self.assertGreater(len(part), 0)
            self.assertTrue(np.all(np.diff(part) == 1))
        self.assertEqual(len(np.intersect1d(s.train, s.val)), 0)
        self.assertEqual(len(np.intersect1d(s.val, s.test)), 0)
        self.assertEqual(len(np.intersect1d(s.train, s.test)), 0)
        # purged: label window (t, t+h] of the last train/val row ends before the next segment
        h = gap + 1
        self.assertLessEqual(s.train[-1] + h, s.val[0])
        self.assertLessEqual(s.val[-1] + h, s.test[0])

    def test_purge_gap(self):
        self.assertEqual(purge_gap(1), 0)
        self.assertEqual(purge_gap(5), 4)

    def test_holdout_split_is_chronological_and_disjoint(self):
        for h in (1, 5):
            s = chronological_split(1000, 0.6, 0.2, purge_gap(h))
            self.assert_chronological(s, purge_gap(h))
            self.assertEqual(s.test[-1], 999)
            self.assertEqual(s.val[0], 600)
            self.assertEqual(s.test[0], 800)

    def test_walk_forward_folds_ordered(self):
        for h in (1, 5):
            folds = walk_forward_splits(1000, 5, 0.6, 0.2, purge_gap(h))
            self.assertEqual(len(folds), 5)
            for f in folds:
                self.assert_chronological(f, purge_gap(h))
            starts = [f.test[0] for f in folds]
            self.assertEqual(starts, sorted(starts))
            for a, b in zip(folds, folds[1:]):
                self.assertEqual(a.test[-1] + 1, b.test[0])          # consecutive, disjoint
                self.assertGreaterEqual(len(b.train), len(a.train))  # expanding window
            self.assertEqual(folds[0].test[0], 600)
            self.assertEqual(folds[-1].test[-1], 999)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            chronological_split(100, 0.8, 0.3)
        with self.assertRaises(ValueError):
            Split(np.arange(0, 10), np.arange(5, 15), np.arange(20, 30)).check()

    def test_inner_cv_is_forward_only(self):
        for tr, te in inner_cv(5, gap=4).split(np.zeros((300, 1))):
            self.assertLess(tr.max() + 4, te.min())

    def test_metrics(self):
        y = np.array([0.01, -0.02, 0.03, -0.01])
        m = regression_metrics(y, np.zeros(4))
        self.assertAlmostEqual(m["r2_vs_zero"], 0.0)
        self.assertAlmostEqual(m["up_share"], 0.5)
        self.assertAlmostEqual(m["direction_accuracy"], 0.5)  # zero forecast calls "not up"
        self.assertAlmostEqual(regression_metrics(y, y)["r2_vs_zero"], 1.0)


if __name__ == "__main__":
    unittest.main()
