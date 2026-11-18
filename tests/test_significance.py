import math
import unittest

import numpy as np
from scipy import stats

from fints.significance import diebold_mariano


class DieboldMarianoTests(unittest.TestCase):
    def test_matches_hand_computation_for_h1(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        a = y + np.array([0.1, -0.2, 0.1, 0.3, -0.1])
        b = y + np.array([0.5, 0.4, -0.6, 0.2, 0.7])
        d = (y - a) ** 2 - (y - b) ** 2
        n = len(d)
        var = np.mean((d - d.mean()) ** 2)          # gamma_0, divisor n
        dm = d.mean() / math.sqrt(var / n)
        stat = dm * math.sqrt((n + 1 - 2) / n)     # HLN with h = 1
        p = 2 * stats.t.sf(abs(stat), n - 1)
        r = diebold_mariano(y, a, b, horizon=1)
        self.assertAlmostEqual(r["statistic"], stat, places=12)
        self.assertAlmostEqual(r["p_value"], p, places=12)
        self.assertLess(r["statistic"], 0)          # a has smaller errors

    def test_clearly_better_forecast_is_significant(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=500)
        good = y + rng.normal(scale=0.3, size=500)
        bad = y + rng.normal(scale=1.0, size=500)
        r = diebold_mariano(y, good, bad, horizon=1)
        self.assertLess(r["statistic"], 0)
        self.assertLess(r["p_value"], 1e-6)

    def test_equal_accuracy_rejects_at_about_nominal_rate(self):
        rng = np.random.default_rng(1)
        rejections = 0
        trials = 400
        for _ in range(trials):
            y = rng.normal(size=250)
            a = y + rng.normal(size=250)
            b = y + rng.normal(size=250)
            rejections += diebold_mariano(y, a, b)["p_value"] < 0.05
        self.assertGreater(rejections / trials, 0.01)
        self.assertLess(rejections / trials, 0.10)

    def test_multi_step_uses_autocovariances(self):
        rng = np.random.default_rng(2)
        e = rng.normal(size=401)
        ma = e[1:] + 0.9 * e[:-1]                   # MA(1) loss pattern, as for h = 2
        y = np.zeros(400)
        r1 = diebold_mariano(y, ma * 0.5, np.full(400, 0.3), horizon=1)
        r2 = diebold_mariano(y, ma * 0.5, np.full(400, 0.3), horizon=2)
        self.assertNotAlmostEqual(r1["statistic"], r2["statistic"], places=6)

    def test_identical_forecasts_give_nan(self):
        y = np.arange(10.0)
        r = diebold_mariano(y, y + 1, y + 1)
        self.assertTrue(math.isnan(r["statistic"]))
        self.assertIn("note", r)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            diebold_mariano([1, 2, 3], [1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            diebold_mariano([1.0] * 10, [1.0] * 10, [0.0] * 10, horizon=0)
        with self.assertRaises(ValueError):
            diebold_mariano([1.0] * 4, [0.0] * 4, [2.0] * 4, horizon=2)


if __name__ == "__main__":
    unittest.main()
