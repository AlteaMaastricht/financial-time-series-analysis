import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import _path  # noqa: F401
from fints.backtest import backtest, threshold_positions
from fints.data import generate_synthetic_prices, load_price_csv
from fints.dl import make_sequences, sequence_split
from fints.stats import (
    garch_11, hurst_rs, interpret_hurst, ljung_box, markov_switching, pelt_change_points,
    regime_separation,
)


class TestBacktest(unittest.TestCase):
    def test_positions(self):
        np.testing.assert_array_equal(threshold_positions([0.02, -0.02, 0.005]), [1, -1, 0])

    def test_running_peak_drawdown_and_costs(self):
        realized = np.array([0.10, -0.50, 0.10])
        bt = backtest(np.full(3, 0.05), realized, threshold=0.01, cost=0.0)
        # equity 1.1 -> 0.55 -> 0.605; running-peak drawdown = 0.55/1.1 - 1 = -0.5
        self.assertAlmostEqual(bt["strategy"]["max_drawdown"], -0.5)
        self.assertAlmostEqual(bt["strategy"]["total_return"], 1.1 * 0.5 * 1.1 - 1)
        flip = backtest(np.array([0.05, -0.05]), np.zeros(2), cost=0.001)
        # enter long (1 unit) then flip to short (2 units)
        self.assertAlmostEqual(flip["strategy"]["total_return"], (1 - 0.001) * (1 - 0.002) - 1)
        self.assertEqual(flip["strategy"]["position_changes"], 2)

    def test_flat_forecast_holds_cash(self):
        bt = backtest(np.zeros(5), np.full(5, 0.01))
        self.assertEqual(bt["strategy"]["total_return"], 0.0)
        self.assertGreater(bt["buy_and_hold"]["total_return"], 0.0)


class TestStats(unittest.TestCase):
    def test_hurst_iid_returns_near_half(self):
        r = np.random.default_rng(0).standard_normal(4000)
        self.assertTrue(0.4 < hurst_rs(r) < 0.65)

    def test_hurst_interpretation_flags_invalid(self):
        self.assertTrue(interpret_hurst(1.0532).startswith("invalid"))
        self.assertTrue(interpret_hurst(0.5).startswith("close"))

    def test_regime_separation_detects_flat_probabilities(self):
        flat = np.full((500, 2), 0.5)
        self.assertFalse(regime_separation(flat)["separated"])
        sharp = np.zeros((500, 2))
        sharp[:250, 0] = 1
        sharp[250:, 1] = 1
        self.assertTrue(regime_separation(sharp)["separated"])

    def test_ljung_box(self):
        rng = np.random.default_rng(1)
        e = rng.standard_normal(2000)
        ar = np.zeros(2000)
        for t in range(1, 2000):
            ar[t] = 0.5 * ar[t - 1] + e[t]
        self.assertGreater(ljung_box(e, 10)["p_value"], 0.01)
        self.assertLess(ljung_box(ar, 10)["p_value"], 1e-6)

    @unittest.skipIf(importlib.util.find_spec("arch") is not None, "arch installed")
    def test_garch_degrades_gracefully(self):
        self.assertFalse(garch_11(np.zeros(10))["available"])

    @unittest.skipIf(importlib.util.find_spec("statsmodels") is not None, "statsmodels installed")
    def test_markov_degrades_gracefully(self):
        self.assertFalse(markov_switching(np.zeros(10))["available"])


class TestOptionalStatsWhenInstalled(unittest.TestCase):
    """Fit the optional models on series with a planted, known structure.

    These run only when the optional package is installed; the core environment
    runs the graceful-degradation tests above instead.
    """

    @unittest.skipIf(importlib.util.find_spec("arch") is None, "arch not installed")
    def test_garch_recovers_planted_persistence(self):
        rng = np.random.default_rng(0)
        omega, alpha, beta, n = 0.05, 0.10, 0.85, 4000  # percent units
        var, r = omega / (1 - alpha - beta), np.empty(n)
        for t in range(n):
            r[t] = np.sqrt(var) * rng.standard_normal()
            var = omega + alpha * r[t] ** 2 + beta * var
        out = garch_11(pd.Series(r / 100))  # garch_11 expects decimal returns
        self.assertTrue(out["available"])
        self.assertAlmostEqual(out["persistence"], alpha + beta, delta=0.03)
        self.assertAlmostEqual(out["alpha"], alpha, delta=0.05)

    @unittest.skipIf(importlib.util.find_spec("statsmodels") is None, "statsmodels not installed")
    def test_markov_separates_planted_variance_regimes(self):
        rng = np.random.default_rng(1)
        sd = np.r_[np.full(500, 0.005), np.full(500, 0.03), np.full(500, 0.005)]
        out = markov_switching(pd.Series(rng.standard_normal(sd.size) * sd))
        self.assertTrue(out["available"])
        self.assertTrue(out["separated"])
        v = sorted(out["regime_variances"])
        self.assertGreater(v[1] / v[0], 10)  # planted ratio is 36

    @unittest.skipIf(importlib.util.find_spec("ruptures") is None, "ruptures not installed")
    def test_pelt_finds_planted_break(self):
        rng = np.random.default_rng(2)
        sd = np.r_[np.full(300, 0.01), np.full(300, 0.05)]
        idx = pd.date_range("2020-01-01", periods=sd.size, freq="B")
        out = pelt_change_points(pd.Series(rng.standard_normal(sd.size) * sd, index=idx))
        self.assertTrue(out["available"])
        self.assertTrue(any(abs(i - 300) <= 15 for i in out["indices"]), out["indices"])
        self.assertEqual(len(out["indices"]), len(out["dates"]))


class TestSequences(unittest.TestCase):
    def test_window_label_alignment(self):
        feats = np.arange(20, dtype=float).reshape(10, 2)
        target = np.arange(10, dtype=float) * 10
        X, y, end = make_sequences(feats, target, seq_len=3)
        self.assertEqual(X.shape, (8, 3, 2))
        for j in range(len(y)):
            np.testing.assert_array_equal(X[j, -1], feats[end[j]])  # last row of window = t
            self.assertEqual(y[j], target[end[j]])                  # label = target at t
        # the original code labelled window ending at t with target[t + 1]
        self.assertEqual(y[0], target[2])

    def test_sequence_split_purged(self):
        s = sequence_split(500, horizon=5)
        self.assertLessEqual(s.train[-1] + 5, s.val[0])
        self.assertLessEqual(s.val[-1] + 5, s.test[0])


class TestData(unittest.TestCase):
    def test_generator_deterministic_and_consistent(self):
        a = generate_synthetic_prices(300, seed=9)
        b = generate_synthetic_prices(300, seed=9)
        c = generate_synthetic_prices(300, seed=10)
        self.assertTrue(a.equals(b))
        self.assertFalse(a.equals(c))
        self.assertTrue((a["High"] >= a[["Open", "Close"]].max(axis=1)).all())
        self.assertTrue((a["Low"] <= a[["Open", "Close"]].min(axis=1)).all())
        self.assertTrue((a["Volume"] > 0).all())
        self.assertNotIn("regime", a.columns)

    def test_csv_roundtrip_with_date_filter(self):
        a = generate_synthetic_prices(300, seed=9)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.csv"
            a.iloc[::-1].to_csv(p)  # unsorted on purpose
            b = load_price_csv(p, start=str(a.index[50].date()))
        self.assertTrue(b.index.is_monotonic_increasing)
        self.assertEqual(len(b), 250)
        np.testing.assert_allclose(b["Close"].to_numpy(), a["Close"].iloc[50:].to_numpy())


if __name__ == "__main__":
    unittest.main()
