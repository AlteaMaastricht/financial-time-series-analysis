import unittest

import numpy as np

from tests import _path  # noqa: F401
from fints.data import generate_synthetic_prices
from fints.features import build_dataset, create_features, feature_columns


class TestNoLookAhead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = generate_synthetic_prices(n_days=420, seed=7)
        cls.full = create_features(cls.prices)
        cls.cols = feature_columns(cls.full)

    def test_feature_at_t_uses_only_data_up_to_t(self):
        # Recompute features on the history truncated at t: row t must be identical.
        for t in (210, 260, 333, 419):
            trunc = create_features(self.prices.iloc[: t + 1])
            a = self.full[self.cols].iloc[t].to_numpy(dtype=float)
            b = trunc[self.cols].iloc[t].to_numpy(dtype=float)
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True,
                                       err_msg=f"feature row {t} depends on future data")

    def test_future_shock_does_not_change_past_features(self):
        shocked = self.prices.copy()
        shocked.iloc[300:, :4] *= 1.5
        f2 = create_features(shocked)
        np.testing.assert_allclose(
            self.full[self.cols].iloc[:300].to_numpy(dtype=float),
            f2[self.cols].iloc[:300].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-12, equal_nan=True,
        )

    def test_targets_are_forward_returns_and_not_features(self):
        c = self.prices["Close"].to_numpy()
        np.testing.assert_allclose(self.full["target_1d"].to_numpy()[:-1], c[1:] / c[:-1] - 1)
        np.testing.assert_allclose(self.full["target_5d"].to_numpy()[:-5], c[5:] / c[:-5] - 1)
        self.assertTrue(np.isnan(self.full["target_1d"].iloc[-1]))
        for col in self.cols:
            self.assertFalse(col.startswith("target_"), col)
        for raw in ("Open", "High", "Low", "Close", "Volume"):
            self.assertNotIn(raw, self.cols)

    def test_stationary_feature_set_drops_levels(self):
        from fints.features import LEVEL_FEATURES
        stat = feature_columns(self.full, "stationary")
        self.assertTrue(set(stat) < set(self.cols))
        self.assertFalse(set(stat) & LEVEL_FEATURES)
        self.assertIn("returns", stat)

    def test_dataset_alignment(self):
        ds = build_dataset(self.prices, horizon=5)
        self.assertTrue(ds.X.index.equals(ds.y.index))
        self.assertFalse(ds.X.isna().any().any())
        self.assertTrue(ds.y.index.is_monotonic_increasing)
        c = self.prices["Close"]
        t = ds.y.index[10]
        pos = c.index.get_loc(t)
        self.assertAlmostEqual(ds.y.loc[t], c.iloc[pos + 5] / c.iloc[pos] - 1)
        self.assertAlmostEqual(ds.past_return.loc[t], c.iloc[pos] / c.iloc[pos - 5] - 1)


if __name__ == "__main__":
    unittest.main()
