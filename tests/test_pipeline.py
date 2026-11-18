import inspect
import unittest

import numpy as np
import pandas as pd

from tests import _path  # noqa: F401
from fints.baselines import baseline_predictions
from fints.data import generate_synthetic_prices
from fints.features import build_dataset
from fints.models import TimeSeriesStackingRegressor, fit_and_select, fit_ensemble_weights
from fints.pipeline import ExperimentConfig, run_fold, run_holdout, run_walk_forward
from fints.validation import chronological_split, purge_gap

FAST = ExperimentConfig(model_names=("ridge", "extra_trees"), n_iter=0, top_k=20,
                        use_stacking=True, seed=0, n_jobs=1, n_folds=3)


class TestBaselines(unittest.TestCase):
    def test_values(self):
        past = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
        b = baseline_predictions(past, y_train=[0.01, 0.03], n_eval=2, eval_idx=[3, 4])
        np.testing.assert_array_equal(b["zero"], [0, 0])
        np.testing.assert_array_equal(b["previous_return"], [0.4, -0.5])
        np.testing.assert_allclose(b["historical_mean"], [0.02, 0.02])

    def test_previous_return_is_lagged_target(self):
        ds = build_dataset(generate_synthetic_prices(600, seed=3), horizon=1)
        # previous 1-day return at t equals the 1-day target at t-1
        np.testing.assert_allclose(ds.past_return.to_numpy()[1:], ds.y.to_numpy()[:-1])


class TestNoTestLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = build_dataset(generate_synthetic_prices(700, seed=11), horizon=1)
        cls.split = chronological_split(len(cls.ds), 0.6, 0.2, purge_gap(1))
        cls.fold = run_fold(cls.ds, cls.split, FAST)

    def test_selection_api_has_no_test_arguments(self):
        params = inspect.signature(fit_and_select).parameters
        self.assertFalse(any("test" in p for p in params), list(params))

    def test_scaler_fitted_on_train_rows_only(self):
        X = self.ds.X.to_numpy(dtype=float)
        np.testing.assert_allclose(self.fold.scaler.mean_, X[self.split.train].mean(axis=0))
        np.testing.assert_allclose(self.fold.scaler.var_, X[self.split.train].var(axis=0))
        self.assertEqual(self.fold.scaler.n_samples_seen_, len(self.split.train))

    def test_selection_unchanged_when_test_rows_are_corrupted(self):
        X = self.ds.X.copy()
        y = self.ds.y.copy()
        test_rows = self.ds.y.index[self.split.test]
        rng = np.random.default_rng(0)
        X.loc[test_rows] = rng.normal(0, 1e3, size=(len(test_rows), X.shape[1]))
        y.loc[test_rows] = rng.normal(0, 10, size=len(test_rows))
        corrupted = type(self.ds)(X=X, y=y, past_return=self.ds.past_return, horizon=1)
        fold2 = run_fold(corrupted, self.split, FAST)
        self.assertEqual(fold2.selected, self.fold.selected)
        np.testing.assert_allclose(fold2.selection.weights, self.fold.selection.weights)
        self.assertEqual(fold2.selection.val_mse, self.fold.selection.val_mse)
        np.testing.assert_allclose(fold2.scaler.mean_, self.fold.scaler.mean_)

    def test_selected_is_validation_argmin(self):
        vm = self.fold.selection.val_mse
        self.assertEqual(self.fold.selected, min(vm, key=vm.get))
        self.assertIn("baseline:zero", self.fold.test_metrics)
        np.testing.assert_array_equal(self.fold.test_predictions["selected"],
                                      self.fold.test_predictions[f"model:{self.fold.selected}"])


class TestEnsembles(unittest.TestCase):
    def test_weights_move_away_from_equal_on_return_scale(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.01, 300)
        good, bad = y + rng.normal(0, 0.005, 300), rng.normal(0, 0.03, 300)
        w = fit_ensemble_weights(np.column_stack([good, bad]), y)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertGreater(w[0], 0.9)

    def test_stacking_meta_features_are_forward_only(self):
        from sklearn.linear_model import Ridge
        X = np.arange(200, dtype=float).reshape(-1, 1)
        y = X.ravel() * 0.1
        st = TimeSeriesStackingRegressor([("r", Ridge())], Ridge(), n_splits=4).fit(X, y)
        # first block never receives an out-of-fold prediction (nothing earlier to train on)
        self.assertFalse(st.oof_mask_[:40].any())
        self.assertTrue(st.oof_mask_[-40:].all())


class TestWalkForwardAndPositiveControl(unittest.TestCase):
    def test_walk_forward_and_planted_signal(self):
        # Positive control: with a planted AR(1) coefficient and a small relevant
        # feature subset, the split/selection/evaluation machinery must beat the zero
        # forecast out of sample. (With the full ~59-feature set, Ridge overfits on
        # 900 rows and does not recover the signal; see README "Limitations".)
        prices = generate_synthetic_prices(900, seed=5, ar_coef=0.4)
        full = build_dataset(prices, horizon=1)
        cols = ["returns", "return_lag_1", "momentum_5"]
        ds = type(full)(X=full.X[cols], y=full.y, past_return=full.past_return, horizon=1)
        cfg = ExperimentConfig(model_names=("ridge",), top_k=0, use_stacking=False,
                               seed=0, n_jobs=1, n_folds=3)
        wf = run_walk_forward(ds, cfg)
        idx = [f.test_index for f in wf.folds]
        for a, b in zip(idx, idx[1:]):
            self.assertLess(a[-1], b[0])
        self.assertTrue(pd.Index(wf.index).is_monotonic_increasing)
        self.assertEqual(len(wf.y), sum(len(f.split.test) for f in wf.folds))
        self.assertGreater(wf.metrics["selected"]["r2_vs_zero"], 0.05)
        self.assertGreater(wf.metrics["selected"]["r2_vs_zero"],
                           wf.metrics["baseline:historical_mean"]["r2_vs_zero"])
        ho = run_holdout(ds, cfg)
        self.assertGreater(ho.test_metrics["selected"]["r2_vs_zero"], 0.05)


if __name__ == "__main__":
    unittest.main()
