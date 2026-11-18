"""Hold-out and walk-forward experiments.

One fold = (train, validation, test) positions in a ``Dataset``:

1. ``StandardScaler`` is fitted on the training rows only (original L213-214 did
   this correctly; kept).
2. Feature ranking, model fitting and hyperparameter search use training rows only.
3. Ensemble weights and the choice of model use validation rows only
   (``models.fit_and_select`` has no test arguments).
4. The test rows are predicted once, by every candidate and by the naive baselines.

``run_walk_forward`` repeats this over expanding windows and reports the
concatenated out-of-sample predictions of each fold's selected model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .baselines import baseline_predictions
from .features import Dataset
from .models import DEFAULT_MODELS, SelectionResult, fit_and_select
from .validation import (
    Split,
    chronological_split,
    purge_gap,
    regression_metrics,
    walk_forward_splits,
)


@dataclass
class ExperimentConfig:
    model_names: tuple = DEFAULT_MODELS
    n_iter: int = 0
    top_k: int = 40
    use_stacking: bool = True
    seed: int = 42
    n_jobs: int = -1
    train_frac: float = 0.6
    val_frac: float = 0.2
    n_folds: int = 5
    initial_train_frac: float = 0.6


@dataclass
class FoldResult:
    split: Split
    scaler: StandardScaler
    selection: SelectionResult
    test_index: pd.DatetimeIndex
    y_test: np.ndarray
    test_predictions: dict  # candidate/baseline name -> np.ndarray
    test_metrics: dict      # name -> metrics dict

    @property
    def selected(self) -> str:
        return self.selection.selected


def run_fold(ds: Dataset, split: Split, cfg: ExperimentConfig) -> FoldResult:
    X = ds.X.to_numpy(dtype=float)
    y = ds.y.to_numpy(dtype=float)
    split.check(purge_gap(ds.horizon))

    scaler = StandardScaler().fit(X[split.train])
    Xs = scaler.transform(X)

    selection = fit_and_select(
        Xs[split.train], y[split.train], Xs[split.val], y[split.val], ds.feature_names,
        model_names=cfg.model_names, n_iter=cfg.n_iter, gap=purge_gap(ds.horizon),
        top_k=cfg.top_k, use_stacking=cfg.use_stacking, seed=cfg.seed, n_jobs=cfg.n_jobs,
    )

    # Test rows are touched only from here on.
    y_test = y[split.test]
    preds = selection.predict_all(Xs[split.test])
    preds = {f"model:{k}": v for k, v in preds.items()}
    preds["selected"] = preds[f"model:{selection.selected}"]
    for name, p in baseline_predictions(
        ds.past_return.to_numpy(), y[split.train], len(split.test), split.test
    ).items():
        preds[f"baseline:{name}"] = p
    metrics = {k: regression_metrics(y_test, v) for k, v in preds.items()}
    return FoldResult(split, scaler, selection, ds.y.index[split.test], y_test, preds, metrics)


def run_holdout(ds: Dataset, cfg: ExperimentConfig) -> FoldResult:
    split = chronological_split(len(ds), cfg.train_frac, cfg.val_frac, purge_gap(ds.horizon))
    return run_fold(ds, split, cfg)


@dataclass
class WalkForwardResult:
    folds: list
    index: pd.DatetimeIndex
    y: np.ndarray
    predictions: dict  # "selected" and baselines, concatenated over folds
    metrics: dict


def run_walk_forward(ds: Dataset, cfg: ExperimentConfig) -> WalkForwardResult:
    splits = walk_forward_splits(
        len(ds), cfg.n_folds, cfg.initial_train_frac, cfg.val_frac, purge_gap(ds.horizon)
    )
    folds = [run_fold(ds, s, cfg) for s in splits]
    keys = ["selected"] + [k for k in folds[0].test_predictions if k.startswith("baseline:")]
    preds = {k: np.concatenate([f.test_predictions[k] for f in folds]) for k in keys}
    y = np.concatenate([f.y_test for f in folds])
    index = folds[0].test_index.append([f.test_index for f in folds[1:]])
    metrics = {k: regression_metrics(y, v) for k, v in preds.items()}
    return WalkForwardResult(folds, index, y, preds, metrics)


def fold_summary(fold: FoldResult) -> dict:
    s = fold.split
    idx = fold.test_index
    return {
        "train_rows": int(len(s.train)),
        "val_rows": int(len(s.val)),
        "test_rows": int(len(s.test)),
        "test_start": str(idx[0].date()),
        "test_end": str(idx[-1].date()),
        "selected_model": fold.selected,
        "validation_mse": fold.selection.val_mse,
        "ensemble_weights": dict(zip(fold.selection.base_models, map(float, fold.selection.weights))),
        "hyperparameters": {k: {p: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                for p, v in d.items()}
                            for k, d in fold.selection.base_params.items()},
        "n_features_used": int(len(fold.selection.feature_idx)),
        "test_metrics": fold.test_metrics,
    }
