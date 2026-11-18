"""Chronological splits, purging, walk-forward folds and evaluation metrics.

Changes against the original ``main6.py``:

* The original used one 80/20 train/test split (L208-210) and then chose the best
  model (L425, L1351) and fitted ensemble weights (L1320) on the test set. Here
  every split has three disjoint, time-ordered parts: train (model fitting),
  validation (model selection and ensemble weights), test (reported once).
* For an h-day target, the label of row t uses the price at t+h. Training rows whose
  label window reaches into a later segment are purged: a gap of ``h - 1`` rows is
  left between train and validation and between validation and test.
* ``walk_forward_splits`` generalises the original ``walk_forward_validation``
  (defined at L1405 but never called) to expanding windows with an inner
  validation block, so selection can be repeated inside every fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def purge_gap(horizon: int) -> int:
    """Rows to drop between consecutive segments for an h-step-ahead target."""
    return max(int(horizon) - 1, 0)


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def check(self, gap: int = 0) -> None:
        """Raise if the segments are empty, unordered, overlapping or under-purged."""
        for name, idx in (("train", self.train), ("val", self.val), ("test", self.test)):
            if len(idx) == 0:
                raise ValueError(f"empty {name} segment")
            if np.any(np.diff(idx) != 1):
                raise ValueError(f"{name} segment is not contiguous and increasing")
        if not (self.train[-1] + gap < self.val[0] and self.val[-1] + gap < self.test[0]):
            raise ValueError("segments are not chronological with the required gap")


def chronological_split(
    n: int, train_frac: float = 0.6, val_frac: float = 0.2, gap: int = 0
) -> Split:
    """Train / validation / test by position, oldest first; test = the remainder."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    split = Split(
        train=np.arange(0, train_end - gap),
        val=np.arange(train_end, val_end - gap),
        test=np.arange(val_end, n),
    )
    split.check(gap)
    return split


def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    initial_train_frac: float = 0.6,
    val_frac: float = 0.2,
    gap: int = 0,
) -> list[Split]:
    """Expanding-window walk-forward folds.

    The history before ``initial = n * initial_train_frac`` is the first window;
    the rest is cut into ``n_folds`` consecutive test blocks (the last block takes
    any remainder). For each fold, the last ``val_frac`` of the window before the
    test block is the inner validation block and the part before it is training.
    Test blocks are disjoint and in time order.
    """
    initial = int(n * initial_train_frac)
    step = (n - initial) // n_folds
    if step < 1:
        raise ValueError("not enough rows for the requested number of folds")
    folds = []
    for k in range(n_folds):
        window_end = initial + k * step
        test_end = n if k == n_folds - 1 else window_end + step
        val_start = window_end - int(window_end * val_frac)
        split = Split(
            train=np.arange(0, val_start - gap),
            val=np.arange(val_start, window_end - gap),
            test=np.arange(window_end, test_end),
        )
        split.check(gap)
        folds.append(split)
    return folds


def inner_cv(n_splits: int = 5, gap: int = 0) -> TimeSeriesSplit:
    """Time-series CV used for hyperparameter search inside the training segment."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def regression_metrics(y_true, y_pred) -> dict:
    """Metrics used throughout. Direction accuracy compares signs (> 0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    ss0 = float(np.sum(y_true**2))
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # r2 relative to the (unknowable) mean of the evaluation window itself
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        # out-of-sample R^2 relative to the zero-return forecast
        "r2_vs_zero": float(1.0 - sse / ss0) if ss0 > 0 else float("nan"),
        "direction_accuracy": float(np.mean((y_true > 0) == (y_pred > 0))),
        # share of up moves: what an "always up" call would score
        "up_share": float(np.mean(y_true > 0)),
    }
