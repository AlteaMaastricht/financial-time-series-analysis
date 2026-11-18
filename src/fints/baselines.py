"""Naive forecasts that every model must be compared against.

The original code compared models only to a 50% "random" line for direction
accuracy. That line is misleading in a rising market: predicting a constant small
positive return already scores the share of up days. These baselines are reported
next to every model:

* ``zero``             r_hat = 0 (random walk for prices)
* ``previous_return``  r_hat[t] = Close[t] / Close[t-h] - 1 (last observed h-day return)
* ``historical_mean``  r_hat = mean of the training targets (constant)
"""

from __future__ import annotations

import numpy as np

BASELINE_NAMES = ("zero", "previous_return", "historical_mean")


def baseline_predictions(past_return, y_train, n_eval: int, eval_idx=None) -> dict[str, np.ndarray]:
    """Return baseline forecasts for the evaluation rows.

    Parameters
    ----------
    past_return : array-like
        Full-length array of ``Close[t]/Close[t-h]-1`` (known at t).
    y_train : array-like
        Training targets only (used for the historical mean).
    n_eval : int
        Number of evaluation rows.
    eval_idx : array-like of int, optional
        Positions of the evaluation rows in ``past_return``.
    """
    past_return = np.asarray(past_return, dtype=float)
    prev = past_return if eval_idx is None else past_return[np.asarray(eval_idx)]
    if len(prev) != n_eval:
        raise ValueError("past_return/eval_idx length does not match n_eval")
    mean = float(np.mean(np.asarray(y_train, dtype=float)))
    return {
        "zero": np.zeros(n_eval),
        "previous_return": prev.astype(float),
        "historical_mean": np.full(n_eval, mean),
    }
