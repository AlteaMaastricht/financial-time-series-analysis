"""Diebold-Mariano test for equal predictive accuracy.

Compares two forecasts of the same target under squared-error loss.

``d_t = (y_t - a_t)^2 - (y_t - b_t)^2``. Under H0, E[d_t] = 0. The variance of the
mean loss differential uses the autocovariances up to lag ``h - 1``, because an
optimal h-step forecast has MA(h-1) errors (Diebold & Mariano, 1995). The statistic
is then rescaled with the Harvey, Leybourne & Newbold (1997) small-sample correction
and compared with a Student-t distribution with ``n - 1`` degrees of freedom.

A **negative** statistic means forecast ``a`` has the lower loss. The test treats
the forecasts as given. It does not account for the fact that ``a`` was chosen
among several candidates on a validation block, so p-values are, if anything,
optimistic for the selected model.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def diebold_mariano(y, pred_a, pred_b, horizon: int = 1) -> dict:
    y = np.asarray(y, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    if not (y.shape == a.shape == b.shape) or y.ndim != 1:
        raise ValueError("y, pred_a and pred_b must be 1-D arrays of equal length")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    d = (y - a) ** 2 - (y - b) ** 2
    n = d.size
    if n < 2 * horizon + 1:
        raise ValueError(f"need at least {2 * horizon + 1} observations, got {n}")
    mean_d = float(d.mean())
    dc = d - mean_d
    gamma = [float(dc[k:] @ dc[: n - k]) / n for k in range(horizon)]
    long_run_var = gamma[0] + 2.0 * sum(gamma[1:])
    if long_run_var <= 0.0:          # can happen with h > 1 in short samples
        long_run_var = gamma[0]
    if long_run_var == 0.0:          # identical losses: nothing to test
        return {"statistic": float("nan"), "p_value": float("nan"), "mean_loss_diff": mean_d,
                "n": n, "horizon": horizon, "note": "zero variance of loss differential"}
    dm = mean_d / math.sqrt(long_run_var / n)
    correction = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * correction
    p = float(2.0 * stats.t.sf(abs(stat), df=n - 1))
    return {"statistic": float(stat), "p_value": p, "mean_loss_diff": mean_d,
            "n": n, "horizon": horizon}
