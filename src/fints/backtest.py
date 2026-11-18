"""Research backtest of a threshold long/short rule, with a buy-and-hold benchmark.

Ported from ``run_pnl_backtest`` in the original ``main6.py`` (L849-968), with
these fixes:

* Max drawdown uses the running peak at each point. The original divided by the
  global maximum of the running peak (L934).
* Sharpe ratio is computed on daily strategy *returns*. The original used dollar
  P&L, whose scale grows with capital (L933).
* Costs are charged per unit of position change (a long -> short flip counts as
  two units). The original charged one cost per change of any size.
* A buy-and-hold benchmark over the same days is always reported.
* The original indexed realised returns with ``.iloc``, which fails for the
  deep-learning path (numpy arrays), and took prices from the last rows of the raw
  history, which are offset from the test rows. Here predictions and realised
  returns are passed as aligned arrays.

The position decided at day t (from a forecast made with data up to t) earns the
realised return from t to t+1, which is ``target_1d[t]``. Only 1-day forecasts are
backtested. This is an evaluation of the forecasts, not a trading system and not
investment advice.
"""

from __future__ import annotations

import numpy as np


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _sharpe(daily: np.ndarray) -> float:
    sd = daily.std(ddof=1) if len(daily) > 1 else 0.0
    return float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def threshold_positions(pred: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """+1 if forecast > threshold, -1 if < -threshold, else 0 (as in the original)."""
    pred = np.asarray(pred, dtype=float)
    return np.where(pred > threshold, 1.0, np.where(pred < -threshold, -1.0, 0.0))


def backtest(pred, realized, threshold: float = 0.01, cost: float = 0.001) -> dict:
    """Backtest the threshold rule and buy-and-hold on aligned 1-day data."""
    realized = np.asarray(realized, dtype=float)
    pos = threshold_positions(pred, threshold)
    if len(pos) != len(realized):
        raise ValueError("pred and realized must have the same length")
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * realized - cost * turnover
    equity = np.cumprod(1.0 + strat)
    bh = realized.copy()
    bh[0] -= cost  # one entry trade
    bh_equity = np.cumprod(1.0 + bh)
    return {
        "days": int(len(realized)),
        "threshold": threshold,
        "cost_per_unit_turnover": cost,
        "strategy": {
            "total_return": float(equity[-1] - 1.0),
            "sharpe_annualised": _sharpe(strat),
            "max_drawdown": _max_drawdown(np.concatenate([[1.0], equity])),
            "exposure": float(np.mean(pos != 0)),
            "position_changes": int(np.sum(turnover > 0)),
        },
        "buy_and_hold": {
            "total_return": float(bh_equity[-1] - 1.0),
            "sharpe_annualised": _sharpe(bh),
            "max_drawdown": _max_drawdown(np.concatenate([[1.0], bh_equity])),
        },
        "equity_curve": equity.tolist(),
        "buy_and_hold_curve": bh_equity.tolist(),
    }
