"""Price data: deterministic synthetic generator, local CSV loader, optional download.

The original scripts downloaded a period relative to "today" from Yahoo Finance
(``yf.Ticker(t).history(period='2y')``), so no run could be reproduced. Here every
data source takes explicit dates, and the default source is an offline synthetic
generator whose output depends only on its parameters and seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def generate_synthetic_prices(
    n_days: int = 2000,
    seed: int = 42,
    start: str = "2015-01-02",
    s0: float = 100.0,
    ar_coef: float = 0.0,
    regime_mu: tuple[float, float] = (0.0004, -0.0006),
    regime_omega: tuple[float, float] = (2e-6, 1.2e-5),
    regime_stay: tuple[float, float] = (0.99, 0.97),
    garch_alpha: float = 0.08,
    garch_beta: float = 0.90,
    t_dof: float = 5.0,
    return_states: bool = False,
):
    """Simulate daily OHLCV bars.

    Log returns follow a two-state Markov-switching mean, a GARCH(1,1) variance
    with a state-dependent intercept, and Student-t shocks (fat tails):

        r_t = mu[s_t] + ar_coef * r_{t-1} + sigma_t * z_t
        sigma_t^2 = omega[s_t] + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    With ``ar_coef = 0`` (default) next-day returns are unpredictable from the
    past apart from the small regime-dependent drift, which is the realistic null
    case. With ``ar_coef > 0`` there is a planted, learnable signal; that is used as
    a positive control to check the pipeline can find signal when it exists.

    Parameters are illustrative. They are not calibrated to any real asset, and
    results on this data say nothing about real markets.

    Returns a DataFrame indexed by business days with columns Open, High, Low,
    Close, Volume. If ``return_states`` is True, also returns the hidden regime path
    (kept out of the DataFrame on purpose so it can never become a feature).
    """
    if n_days < 50:
        raise ValueError("n_days must be >= 50")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, name="Date")

    # Hidden regime path.
    states = np.empty(n_days, dtype=int)
    states[0] = 0
    for t in range(1, n_days):
        stay = regime_stay[states[t - 1]]
        states[t] = states[t - 1] if rng.random() < stay else 1 - states[t - 1]

    # Unit-variance Student-t shocks.
    z = rng.standard_t(t_dof, size=n_days) * np.sqrt((t_dof - 2.0) / t_dof)

    mu = np.asarray(regime_mu)[states]
    omega = np.asarray(regime_omega)[states]
    r = np.zeros(n_days)
    var = np.zeros(n_days)
    eps_prev = 0.0
    var_prev = omega[0] / max(1e-12, 1.0 - garch_alpha - garch_beta)
    r_prev = 0.0
    for t in range(n_days):
        var[t] = omega[t] + garch_alpha * eps_prev**2 + garch_beta * var_prev
        eps = np.sqrt(var[t]) * z[t]
        r[t] = mu[t] + ar_coef * r_prev + eps
        eps_prev, var_prev, r_prev = eps, var[t], r[t]

    close = s0 * np.exp(np.cumsum(r))
    prev_close = np.concatenate([[s0], close[:-1]])
    sigma = np.sqrt(var)
    open_ = prev_close * np.exp(0.2 * sigma * rng.standard_normal(n_days))
    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    high = hi_base * np.exp(np.abs(rng.standard_normal(n_days)) * 0.5 * sigma)
    low = lo_base * np.exp(-np.abs(rng.standard_normal(n_days)) * 0.5 * sigma)
    volume = np.round(
        1e6 * np.exp(0.3 * rng.standard_normal(n_days)) * (1.0 + 20.0 * np.abs(r))
    )

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    if return_states:
        return df, pd.Series(states, index=dates, name="regime")
    return df


def load_price_csv(
    path: str | Path,
    date_column: str = "Date",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV bars from a local CSV file.

    Required columns: ``Date`` (or ``date_column``), Open, High, Low, Close, Volume.
    Rows are sorted by date, duplicate dates are dropped (first kept), and the
    optional inclusive ``start``/``end`` filter is applied.
    """
    df = pd.read_csv(path)
    if date_column not in df.columns:
        raise ValueError(f"CSV has no '{date_column}' column; columns: {list(df.columns)}")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    dates = pd.to_datetime(df[date_column], utc=True).dt.tz_localize(None).dt.normalize()
    out = df[OHLCV_COLUMNS].astype(float)
    out.index = pd.DatetimeIndex(dates, name="Date")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]
    return validate_prices(out)


def validate_prices(df: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    """Basic sanity checks carried over from the original analyser."""
    if len(df) < min_rows:
        raise ValueError(f"Insufficient data: {len(df)} rows (minimum {min_rows}).")
    if df["Close"].isna().sum() > 0.1 * len(df):
        raise ValueError("Too many missing values in Close.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Price index must be sorted in time.")
    return df


def download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:  # pragma: no cover
    """Download daily bars with yfinance for an explicit, fixed date range.

    Needs the optional ``yfinance`` package and network access. Not used by the
    tests or the demo. Yahoo Finance data is subject to Yahoo's terms of use: keep
    downloaded files local and do not commit them to the repository.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("download_yfinance needs the optional package 'yfinance'.") from exc
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    hist = hist[OHLCV_COLUMNS].copy()
    hist.index = pd.DatetimeIndex(hist.index.tz_localize(None).normalize(), name="Date")
    return validate_prices(hist)
