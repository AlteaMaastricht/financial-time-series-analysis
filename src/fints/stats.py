"""Descriptive statistics of a return series (the analysis line of the original).

Ported from ``AdvancedStockAnalyzer`` in the original ``main4.py``. Changes:

* Hurst exponent: the original ran ``hurst.compute_Hc(kind='price')`` on ten years
  of price *levels* (``main4.py`` L40-47, L201) and reported H = 1.0532 for TSLA,
  outside the valid range [0, 1], interpreted as "Trending market". Here H is
  estimated with a self-contained rescaled-range (R/S) estimator on *returns*, and
  values outside [0, 1] are flagged as invalid instead of interpreted.
* Markov switching: the original used ``MarkovRegression(k_regimes=2, trend='c')``
  without ``switching_variance`` (L80), so regimes could differ only in their mean.
  The TSLA figure shows smoothed probabilities flat at 0.5, i.e. no separation,
  and the code did not notice. Here ``switching_variance=True`` is used and
  ``regime_separation`` reports whether the regimes actually separate.
* Change points: the original applied PELT (``ruptures``, rbf cost, pen=10) to
  price levels and called it "Bayesian changepoint detection" (L87-97). PELT is not
  Bayesian. Here it is named PELT and applied to returns.
* Ljung-Box is implemented here with numpy/scipy, so it runs without statsmodels.

GARCH (``arch``), Markov switching (``statsmodels``) and PELT (``ruptures``) are
optional. If a package is missing, the function returns
``{"available": False, ...}`` instead of failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


# ----------------------------------------------------------------- risk & shape
def risk_metrics(returns: pd.Series) -> dict:
    """Annualised return/vol, Sharpe and Sortino (rf = 0), VaR/CVaR, drawdown, moments."""
    r = pd.Series(returns).dropna()
    mean_ann = r.mean() * 252
    vol_ann = r.std() * np.sqrt(252)
    neg = r[r < 0]
    dd_ann = neg.std() * np.sqrt(252) if len(neg) > 1 else np.nan
    var95 = float(np.percentile(r, 5))
    cum = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    return {
        "annual_return": float(mean_ann),
        "annual_volatility": float(vol_ann),
        "sharpe_ratio": float(mean_ann / vol_ann) if vol_ann > 0 else float("nan"),
        "sortino_ratio": float(mean_ann / dd_ann) if dd_ann and dd_ann > 0 else float("nan"),
        "var_95": var95,
        "cvar_95": float(r[r <= var95].mean()),
        "max_drawdown": float(drawdown.min()),
        "skewness": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
    }


def normality_tests(returns: pd.Series) -> dict:
    r = pd.Series(returns).dropna().to_numpy()
    sw = sps.shapiro(r[:5000])
    jb = sps.jarque_bera(r)
    da = sps.normaltest(r)
    return {
        "shapiro_p": float(sw.pvalue),
        "jarque_bera_p": float(jb.pvalue),
        "dagostino_p": float(da.pvalue),
        "normal_at_5pct": bool(min(sw.pvalue, jb.pvalue, da.pvalue) > 0.05),
    }


def autocorrelation(x, nlags: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = np.sum(x**2)
    return np.array([1.0] + [np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def ljung_box(x, lags: int = 10) -> dict:
    """Ljung-Box Q statistic and chi-square p-value for lags 1..``lags``."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    acf = autocorrelation(x, lags)[1:]
    q = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    return {"lags": lags, "q_stat": float(q), "p_value": float(sps.chi2.sf(q, lags))}


# ----------------------------------------------------------------------- Hurst
def hurst_rs(increments, min_window: int = 10, max_window: int | None = None,
             n_windows: int = 20) -> float:
    """Rescaled-range Hurst estimate for a series of *increments* (e.g. returns).

    For iid increments the expected value is about 0.5 (the classical R/S
    estimator is biased slightly upward in small windows). Passing price levels
    instead of increments is a misuse that pushes H towards or above 1.
    """
    x = np.asarray(increments, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    max_window = max_window or n // 2
    if n < 2 * min_window or max_window <= min_window:
        raise ValueError("series too short for R/S analysis")
    windows = np.unique(np.floor(np.logspace(np.log10(min_window), np.log10(max_window), n_windows)).astype(int))
    log_w, log_rs = [], []
    for w in windows:
        vals = []
        for i in range(n // w):
            c = x[i * w:(i + 1) * w]
            z = np.cumsum(c - c.mean())
            s = c.std()
            if s > 0:
                vals.append((z.max() - z.min()) / s)
        if vals:
            log_w.append(np.log(w))
            log_rs.append(np.log(np.mean(vals)))
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(slope)


def interpret_hurst(h: float, band: float = 0.05) -> str:
    if not np.isfinite(h) or h < 0 or h > 1:
        return "invalid (outside [0, 1]): check the input series and estimator"
    if h > 0.5 + band:
        return "persistent (H > 0.5)"
    if h < 0.5 - band:
        return "anti-persistent (H < 0.5)"
    return "close to 0.5 (no evidence of long memory)"


# ------------------------------------------------------- optional-dependency models
def garch_11(returns: pd.Series) -> dict:
    """GARCH(1,1) on percentage returns with Ljung-Box checks of standardised residuals."""
    try:
        from arch import arch_model
    except ImportError:
        return {"available": False, "reason": "optional package 'arch' not installed"}
    r = pd.Series(returns).dropna() * 100
    res = arch_model(r, vol="Garch", p=1, q=1, dist="normal").fit(disp="off", show_warning=False)
    std_resid = (res.resid / res.conditional_volatility).dropna()
    alpha, beta = float(res.params["alpha[1]"]), float(res.params["beta[1]"])
    return {
        "available": True,
        "omega": float(res.params["omega"]),
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "ljung_box_resid_p": ljung_box(std_resid, 10)["p_value"],
        "ljung_box_sq_resid_p": ljung_box(std_resid**2, 10)["p_value"],
        "conditional_volatility": (res.conditional_volatility / 100).rename("garch_vol"),
    }


def regime_separation(smoothed_probs, threshold: float = 0.8) -> dict:
    """Check whether a fitted regime model actually separates regimes.

    ``smoothed_probs`` is (n_obs, k). Reports the share of days on which some regime
    has probability above ``threshold``. Probabilities stuck near 1/k (like the
    flat 0.5 lines in the original TSLA figure) give a share near zero and
    ``separated = False``.
    """
    p = np.asarray(smoothed_probs, dtype=float)
    confident = float(np.mean(p.max(axis=1) > threshold))
    return {"share_confident_days": confident, "separated": bool(confident >= 0.5),
            "mean_max_prob": float(p.max(axis=1).mean())}


def markov_switching(returns: pd.Series, k_regimes: int = 2) -> dict:
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        return {"available": False, "reason": "optional package 'statsmodels' not installed"}
    r = pd.Series(returns).dropna()
    res = MarkovRegression(r, k_regimes=k_regimes, trend="c", switching_variance=True).fit(disp=False)
    probs = res.smoothed_marginal_probabilities
    out = {
        "available": True,
        "k_regimes": k_regimes,
        "llf": float(res.llf), "aic": float(res.aic), "bic": float(res.bic),
        "regime_means": [float(res.params[f"const[{i}]"]) for i in range(k_regimes)],
        "regime_variances": [float(res.params[f"sigma2[{i}]"]) for i in range(k_regimes)],
        "smoothed_probabilities": probs,
    }
    out.update(regime_separation(np.asarray(probs)))
    return out


def pelt_change_points(series: pd.Series, penalty: float = 10.0, model: str = "rbf") -> dict:
    """PELT change points (``ruptures``) on a return series. Not a Bayesian method."""
    try:
        import ruptures as rpt
    except ImportError:
        return {"available": False, "reason": "optional package 'ruptures' not installed"}
    s = pd.Series(series).dropna()
    bkps = rpt.Pelt(model=model).fit(s.to_numpy().reshape(-1, 1)).predict(pen=penalty)[:-1]
    return {"available": True, "method": f"PELT ({model} cost, pen={penalty})",
            "indices": [int(b) for b in bkps], "dates": [str(s.index[b].date()) for b in bkps]}


# -------------------------------------------------------------------- summary
def describe(prices: pd.DataFrame) -> dict:
    """All descriptive statistics for a price frame (JSON-serialisable parts only)."""
    r = prices["Close"].pct_change().dropna()
    out = {
        "rows": int(len(prices)),
        "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()),
        "risk": risk_metrics(r),
        "normality": normality_tests(r),
        "ljung_box_returns": ljung_box(r, 10),
        "ljung_box_squared_returns": ljung_box(r**2, 10),
    }
    h = hurst_rs(r.to_numpy())
    out["hurst_returns"] = {"H": h, "interpretation": interpret_hurst(h)}
    out["garch"] = {k: v for k, v in garch_11(r).items() if k != "conditional_volatility"}
    out["markov_switching"] = {k: v for k, v in markov_switching(r).items() if k != "smoothed_probabilities"}
    out["change_points"] = pelt_change_points(r)
    return out
