# financial-time-series-analysis

Leakage-aware evaluation of machine-learning return forecasts (tuned tree
ensembles, stacking, optional LSTM/GRU/Transformer) against naive baselines,
with walk-forward validation and descriptive time-series statistics (risk, Hurst,
Ljung-Box, GARCH, Markov switching, change points).

> **Research and educational code only. Nothing here is a trading signal or
> investment advice.** The results shipped in `results/` come from **synthetic
> data** and say **nothing about real markets**.

## 1. Overview

The project started as a single-file stock analyser built on yfinance. It
engineered about 60 technical features, tuned six tree-based regressors, built
ensembles and deep-learning models, and printed BUY/SELL recommendations. A review
of that version found that the reported "best model" and the ensemble weights were
selected and fitted **on the test set**, that there was no naive baseline, and that
no run could be reproduced (live, date-relative downloads; interactive prompts).

This repository is the corrected version:

- a package (`src/fints`) with a CLI;
- a chronological **train / validation / test** protocol with purging;
- model selection and ensemble weights on **validation only**;
- **walk-forward** evaluation;
- **naive baselines** reported next to every model;
- fixed date ranges or deterministic synthetic data;
- unit tests that check the leakage guarantees.

What changed relative to the original, and why, is listed in
[Methodology changes vs the original version](#methodology-changes-vs-the-original-version)
and in detail in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## 2. Problem

The question: can next-day (and 5-day) equity returns be forecast from
backward-looking technical features well enough to beat trivial forecasts out of
sample?

Answering that honestly needs four things:

1. features that use only information available at the forecast time;
2. a test period that plays no part in fitting, tuning, model selection or
   ensembling;
3. comparison with forecasts that need no model at all: zero return, last return,
   historical mean;
4. results someone else can regenerate.

## 3. Approach

| Step | What happens | Where |
|---|---|---|
| Data | Deterministic synthetic OHLCV (regime-switching drift, GARCH(1,1) variance, Student-t shocks), or a local CSV with fixed `--start/--end` | `data.py` |
| Features | About 59 backward-looking features: SMAs/EMAs, volatilities, RSI, MACD, Bollinger, momentum/ROC, lagged returns, Parkinson vol, ATR, VWAP gap, channels. Targets are `Close[t+h]/Close[t]-1` | `features.py` |
| Split | Train / validation / test (60/20/20), oldest first, with `h-1` rows purged between segments | `validation.py` |
| Fit (train only) | `StandardScaler`; RF importance ranking (top 40); Random Forest, Gradient Boosting, Extra Trees, Ridge (+ XGBoost/LightGBM/CatBoost if installed), optionally `RandomizedSearchCV` with `TimeSeriesSplit` | `models.py` |
| Ensembles | Simple average; weighted average (SLSQP weights fitted on **validation**); stacking with forward-only out-of-fold meta-features | `models.py` |
| Select (validation only) | Lowest validation MSE among all candidates | `models.fit_and_select` (has no test arguments) |
| Evaluate (test, once) | RMSE, MAE, R², R² vs zero forecast, direction accuracy, share of up days; same metrics for the zero / previous-return / historical-mean baselines | `pipeline.py`, `baselines.py` |
| Walk-forward | 5 expanding windows; the full fit → validate → select loop is repeated inside each fold, and the out-of-sample predictions are concatenated | `pipeline.run_walk_forward` |
| Research backtest | ±1% threshold long/short rule on the walk-forward 1-day forecasts, with costs, against buy-and-hold | `backtest.py` |
| Descriptive stats | Risk metrics, normality tests, R/S Hurst on returns with validity check, Ljung-Box; GARCH(1,1), 2-regime Markov switching with a regime-separation check, PELT (optional packages) | `stats.py` |

## 4. Architecture

```
            prices (synthetic | CSV | optional yfinance with fixed dates)
                                   │
                         features.create_features
                                   │
                        features.build_dataset(h)  ── X, y, past_return
                                   │
          validation.chronological_split / walk_forward_splits (purged)
                                   │
      ┌──────────── per fold ─────────────────────────────────────────┐
      │ train:  StandardScaler.fit, RF top-k, fit candidates, stacking │
      │ val:    ensemble weights, choose min-MSE candidate             │
      │ test:   predict once  +  baselines.zero/previous/mean          │
      └────────────────────────────────────────────────────────────────┘
                                   │
          validation.regression_metrics  →  backtest (1-day, research)
                                   │
                results/<run>/metrics.json, summary.md, *.png
```

```
src/fints/
  data.py        synthetic generator, CSV loader, optional download
  features.py    feature engineering + targets
  validation.py  splits, purging, walk-forward, metrics
  baselines.py   naive forecasts
  models.py      candidates, stacking, weights, validation-only selection
  pipeline.py    hold-out and walk-forward experiments
  backtest.py    threshold rule vs buy-and-hold
  stats.py       descriptive statistics
  dl.py          optional LSTM / GRU / Transformer (TensorFlow)
  cli.py         `python -m fints ...`
tests/           unittest suite (42 tests)
examples/        minimal_holdout.py
results/         synthetic demo outputs (with command + seed)
docs/            PROVENANCE.md
data/            README only (no market data is shipped)
```

## 5. Technologies

Python ≥ 3.10, NumPy, pandas, SciPy, scikit-learn, Matplotlib.

Optional: XGBoost, LightGBM, CatBoost, TensorFlow/Keras, `arch`, `statsmodels`,
`ruptures`, `yfinance` (see `requirements-optional.txt`). `arch`, `statsmodels` and
`ruptures` were installed and tested in a second environment. The other optional
packages were not installed, so the code paths that use them are **implemented but
untested**.

## 6. Installation & running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core only
# pip install -r requirements-optional.txt # optional extras

# run without installing the package:
export PYTHONPATH=src
python3 -m unittest discover -s tests -t .  # 42 tests, ~15 s; 3 skip without the stats extras
```

`pip install -e .` (via `pyproject.toml`) should also work, but was not verified
during preparation.

## 7. Example usage

```bash
export PYTHONPATH=src

# offline synthetic demo (the run shipped in results/synthetic_demo)
python3 -m fints forecast --synthetic --n-days 1500 --seed 42 \
    --models rf,gb,et,ridge --mode both --folds 5 --out results/synthetic_demo

# positive control: planted AR(1) signal, stationary features
python3 -m fints forecast --synthetic --n-days 1500 --seed 42 --ar-coef 0.4 \
    --feature-set stationary --horizons 1 --models ridge,et --mode both --folds 5 \
    --out results/synthetic_planted_signal

# descriptive statistics
python3 -m fints describe --synthetic --n-days 1500 --seed 42 --out results/synthetic_describe

# your own data, fixed sample
python3 -m fints forecast --csv data/my_prices.csv --start 2015-01-01 --end 2024-12-31 --out results/run

# original-style hyperparameter search (slow): --n-iter 20 ; deep learning: --with-dl (needs tensorflow)
python3 examples/minimal_holdout.py
```

## 8. Results

All numbers below are from **synthetic data** produced by this repository's
generator. They test the evaluation machinery and **say nothing about real
markets**. Full tables are in each run's `summary.md`; the exact command, seed and
library versions are in `metrics.json`.

**Run A: realistic synthetic series** (`results/synthetic_demo/`; seed 42,
1,500 days, models rf, gb, et, ridge plus the three ensembles, default parameters).
The generator has regime-dependent drift and volatility, so a small part of the
return is predictable in principle.

| Horizon | Evaluation | Selected model | R² vs zero | Direction acc. | Up share | Zero baseline RMSE | Selected RMSE |
|---|---|---|---|---|---|---|---|
| 1 day | hold-out test (260 days) | stacking_ensemble | −0.0049 | 0.427 | 0.573 | 0.013856 | 0.013890 |
| 1 day | walk-forward (5 folds, 520 days) | per fold: stacking ×3, extra_trees ×2 | 0.0319 | 0.494 | 0.535 | 0.012288 | 0.012091 |
| 5 days | hold-out test (259 days) | stacking_ensemble | −0.0339 | 0.394 | 0.606 | 0.029245 | 0.029736 |
| 5 days | walk-forward (5 folds, 518 days) | per fold: stacking ×3, weighted ×2 | −0.0822 | 0.460 | 0.560 | 0.025879 | 0.026921 |

What this shows:

- The model chosen on validation is about as good as, or worse than, the zero
  forecast. The one exception is a small gain in the 1-day walk-forward run.
- **Diebold-Mariano tests** (squared error, selected model vs zero forecast,
  Harvey-Leybourne-Newbold correction; stored in `metrics.json` as
  `dm_selected_vs_zero`):
  - the 1-day walk-forward gain is **not significant** (DM = −0.533, p = 0.594, n = 520);
  - 1-day hold-out: DM = +1.793, p = 0.074 (selected model slightly worse, not significant at 5%);
  - at 5 days the selected model is **significantly worse** than the zero forecast
    (hold-out DM = +2.170, p = 0.031; walk-forward DM = +2.179, p = 0.030).
  The p-values do not adjust for the model having been selected among several
  candidates.
- Direction accuracy is below the share of up days in every row, so "always up"
  would have scored higher.
- On the hold-out, some candidates that were *not* selected scored better on test
  (e.g. random_forest 1-day R² vs zero 0.0648). Picking them after seeing the test
  set is exactly the optimistic bias the original version had.
- The 1-day research backtest of the walk-forward forecasts returned 8.9% (Sharpe
  0.48, 1.7% of days in a position). Buy-and-hold returned 36.3% (Sharpe 0.87) over
  the same days.

**Run B: positive control with a planted AR(1) coefficient of 0.4**
(`results/synthetic_planted_signal/`; stationary features, ridge + extra trees).
This checks that the pipeline finds signal when signal exists. Walk-forward 1-day:
R² vs zero 0.1174, direction accuracy 0.625 against an up share of 0.552. Hold-out:
R² vs zero 0.1506 (weighted ensemble). The Diebold-Mariano test detects the planted
signal: walk-forward DM = −2.409, p = 0.016 (n = 520); hold-out DM = −1.880,
p = 0.061 (n = 260), i.e. the shorter hold-out alone is borderline.

**Descriptive statistics** (`results/synthetic_describe/stats.json`, same seed as
Run A):

- Ljung-Box on returns: p = 0.30, no linear autocorrelation.
- Ljung-Box on squared returns: p ≈ 5e-19, volatility clustering, as generated.
- R/S Hurst on returns: H = 0.56. The classical R/S estimator is biased upward in
  finite samples (iid returns give roughly 0.5–0.6 in the tests), so this is not
  evidence of long memory.
- GARCH, Markov switching and PELT were reported "not available", because the
  optional packages were not installed for the shipped run. With them installed
  (arch 9.0.0, statsmodels 0.15.0, ruptures 1.1.10), the same command fits GARCH(1,1)
  with α + β = 0.963, two Markov regimes, and finds no PELT change points (pen = 10).
  That output is not shipped.

The original version's figures (e.g. ^SPX 1-day R² −0.0034, deep-learning ensemble
R² −4.69) are **not** results of this repository. They came from test-set model
selection on date-relative downloads and are cited only in `docs/PROVENANCE.md`
as history.

## Methodology changes vs the original version

| Issue in the original (`main6.py` / `main4.py`) | Now |
|---|---|
| Best model chosen by **test** R² (`max(..., key=test_r2)`, L1351 and L425) | Chosen by **validation** MSE; the selection function takes no test data |
| Ensemble weights optimised to minimise **test** MSE (L1180-1214, called at L1320) | Weights fitted on validation predictions |
| SLSQP on raw MSE ≈ 1e-4: the optimiser stops at the equal-weight start, so the "optimal" weights were very likely just the average | Objective normalised by mean(y²); a test checks that weights move |
| Stacking with unshuffled K-fold (`cv=5`): meta-features for early rows came from models trained on later rows | Forward-only out-of-fold stacking (`TimeSeriesSplit`) |
| 80/20 train/test only, no purge for the 5-day target | 60/20/20 train/val/test with `h-1` purged rows; `TimeSeriesSplit(gap=h-1)` inside training |
| `walk_forward_validation` defined (L1405) but never called | Expanding-window walk-forward with per-fold selection, used by default |
| No naive baseline; direction accuracy compared with 50% | Zero, previous-return and historical-mean baselines, R² vs zero, share of up days |
| DL target off by one: window ending at t labelled with the t+1→t+2 return (L223) | Window ending at t labelled with `target[t]` (tested) |
| DL inputs included price levels (`Close`, `SMA_*`, `EMA_20`) | Stationary inputs by default; original list kept for comparison |
| Backtest: drawdown divided by the global peak (L934), Sharpe on dollar P&L (L933), `.iloc` on a numpy array for the DL path | Running-peak drawdown, Sharpe on returns, aligned arrays, buy-and-hold benchmark |
| `generate_trading_signals`: BUY/SELL, stop-loss, take-profit; wrong model keys (`'rf'`/`'gb'`), 1-day feature indices reused for 5-day models | **Removed.** The package evaluates forecasts and does not issue signals |
| Hurst on 10 years of price levels → H = 1.0532 (invalid) interpreted as "trending" | R/S Hurst on returns; values outside [0, 1] flagged as invalid |
| Markov switching with mean-only regimes; smoothed probabilities flat at 0.5 went unnoticed | `switching_variance=True` + regime-separation check |
| PELT on price levels labelled "Bayesian changepoint detection" | Labelled PELT, run on returns |
| GARCH/Markov/ruptures/hurst/TensorFlow imported unconditionally in `main6.py`, though the GARCH/regime/Hurst code was unused there | Optional imports; core needs only numpy/pandas/scipy/sklearn/matplotlib |
| `input()` prompts, `period='2y'` relative to today, runtime `pip install` | argparse CLI, fixed dates or seeded synthetic data, `requirements.txt` |

## 9. Limitations

- **Statistics step runtime.** With the optional `stats` extras installed, a timed run of `fints describe`
  on 1,500 synthetic days took 40 s on a 2015 dual-core laptop (i5-5350U). An earlier, untimed run on the
  same machine took about 30 minutes; the cause of the difference was not established. On that series
  PELT (pen = 10) found no change points, so it should not be read as regime detection.
- **The regime-separation check only detects flat probabilities.** It does not test whether regimes
  exist: on i.i.d. Gaussian returns the 2-regime model still reported `separated = True` (variance
  ratio 3.6), in one check during preparation. A likelihood-ratio or information-criterion comparison
  against a 1-regime model would be needed.
- **No real-market results.** The shipped results use synthetic data. With network
  access and `yfinance`, the same commands run on real tickers with fixed dates,
  but such a run was not made during preparation. The figures produced by the
  original version are not reproduced here, because they came from test-set
  selection on date-relative downloads.
- **Synthetic data is not calibrated** to any asset. It has volatility clustering,
  fat tails and regime-dependent drift, but no microstructure, earnings or
  cross-asset effects.
- **The feature set is large for the sample size** (about 59 features against a
  few hundred training rows per fold). In the positive-control experiment during
  preparation, Ridge on the full feature set did not recover a strong planted
  AR(1) signal, while a three-feature subset did (`tests/test_pipeline.py`). Price-
  and volume-level features extrapolate badly. `--feature-set stationary` helps,
  but the feature set is not otherwise curated.
- **The Hurst interpretation thresholds (±0.05 around 0.5) ignore R/S small-sample
  bias.** An Anis-Lloyd correction or a simulated iid benchmark would be needed
  before reading anything into H.
- **Limited significance testing.** Diebold-Mariano tests compare the selected model
  with the zero forecast only; they do not correct for model selection, and there are
  no bootstrap confidence intervals for direction accuracy.
- **Models are not refitted on train + validation** before testing. Models fitted
  on train are tested directly, which is simpler but uses less data.
- **Untested optional paths:** XGBoost/LightGBM/CatBoost candidates, the
  TensorFlow models (`dl.py`) and the yfinance download were not executed in the
  validation environment. GARCH/Markov/PELT (`stats.py`) were executed in a second
  environment with the `stats` extras and are covered by tests that fit series with
  a planted structure; those tests skip when the extras are missing.
- **The backtest is a research check of forecast usefulness.** It ignores
  slippage, borrow costs for shorts and position sizing.
- GARCH conditional volatility and regime probabilities are not yet used as
  forecasting features. Doing so needs a rolling, train-only fit to avoid
  leakage.

## 10. Reproducibility

- Every forecast run writes `metrics.json` with the exact command, seed, data
  source and library versions, plus a `summary.md`.
- The synthetic generator is fully determined by its parameters and `--seed`
  (`tests/test_backtest_stats_dl_data.py` checks this). Model `random_state`
  values come from `--seed`.
- Real data: pass a CSV with fixed `--start/--end`. Downloaded files are not
  committed (`data/*.csv` is git-ignored).
- Exact numeric reproduction across library versions is not guaranteed for tree
  models; the versions used are recorded in each `metrics.json`.
- Can't be regenerated: the original version's figures (date-relative live
  downloads).

## 11. Context, attribution & licence

- Solo personal project. The original single-file versions date from September
  2025. This restructured, methodologically corrected version was prepared in
  November 2026; the change log is in `docs/PROVENANCE.md`.
- Not a course assignment. No course or company data is used.
- AI assistance:
  - **Original 2025 scripts:** developed with AI assistance. The specific tools used and the extent of
    their contribution were not recorded and are not specified here.
  - **2026 rework:** the leakage-safe evaluation protocol, naive baselines, Diebold-Mariano significance
    tests, unit tests, the synthetic demo and the documentation were done with Claude Code under my
    direction. I reviewed the changes.
  - All shipped results are synthetic and reproducible from the recorded commands; details are in
    [docs/PROVENANCE.md](docs/PROVENANCE.md).
- Data: synthetic, generated by `src/fints/data.py`. The optional Yahoo Finance
  download is subject to Yahoo's terms, and such data is not redistributed here.
- License: MIT — see [`LICENSE`](LICENSE).
