# Provenance

This repository is a restructured and methodologically corrected version of a solo
project written in September 2025 as a series of single-file scripts
(`main.py` … `main6.py`). Those originals are **not** included here. This file maps
every staged module to its source and lists each change and the reason for it, so
the original work and the later corrections stay distinguishable.

Line numbers refer to the original files.

## Source files used

| Original file | LOC | Role | Used for |
|---|---|---|---|
| `main6.py` | 1,507 | final forecasting script (features, tuned tree models, ensembles, LSTM/GRU/Transformer, signals, backtest) | `features.py`, `models.py`, `pipeline.py`, `backtest.py`, `dl.py`, `validation.py` |
| `main4.py` | 520 | final descriptive-statistics script (risk, normality, Hurst, GARCH, Markov switching, PELT, 12-panel figure) | `stats.py` |
| `main6_ml_fine-tune.py` | 556 | `EnhancedMLTrainer`, already merged almost verbatim into `main6.py` | nothing beyond what `main6.py` contains |
| `main6_dl_fine-tune.py` | 436 | `EnhancedDeepLearningTrainer` (CNN-LSTM, custom losses, grid search); never imported (a hyphen in the filename makes it unimportable) | **not ported** (see "Not ported") |
| `main6_guide.txt` | 335 | AI-assistant-style tips sheet | not ported; evidence of AI assistance |
| `main.py`–`main3.py`, `main5.py` | — | earlier iterations of the two lines | superseded, not ported |

## Module map and changes

### `src/fints/data.py` (new, plus the validation from `main6.py` L55-60, L76-82)
* **New:** a deterministic synthetic OHLCV generator and a local CSV loader. **Why:** the
  original downloaded `period='2y'` relative to today with yfinance and asked for
  the ticker with `input()`, so no run could be reproduced.
* yfinance is optional and used only by `download_yfinance`, which takes fixed
  `start`/`end` dates.
* Kept: the minimum-row and missing-value checks.

### `src/fints/features.py` ← `main6.py` L84-191 (`create_features`), L193-198 (`prepare_ml_data` column filter)
* Feature formulas are **unchanged**.
* Added: replacing ±inf with NaN. **Why:** division by zero, e.g. in `RSI` when there
  are no losses in a window.
* Added: a `Dataset` object and the `past_return` column for the previous-return baseline.
* Added: an optional `feature_set="stationary"` that drops price/volume-level
  columns. **Why:** with level features, linear models extrapolate badly out of
  sample (observed on synthetic data, see README). The default is still the
  original set.

### `src/fints/validation.py` ← `main6.py` L207-214 (split), L1405-1471 (`walk_forward_validation`)
* **Changed:** the 80/20 train/test split became a train/validation/test split
  (default 60/20/20). **Why:** the original selected models and fitted ensemble
  weights on the test set (see `models.py`).
* **Added:** purging (`horizon - 1` rows between segments). **Why:** the last 5-day
  training labels overlapped the test window.
* **Changed:** `walk_forward_validation` was defined but never called. It became
  `walk_forward_splits` (expanding window, same `initial_train_frac=0.6`), with an
  inner validation block, and it is now used by the pipeline.
* **Added:** metrics `r2_vs_zero` (out-of-sample R² against a zero forecast) and
  `up_share` (what "always up" would score).

### `src/fints/baselines.py` (new)
* Zero, previous h-day return and training-mean forecasts. **Why:** the original
  had no naive baseline, only a 50% "random" line.

### `src/fints/models.py` ← `main6.py` L972-1401 (`EnhancedMLTrainer`)
* Search spaces are **unchanged**. RandomizedSearchCV with TimeSeriesSplit is kept,
  and `TimeSeriesSplit(gap=horizon-1)` is added.
* **Fixed (test-set selection):** `best_model_name = max(results, key=test_r2)`
  (L1351, and again at L425) became selection by **validation** MSE in
  `fit_and_select`, which takes no test arguments.
* **Fixed (test-set fitting):** `create_weighted_ensemble(predictions, y_test)`
  (L1180-1214, called at L1320) now fits weights on **validation** predictions.
  The simple-average ensemble is no longer one of the weighted inputs
  (L1317 → L1320).
* **Fixed (optimiser scaling):** SLSQP minimised raw MSE (~1e-4 for daily returns),
  which is below SLSQP's default tolerance, so it returned the equal-weight start
  point. This was demonstrated with the unscaled objective during preparation: the
  weights stayed at `[0.5, 0.5]` even when one input was pure noise. The objective
  is now divided by mean(y²); `tests/test_pipeline.py` covers this. Consequence: the
  original "optimal" weighted ensemble was very likely identical to the simple
  average.
* **Changed (stacking):** `StackingRegressor(cv=5)` used unshuffled K-fold, so
  meta-features for early rows came from base models trained on later rows
  (L1152-1157). It is replaced by `TimeSeriesStackingRegressor`, which builds
  forward-only out-of-fold predictions. Base models are the selected candidates
  (the original used fixed rf/gb/et[/xgb/lgbm]).
* **Added:** Ridge is a default candidate. It was configured in the original
  (L1064-1069) but never trained (L1229).
* Added: `n_iter=0` fits fixed defaults (the original stacking base-model
  parameters) for fast runs. The original always used `n_iter=20`.
* xgboost, lightgbm and catboost stay behind import guards (as in the original).

### `src/fints/pipeline.py` ← `main6.py` L375-437 (`train_ml_models`)
* Same order of steps: scale on train → RF feature ranking on train (top 40) →
  tuned models → ensembles. Selection and weights now use validation, and the test
  set is predicted once.
* Added: baselines on every test segment, and walk-forward aggregation.

### `src/fints/backtest.py` ← `main6.py` L849-968 (`run_pnl_backtest`)
* Kept: the ±1% threshold rule and the 0.1% cost.
* **Fixed:** max drawdown now uses the running peak (it divided by the global peak,
  L934).
* **Fixed:** Sharpe is computed on returns (it used dollar P&L, L933).
* **Fixed:** realised returns are aligned arrays. `.iloc` on the DL numpy array
  (L862, L891) would raise.
* **Fixed:** removed the price lookup from the last raw rows (L879), which was
  offset from the test rows.
* Changed: cost is charged per unit of turnover.
* Added: a buy-and-hold benchmark.
* It now backtests the walk-forward forecasts, not the test-selected "ensemble" (L865).

### `src/fints/dl.py` ← `main6.py` L218-373
* Architectures are **unchanged** (Bi-LSTM 128/64, Bi-GRU 128/64, one-block Transformer).
* **Fixed:** the target was off by one. `y.append(target_data[i + seq_length])`
  (L223) paired the window ending at t with the return from t+1 to t+2.
  `make_sequences` now uses `target[t]`.
* **Changed:** windows are split train/val/test, with early stopping on val. The
  original used Keras `validation_split` on the train tail and an 80/20 test split.
* **Changed:** the default inputs are stationary. The original list (with `Close`,
  `SMA_20`, `SMA_50`, `EMA_20` levels) is kept as `ORIGINAL_SEQUENCE_FEATURES`.
* TensorFlow is imported lazily. This module was **not executed** during
  preparation (TensorFlow not installed).

### `src/fints/stats.py` ← `main4.py` L40-158
* Risk metrics and normality tests are ported as is.
* **Changed (Hurst):** `compute_Hc(kind='price')` on price levels (L43, L201)
  produced H = 1.0532 for TSLA, which is invalid. It is replaced by an own R/S
  estimator on returns, plus a validity check.
* **Changed (Markov):** `switching_variance=True` is now used, and
  `regime_separation` detects flat probabilities. The original TSLA figure shows
  probabilities stuck at 0.5.
* **Changed (change points):** "Bayesian changepoint detection" (L87-97) was PELT,
  and is now labelled PELT. It runs on returns instead of price levels.
* Added: a numpy Ljung-Box test, so the statsmodels dependency is not needed for it.
* GARCH, Markov and PELT need optional packages (`arch`, `statsmodels`,
  `ruptures`) and were **not executed** during preparation.

### `src/fints/cli.py` ← `main6.py` L1474-1507, `main4.py` L507-521
* `input()` prompts are replaced by argparse subcommands.
* Removed: the runtime `pip install` (`main4.py` L494-505).
* Every run writes its command, seed and library versions to `metrics.json`.

## Not ported (and why)

* **`generate_trading_signals`** (`main6.py` L460-649): STRONG BUY/SELL, stop-loss
  and take-profit levels. It is trading-advice framing built on test-selected
  models. It also had bugs:
  * it looked up the model keys `'rf'`/`'gb'`, which never exist (L498-499);
  * it reused the 1-day selected-feature indices for the 5-day models (L490-494);
  * it used the scaler last fitted for `target_5d` (L487) and the DL scaler for
    the last target.

  The repository evaluates forecasts and does not issue signals.
* **Plot functions** (`plot_predictions`, `plot_feature_importance`,
  `plot_deep_learning_analysis`, the 12-panel `plot_advanced_analysis`): replaced by
  one walk-forward figure. The descriptive figure is future work.
* **`main6_dl_fine-tune.py`**: never imported or run, and untestable here without
  TensorFlow. It can be re-added as an experimental module.
* **Original PNG outputs** (15 files, ~8 MB): they come from the flawed pipeline
  (test-set selection) and from date-relative downloads. They are not presented as
  results.

## Historical outputs of the original version (not valid results)

These figures were produced by the original `main6.py` (September 2025). The
model shown was selected on the test set, and the data came from a
date-relative download, so they are **not** results of this repository. They
are recorded only to document where the project started. Values were read from
the original PNG files, which are not included here.

| Original figure | What it showed |
|---|---|
| `^SPX_predictions.png` | 1-day: test R² −0.0034, direction acc. 55.0%. 5-day: R² −0.0282, direction acc. 71.2%. Test window Jul–Sep 2025. Predictions nearly constant (about +0.001 / +0.004) |
| `^SPX_deep_learning.png` | DL ensemble 1-day R² −4.6882. Predictions drift to a constant ≈ −0.015 after about 40 test steps. 1-day direction acc.: LSTM 56.4%, GRU 43.6%, Transformer 48.7% |
| `TSLA_analysis.png` | Hurst exponent 1.0532 (invalid, > 1); Markov regime probabilities flat at about 0.5 |

## Phase 4 changes (18 Nov 2026)
- New `src/fints/significance.py`: Diebold-Mariano test (squared-error loss, autocovariances up to lag h-1, Harvey-Leybourne-Newbold small-sample correction, Student-t p-value). New `tests/test_significance.py` (6 tests).
- `cli.py`: the `forecast` report now includes `dm_selected_vs_zero` for the hold-out and walk-forward evaluations (in `metrics.json` and `summary.md`).
- `results/synthetic_demo/` and `results/synthetic_planted_signal/` regenerated with their recorded commands (Python 3.14, system packages). All previously reported values are unchanged (checked key by key against the earlier files); only the DM entries were added. The plots were re-rendered.
- README §8 / §9: DM results added; the "no significance testing" limitation narrowed.
- `LICENSE` added: MIT, copyright 2026 Arthur J. Maarschalkerweerd (author decision D6).
- Default feature set intentionally left at `all`. Switching to `stationary` would change the shipped results; README §9 discloses the issue.

- README §11: AI-assistance statement in its final wording (decision D7); `Status: preparation draft` line removed for publication.
