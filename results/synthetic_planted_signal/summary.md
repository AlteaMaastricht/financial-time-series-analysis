# Forecast evaluation (synthetic(n_days=1500, seed=42, ar_coef=0.4))

> Research/educational output only. These are evaluations of statistical forecasts, not trading signals and not investment advice.

Command: `python -m fints forecast --synthetic --n-days 1500 --seed 42 --ar-coef 0.4 --feature-set stationary --horizons 1 --models ridge,et --mode both --folds 5 --out results/synthetic_planted_signal`  
Seed: 42

## Horizon 1 day(s): hold-out test (2019-10-03 to 2020-09-30)

Selected on validation: **weighted_ensemble**

| forecast | n | RMSE | MAE | R² | R² vs zero | direction acc. | up share |
|---|---|---|---|---|---|---|---|
| selected | 260 | 0.013568 | 0.008498 | 0.1312 | 0.1506 | 0.6269 | 0.5846 |
| baseline:historical_mean | 260 | 0.014859 | 0.009013 | -0.0420 | -0.0188 | 0.4154 | 0.5846 |
| baseline:previous_return | 260 | 0.016886 | 0.010611 | -0.3456 | -0.3156 | 0.6385 | 0.5846 |
| baseline:zero | 260 | 0.014722 | 0.008850 | -0.0228 | 0.0000 | 0.4154 | 0.5846 |
| model:extra_trees | 260 | 0.013405 | 0.008484 | 0.1520 | 0.1709 | 0.6500 | 0.5846 |
| model:ridge | 260 | 0.014263 | 0.008689 | 0.0399 | 0.0613 | 0.6038 | 0.5846 |
| model:simple_average | 260 | 0.013720 | 0.008521 | 0.1116 | 0.1314 | 0.6115 | 0.5846 |
| model:stacking_ensemble | 260 | 0.014779 | 0.008931 | -0.0308 | -0.0078 | 0.4192 | 0.5846 |
| model:weighted_ensemble | 260 | 0.013568 | 0.008498 | 0.1312 | 0.1506 | 0.6269 | 0.5846 |

Diebold-Mariano (selected vs zero, squared error, HLN-corrected, h=1): DM = -1.880, p = 0.0613, n = 260 (the selected model has lower mean loss; p does not adjust for model selection).

## Horizon 1 day(s): walk-forward (5 expanding folds, 2018-10-04 to 2020-09-30)

Selected per fold (on that fold's validation block): extra_trees, weighted_ensemble, weighted_ensemble, weighted_ensemble, weighted_ensemble

| forecast | n | RMSE | MAE | R² | R² vs zero | direction acc. | up share |
|---|---|---|---|---|---|---|---|
| selected | 520 | 0.012311 | 0.008183 | 0.1116 | 0.1174 | 0.6250 | 0.5519 |
| baseline:zero | 520 | 0.013104 | 0.008606 | -0.0066 | 0.0000 | 0.4481 | 0.5519 |
| baseline:previous_return | 520 | 0.014933 | 0.010083 | -0.3072 | -0.2986 | 0.6231 | 0.5519 |
| baseline:historical_mean | 520 | 0.013186 | 0.008725 | -0.0193 | -0.0126 | 0.4481 | 0.5519 |

Diebold-Mariano (selected vs zero, squared error, HLN-corrected, h=1): DM = -2.409, p = 0.0163, n = 520 (the selected model has lower mean loss; p does not adjust for model selection).

### Research backtest of the walk-forward forecasts (1-day)

Rule: long if forecast > 0.01, short if < -0.01, else flat; cost 0.001 per unit of position change.

| | total return | Sharpe (ann.) | max drawdown | exposure |
|---|---|---|---|---|
| threshold rule | 0.2830 | 1.202 | -0.0598 | 0.040 |
| buy and hold | 0.6589 | 1.284 | -0.2184 | 1.000 |

