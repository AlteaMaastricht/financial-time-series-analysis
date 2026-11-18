# Forecast evaluation (synthetic(n_days=1500, seed=42, ar_coef=0.0))

> Research/educational output only. These are evaluations of statistical forecasts, not trading signals and not investment advice.

Command: `python -m fints forecast --synthetic --n-days 1500 --seed 42 --models rf,gb,et,ridge --mode both --folds 5 --out results/synthetic_demo`  
Seed: 42

## Horizon 1 day(s): hold-out test (2019-10-03 to 2020-09-30)

Selected on validation: **stacking_ensemble**

| forecast | n | RMSE | MAE | R² | R² vs zero | direction acc. | up share |
|---|---|---|---|---|---|---|---|
| selected | 260 | 0.013890 | 0.008471 | -0.0150 | -0.0049 | 0.4269 | 0.5731 |
| baseline:historical_mean | 260 | 0.013908 | 0.008496 | -0.0177 | -0.0075 | 0.4269 | 0.5731 |
| baseline:previous_return | 260 | 0.020351 | 0.012798 | -1.1788 | -1.1571 | 0.4538 | 0.5731 |
| baseline:zero | 260 | 0.013856 | 0.008418 | -0.0101 | 0.0000 | 0.4269 | 0.5731 |
| model:extra_trees | 260 | 0.013472 | 0.008522 | 0.0452 | 0.0547 | 0.4846 | 0.5731 |
| model:gradient_boosting | 260 | 0.013930 | 0.009078 | -0.0208 | -0.0107 | 0.5500 | 0.5731 |
| model:random_forest | 260 | 0.013400 | 0.008503 | 0.0554 | 0.0648 | 0.4731 | 0.5731 |
| model:ridge | 260 | 0.014392 | 0.009005 | -0.0897 | -0.0788 | 0.4808 | 0.5731 |
| model:simple_average | 260 | 0.013588 | 0.008555 | 0.0287 | 0.0384 | 0.5154 | 0.5731 |
| model:stacking_ensemble | 260 | 0.013890 | 0.008471 | -0.0150 | -0.0049 | 0.4269 | 0.5731 |
| model:weighted_ensemble | 260 | 0.013590 | 0.008562 | 0.0283 | 0.0380 | 0.4885 | 0.5731 |

Diebold-Mariano (selected vs zero, squared error, HLN-corrected, h=1): DM = 1.793, p = 0.0742, n = 260 (the selected model has higher mean loss; p does not adjust for model selection).

## Horizon 1 day(s): walk-forward (5 expanding folds, 2018-10-04 to 2020-09-30)

Selected per fold (on that fold's validation block): stacking_ensemble, stacking_ensemble, extra_trees, stacking_ensemble, extra_trees

| forecast | n | RMSE | MAE | R² | R² vs zero | direction acc. | up share |
|---|---|---|---|---|---|---|---|
| selected | 520 | 0.012091 | 0.008088 | 0.0290 | 0.0319 | 0.4942 | 0.5346 |
| baseline:zero | 520 | 0.012288 | 0.007995 | -0.0030 | 0.0000 | 0.4654 | 0.5346 |
| baseline:previous_return | 520 | 0.017874 | 0.012048 | -1.1220 | -1.1157 | 0.4769 | 0.5346 |
| baseline:historical_mean | 520 | 0.012320 | 0.008045 | -0.0082 | -0.0051 | 0.4654 | 0.5346 |

Diebold-Mariano (selected vs zero, squared error, HLN-corrected, h=1): DM = -0.533, p = 0.5942, n = 520 (the selected model has lower mean loss; p does not adjust for model selection).

### Research backtest of the walk-forward forecasts (1-day)

Rule: long if forecast > 0.01, short if < -0.01, else flat; cost 0.001 per unit of position change.

| | total return | Sharpe (ann.) | max drawdown | exposure |
|---|---|---|---|---|
| threshold rule | 0.0892 | 0.476 | -0.0524 | 0.017 |
| buy and hold | 0.3629 | 0.866 | -0.1613 | 1.000 |

## Horizon 5 day(s): hold-out test (2019-09-30 to 2020-09-24)

Selected on validation: **stacking_ensemble**

| forecast | n | RMSE | MAE | R² | R² vs zero | direction acc. | up share |
|---|---|---|---|---|---|---|---|
| selected | 259 | 0.029736 | 0.018267 | -0.0820 | -0.0339 | 0.3938 | 0.6062 |
| baseline:historical_mean | 259 | 0.029794 | 0.018351 | -0.0863 | -0.0379 | 0.3938 | 0.6062 |
| baseline:previous_return | 259 | 0.041269 | 0.025387 | -1.0841 | -0.9913 | 0.4556 | 0.6062 |
| baseline:zero | 259 | 0.029245 | 0.017752 | -0.0466 | 0.0000 | 0.3938 | 0.6062 |
| model:extra_trees | 259 | 0.030623 | 0.021081 | -0.1475 | -0.0964 | 0.5058 | 0.6062 |
| model:gradient_boosting | 259 | 0.027674 | 0.018930 | 0.0628 | 0.1045 | 0.5792 | 0.6062 |
| model:random_forest | 259 | 0.028627 | 0.019619 | -0.0028 | 0.0419 | 0.5637 | 0.6062 |
| model:ridge | 259 | 0.031595 | 0.021716 | -0.2216 | -0.1671 | 0.4981 | 0.6062 |
| model:simple_average | 259 | 0.028692 | 0.019526 | -0.0074 | 0.0375 | 0.5444 | 0.6062 |
| model:stacking_ensemble | 259 | 0.029736 | 0.018267 | -0.0820 | -0.0339 | 0.3938 | 0.6062 |
| model:weighted_ensemble | 259 | 0.030063 | 0.020539 | -0.1060 | -0.0567 | 0.5097 | 0.6062 |

Diebold-Mariano (selected vs zero, squared error, HLN-corrected, h=5): DM = 2.170, p = 0.0309, n = 259 (the selected model has higher mean loss; p does not adjust for model selection).

## Horizon 5 day(s): walk-forward (5 expanding folds, 2018-10-02 to 2020-09-24)

Selected per fold (on that fold's validation block): stacking_ensemble, stacking_ensemble, weighted_ensemble, weighted_ensemble, stacking_ensemble

| forecast | n | RMSE | MAE | R² | R² vs zero | direction acc. | up share |
|---|---|---|---|---|---|---|---|
| selected | 518 | 0.026921 | 0.019008 | -0.0972 | -0.0822 | 0.4595 | 0.5598 |
| baseline:zero | 518 | 0.025879 | 0.017606 | -0.0138 | 0.0000 | 0.4402 | 0.5598 |
| baseline:previous_return | 518 | 0.036439 | 0.025332 | -1.0101 | -0.9827 | 0.4672 | 0.5598 |
| baseline:historical_mean | 518 | 0.026224 | 0.017963 | -0.0411 | -0.0269 | 0.4402 | 0.5598 |

Diebold-Mariano (selected vs zero, squared error, HLN-corrected, h=5): DM = 2.179, p = 0.0298, n = 518 (the selected model has higher mean loss; p does not adjust for model selection).

