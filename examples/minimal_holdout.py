"""Minimal library use: one purged train/validation/test split on synthetic data.

Run from the repository root:
    PYTHONPATH=src python3 examples/minimal_holdout.py

Synthetic data says nothing about real markets; this only shows the API.
"""
from fints.data import generate_synthetic_prices
from fints.features import build_dataset
from fints.pipeline import ExperimentConfig, run_holdout

prices = generate_synthetic_prices(n_days=1000, seed=0)
ds = build_dataset(prices, horizon=1, feature_set="stationary")
cfg = ExperimentConfig(model_names=("ridge", "extra_trees"), top_k=20, n_jobs=1)
fold = run_holdout(ds, cfg)

print("selected on validation:", fold.selected)
for name in ("selected", "baseline:zero", "baseline:historical_mean", "baseline:previous_return"):
    m = fold.test_metrics[name]
    print(f"{name:28s} RMSE={m['rmse']:.6f}  R2_vs_zero={m['r2_vs_zero']:+.4f}  dir_acc={m['direction_accuracy']:.3f}")
