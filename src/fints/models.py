"""Candidate regressors, ensembles and validation-only model selection.

Ported from ``EnhancedMLTrainer`` in the original ``main6.py`` (L972-1471). The
search spaces are unchanged. What changed:

* ``fit_and_select`` has no test-set arguments. It fits candidates on the
  training segment, then fits ensemble weights and picks the best candidate on the
  validation segment only. In the original, ``create_weighted_ensemble`` minimised
  *test* MSE (L1180-1214, called at L1320) and the best model was the one with the
  highest *test* R^2 (L1351, L425).
* The weighted ensemble is fitted over the base models only. The original also fed
  the simple-average ensemble in as one more "model" (L1317 before L1320).
* Stacking uses out-of-fold predictions from ``TimeSeriesSplit`` (each base model
  only predicts rows later than the ones it was trained on). The original used
  ``StackingRegressor(cv=5)``, i.e. unshuffled K-fold, so meta-features for early
  rows came from base models trained on later data (L1152-1157).
* Hyperparameter search uses ``TimeSeriesSplit`` with a purge gap for multi-day
  targets.
* xgboost / lightgbm / catboost are optional and are skipped if not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:  # optional
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    XGB_AVAILABLE = False
try:  # optional
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LGBM_AVAILABLE = False
try:  # optional
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    CATBOOST_AVAILABLE = False

# Short CLI names -> internal names
ALIASES = {
    "rf": "random_forest", "gb": "gradient_boosting", "et": "extra_trees",
    "ridge": "ridge", "xgb": "xgboost", "lgbm": "lightgbm", "cat": "catboost",
}
DEFAULT_MODELS = ("random_forest", "gradient_boosting", "extra_trees", "ridge")


def available_models() -> list[str]:
    names = list(DEFAULT_MODELS)
    if XGB_AVAILABLE:
        names.append("xgboost")
    if LGBM_AVAILABLE:
        names.append("lightgbm")
    if CATBOOST_AVAILABLE:
        names.append("catboost")
    return names


def resolve_model_names(spec: str | None) -> list[str]:
    if not spec:
        return list(DEFAULT_MODELS)
    names = [ALIASES.get(s.strip(), s.strip()) for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in available_models()]
    if unknown:
        raise ValueError(f"unknown or unavailable models: {unknown}; available: {available_models()}")
    return names


def get_model_configs(seed: int = 42, n_jobs: int = -1) -> dict:
    """Estimator + search space per model (search spaces as in the original)."""
    configs = {
        "random_forest": {
            "model": RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
            "default": dict(n_estimators=200, max_depth=10),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", 0.5, 0.7],
                "bootstrap": [True, False],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(random_state=seed),
            "default": dict(n_estimators=200, learning_rate=0.05, max_depth=5),
            "params": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "max_features": ["sqrt", "log2", 0.5],
            },
        },
        "extra_trees": {
            "model": ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs),
            "default": dict(n_estimators=200, max_depth=10),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5],
            },
        },
        "ridge": {
            "model": Ridge(random_state=seed),
            "default": dict(alpha=1.0),
            "params": {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        },
    }
    if XGB_AVAILABLE:  # pragma: no cover
        configs["xgboost"] = {
            "model": XGBRegressor(random_state=seed, n_jobs=n_jobs),
            "default": dict(n_estimators=200, learning_rate=0.05, max_depth=5),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7, 9],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "reg_alpha": [0, 0.1, 1],
                "reg_lambda": [0.1, 1, 10],
            },
        }
    if LGBM_AVAILABLE:  # pragma: no cover
        configs["lightgbm"] = {
            "model": LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1),
            "default": dict(n_estimators=200, learning_rate=0.05, num_leaves=31),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63, 127],
                "max_depth": [3, 5, 7, 9, -1],
                "min_child_samples": [5, 10, 20],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "reg_alpha": [0, 0.1, 1],
                "reg_lambda": [0.1, 1, 10],
            },
        }
    if CATBOOST_AVAILABLE:  # pragma: no cover
        configs["catboost"] = {
            "model": CatBoostRegressor(random_state=seed, verbose=0),
            "default": dict(iterations=200, learning_rate=0.05, depth=6),
            "params": {
                "iterations": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1, 3, 5, 10],
                "border_count": [32, 64, 128],
                "bagging_temperature": [0, 1, 10],
            },
        }
    return configs


def build_model(name: str, X, y, n_iter: int = 0, cv_splits: int = 5, gap: int = 0,
                seed: int = 42, n_jobs: int = -1):
    """Fit one candidate on (X, y).

    ``n_iter == 0`` fits fixed default parameters (those of the original stacking
    base models). ``n_iter > 0`` runs ``RandomizedSearchCV`` with
    ``TimeSeriesSplit(cv_splits, gap)`` inside (X, y), as the original did with
    ``n_iter=20``.
    """
    cfg = get_model_configs(seed=seed, n_jobs=n_jobs)[name]
    if n_iter <= 0:
        model = clone(cfg["model"]).set_params(**cfg["default"])
        model.fit(X, y)
        return model, dict(cfg["default"])
    search = RandomizedSearchCV(
        clone(cfg["model"]),
        param_distributions=cfg["params"],
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=cv_splits, gap=gap),
        scoring="neg_mean_squared_error",
        n_jobs=1,
        random_state=seed,
    )
    search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_)


def select_features_rf(X, y, feature_names, top_k: int = 40, seed: int = 42, n_jobs: int = -1):
    """Random-forest importance ranking on the training rows (original L1371-1401).

    ``top_k <= 0`` or ``top_k >= n_features`` keeps every feature (no RF fit).
    """
    if top_k <= 0 or top_k >= len(feature_names):
        return np.arange(len(feature_names)), {}
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=n_jobs)
    rf.fit(X, y)
    order = np.argsort(-rf.feature_importances_, kind="stable")
    keep = np.sort(order[: min(top_k, len(feature_names))])
    scores = {feature_names[i]: float(rf.feature_importances_[i]) for i in range(len(feature_names))}
    return keep, scores


class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking whose meta-features are forward-only out-of-fold predictions.

    For each ``TimeSeriesSplit`` fold, base models are fitted on the earlier block
    and predict the next block. The Ridge meta-model is fitted on those
    out-of-fold predictions (rows of the first block have none and are skipped).
    Base models are then refitted on all training rows for prediction.
    """

    def __init__(self, estimators, final_estimator=None, n_splits: int = 5, gap: int = 0):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits
        self.gap = gap

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        oof = np.full((len(y), len(self.estimators)), np.nan)
        for tr, te in TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap).split(X):
            for j, (_, est) in enumerate(self.estimators):
                oof[te, j] = clone(est).fit(X[tr], y[tr]).predict(X[te])
        mask = ~np.isnan(oof).any(axis=1)
        self.oof_mask_ = mask
        meta = clone(self.final_estimator) if self.final_estimator is not None else Ridge(alpha=1.0)
        self.final_estimator_ = meta.fit(oof[mask], y[mask])
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        return self

    def predict(self, X):
        X = np.asarray(X)
        base = np.column_stack([est.predict(X) for est in self.estimators_])
        return self.final_estimator_.predict(base)


def fit_ensemble_weights(pred_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise MSE on (pred_matrix, y).

    Same SLSQP set-up as the original ``create_weighted_ensemble``; the caller must
    pass *validation* predictions and targets.
    The objective is MSE divided by mean(y^2). Without this scaling, daily-return
    MSEs (~1e-4) are below SLSQP's default tolerance and the optimiser stops at the
    equal-weight starting point; that is what the original set-up did.
    """
    pred_matrix = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    k = pred_matrix.shape[1]
    scale = float(np.mean(y**2)) or 1.0

    def objective(w):
        return mean_squared_error(y, pred_matrix @ w) / scale

    res = minimize(
        objective,
        np.full(k, 1.0 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)


@dataclass
class SelectionResult:
    """Everything decided on train + validation. Test data is never passed in."""

    base_models: dict
    base_params: dict
    weights: np.ndarray
    stacking: TimeSeriesStackingRegressor | None
    val_mse: dict
    selected: str
    feature_idx: np.ndarray
    feature_scores: dict = field(default_factory=dict)

    def predict_all(self, X) -> dict[str, np.ndarray]:
        """Predictions of every candidate on X (columns already scaled, all features)."""
        Xs = np.asarray(X)[:, self.feature_idx]
        preds = {n: m.predict(Xs) for n, m in self.base_models.items()}
        stack = np.column_stack([preds[n] for n in self.base_models])
        preds["simple_average"] = stack.mean(axis=1)
        preds["weighted_ensemble"] = stack @ self.weights
        if self.stacking is not None:
            preds["stacking_ensemble"] = self.stacking.predict(Xs)
        return preds

    def predict(self, X) -> np.ndarray:
        return self.predict_all(X)[self.selected]


def fit_and_select(
    X_train, y_train, X_val, y_val, feature_names,
    model_names=DEFAULT_MODELS, n_iter: int = 0, gap: int = 0, top_k: int = 40,
    use_stacking: bool = True, seed: int = 42, n_jobs: int = -1,
) -> SelectionResult:
    """Fit candidates on train; fit ensemble weights and select on validation."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    feat_idx, scores = select_features_rf(X_train, y_train, list(feature_names), top_k, seed, n_jobs)
    Xt, Xv = X_train[:, feat_idx], X_val[:, feat_idx]

    base, params = {}, {}
    for name in model_names:
        base[name], params[name] = build_model(name, Xt, y_train, n_iter=n_iter, gap=gap,
                                               seed=seed, n_jobs=n_jobs)

    val_preds = {n: m.predict(Xv) for n, m in base.items()}
    matrix = np.column_stack([val_preds[n] for n in base])
    weights = fit_ensemble_weights(matrix, y_val)
    val_preds["simple_average"] = matrix.mean(axis=1)
    val_preds["weighted_ensemble"] = matrix @ weights

    stacking = None
    if use_stacking and len(base) > 1:
        cfgs = get_model_configs(seed=seed, n_jobs=n_jobs)
        estimators = [(n, clone(cfgs[n]["model"]).set_params(**cfgs[n]["default"])) for n in base]
        stacking = TimeSeriesStackingRegressor(estimators, Ridge(alpha=1.0), n_splits=5, gap=gap)
        stacking.fit(Xt, y_train)
        val_preds["stacking_ensemble"] = stacking.predict(Xv)

    val_mse = {n: float(mean_squared_error(y_val, p)) for n, p in val_preds.items()}
    selected = min(val_mse, key=val_mse.get)
    return SelectionResult(
        base_models=base, base_params=params, weights=weights, stacking=stacking,
        val_mse=val_mse, selected=selected, feature_idx=feat_idx, feature_scores=scores,
    )
