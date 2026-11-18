"""Backward-looking technical features and forward-return targets.

``create_features`` is a port of ``PredictiveStockAnalyzer.create_features`` from
the original ``main6.py`` (L84-191). The feature definitions are unchanged. Every
feature at row t uses only prices and volumes up to and including t
(rolling windows, ewm, positive shifts); ``tests/test_features.py`` checks this by
recomputing features on truncated histories.

Targets are forward returns and are never used as features:
``target_{h}d[t] = Close[t+h] / Close[t] - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns never used as model inputs (raw OHLCV, targets, corporate actions).
NON_FEATURE_COLUMNS = {
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "target_direction",
}


def target_column(horizon: int) -> str:
    return f"target_{horizon}d"


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from the close at t to the close at t + horizon (NaN for the last rows)."""
    return close.shift(-horizon) / close - 1.0


def create_features(prices: pd.DataFrame, horizons: tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    """Compute the original ~70-column feature set plus forward targets."""
    df = prices.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # Price-based features
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))

    # Technical indicators
    for window in [5, 10, 20, 50, 200]:
        df[f"SMA_{window}"] = close.rolling(window).mean()
        df[f"EMA_{window}"] = close.ewm(span=window).mean()
        df[f"volatility_{window}"] = df["returns"].rolling(window).std()

    # Price position relative to moving averages
    df["price_to_sma20"] = close / df["SMA_20"]
    df["price_to_sma50"] = close / df["SMA_50"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    df["BB_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + bb_std * 2
    df["BB_lower"] = df["BB_middle"] - bb_std * 2
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    df["BB_position"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # Volume features
    df["volume_sma"] = vol.rolling(20).mean()
    df["volume_ratio"] = vol / df["volume_sma"]

    # Momentum
    df["momentum_5"] = close / close.shift(5) - 1
    df["momentum_10"] = close / close.shift(10) - 1
    df["momentum_20"] = close / close.shift(20) - 1

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)

    # Volatility measures
    df["realized_volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)
    df["parkinson_volatility"] = np.sqrt(
        1 / (4 * np.log(2)) * (np.log(high / low) ** 2).rolling(20).mean()
    ) * np.sqrt(252)

    # Advanced features (merged into main6.py from the "EnhancedMLTrainer")
    df["volatility_ratio_10_50"] = close.rolling(10).std() / close.rolling(50).std()
    df["price_acceleration"] = close.diff().diff()
    df["vwap"] = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
    df["vwap_diff"] = (close - df["vwap"]) / df["vwap"]
    df["hl_ratio"] = (high - low) / close
    df["hl_ma"] = df["hl_ratio"].rolling(20).mean()
    df["gap"] = (df["Open"] - close.shift(1)) / close.shift(1)
    df["trend_strength"] = (close - close.rolling(20).mean()).abs() / close.rolling(20).std()
    for window in [3, 7, 14, 21, 30]:
        df[f"roc_{window}"] = (close / close.shift(window) - 1) * 100
    df["ema_8_21_ratio"] = close.ewm(span=8).mean() / close.ewm(span=21).mean()
    df["ema_21_55_ratio"] = close.ewm(span=21).mean() / close.ewm(span=55).mean()
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df["atr"] = ranges.max(axis=1).rolling(14).mean()
    df["atr_percent"] = df["atr"] / close
    df["upper_channel"] = close.rolling(20).mean() + 2 * close.rolling(20).std()
    df["lower_channel"] = close.rolling(20).mean() - 2 * close.rolling(20).std()
    df["channel_position"] = (close - df["lower_channel"]) / (
        df["upper_channel"] - df["lower_channel"]
    )

    # Targets (future returns) -- excluded from features by name.
    for h in horizons:
        df[target_column(h)] = forward_return(close, h)
    if 1 in horizons:
        df["target_direction"] = (df["target_1d"] > 0).astype(int)

    return df.replace([np.inf, -np.inf], np.nan)


# Features measured in price or volume units (non-stationary levels). Models fitted
# on one price range extrapolate badly once the test period leaves that range.
LEVEL_FEATURES = {
    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_200",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_200",
    "MACD", "MACD_signal", "BB_middle", "BB_upper", "BB_lower", "volume_sma",
    "price_acceleration", "vwap", "atr", "upper_channel", "lower_channel",
}


def feature_columns(df: pd.DataFrame, feature_set: str = "all") -> list[str]:
    """Model-input columns: everything except raw OHLCV and targets.

    ``feature_set="all"`` is the original feature set; ``"stationary"`` drops the
    price/volume-level columns in ``LEVEL_FEATURES`` (an addition, not in the original).
    """
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and not c.startswith("target_")]
    if feature_set == "stationary":
        cols = [c for c in cols if c not in LEVEL_FEATURES]
    elif feature_set != "all":
        raise ValueError("feature_set must be 'all' or 'stationary'")
    return cols


@dataclass
class Dataset:
    """Model-ready rows for one forecast horizon.

    X, y and ``past_return`` share one DatetimeIndex. Rows with any NaN feature or
    target are dropped, so the index is the warm-up-trimmed, horizon-trimmed sample.
    ``past_return[t] = Close[t] / Close[t-h] - 1`` is known at t and is used by the
    previous-return baseline.
    """

    X: pd.DataFrame
    y: pd.Series
    past_return: pd.Series
    horizon: int

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(prices: pd.DataFrame, horizon: int = 1, features: pd.DataFrame | None = None,
                  feature_set: str = "all") -> Dataset:
    if features is None:
        features = create_features(prices, horizons=(horizon,))
    cols = feature_columns(features, feature_set)
    tcol = target_column(horizon)
    if tcol not in features.columns:
        features = features.copy()
        features[tcol] = forward_return(features["Close"], horizon)
    past = features["Close"] / features["Close"].shift(horizon) - 1.0
    frame = features[cols + [tcol]].copy()
    frame["__past"] = past
    frame = frame.dropna()
    return Dataset(
        X=frame[cols],
        y=frame[tcol].rename(tcol),
        past_return=frame["__past"].rename(f"past_return_{horizon}d"),
        horizon=horizon,
    )
