"""Optional deep-learning forecasters (Bi-LSTM, Bi-GRU, small Transformer).

Architectures are ported unchanged from the original ``main6.py`` (L226-287).
TensorFlow is imported lazily, so the rest of the package works without it.
Not executed in the validation environment of this repository (TensorFlow is not
installed there); only ``make_sequences`` and the split logic are unit-tested.

Fixes against the original ``train_deep_learning_models`` (L289-373):

* Target alignment. The original paired the window ``data[i : i+L]`` (last day
  t = i+L-1) with ``target[i+L]``, the return from t+1 to t+2, one day later than
  the "next-day" label and than the live forecast. ``make_sequences`` pairs the
  window ending at t with ``target[t]`` (return from t to t+h).
* Train / validation / test. The original split windows 80/20 and early-stopped
  on Keras' ``validation_split`` (the tail of the training windows). Here windows
  are split chronologically into train / validation / test by the date of their
  last row, with a purge gap for multi-day targets. Early stopping uses the
  validation block, and the test block is predicted once.
* Inputs. The original fed price *levels* (Close, SMA_20, SMA_50, EMA_20) into a
  MinMax scaler fitted on the training period. Once test prices leave that range
  the scaled inputs are out of distribution; the saved ^SPX figure shows the
  ensemble's predictions drifting to a constant (R^2 = -4.69). The default input
  list here is stationary (returns, ratios, oscillators). The original list is
  kept as ``ORIGINAL_SEQUENCE_FEATURES`` for comparison.
* The scaler is fitted on the feature rows that belong to training windows only.
"""

from __future__ import annotations

import numpy as np

from .validation import chronological_split, purge_gap, regression_metrics

ORIGINAL_SEQUENCE_FEATURES = [
    "Close", "Volume", "returns", "RSI", "MACD", "volatility_20", "SMA_20",
    "SMA_50", "EMA_20", "BB_position", "momentum_10", "realized_volatility",
]
STATIONARY_SEQUENCE_FEATURES = [
    "returns", "log_returns", "RSI", "BB_position", "momentum_10", "volatility_20",
    "realized_volatility", "volume_ratio", "price_to_sma20", "price_to_sma50",
    "hl_ratio", "atr_percent",
]


def make_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    """Windows of ``seq_len`` rows, each labelled with the target of its last row.

    Returns ``X`` (n_windows, seq_len, n_features), ``y`` (n_windows,) and
    ``end_idx``, the row index of each window's last row. ``y[j] == target[end_idx[j]]``.
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(features)
    if n != len(target):
        raise ValueError("features and target must have the same length")
    end_idx = np.arange(seq_len - 1, n)
    X = np.stack([features[e - seq_len + 1:e + 1] for e in end_idx]) if len(end_idx) else np.empty((0, seq_len, features.shape[1]))
    return X, target[end_idx], end_idx


def sequence_split(n_windows: int, horizon: int, train_frac=0.6, val_frac=0.2):
    return chronological_split(n_windows, train_frac, val_frac, purge_gap(horizon))


def _keras():  # pragma: no cover - needs TensorFlow
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("Deep-learning models need the optional package 'tensorflow'.") from exc
    return tf, keras


def build_lstm_model(input_shape):  # pragma: no cover
    _, keras = _keras()
    L = keras.layers
    model = keras.Sequential([
        L.Input(shape=input_shape),
        L.Bidirectional(L.LSTM(128, return_sequences=True)), L.Dropout(0.3),
        L.Bidirectional(L.LSTM(64)), L.Dropout(0.3),
        L.Dense(32, activation="relu"), L.Dropout(0.2), L.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_gru_model(input_shape):  # pragma: no cover
    _, keras = _keras()
    L = keras.layers
    model = keras.Sequential([
        L.Input(shape=input_shape),
        L.Bidirectional(L.GRU(128, return_sequences=True)), L.Dropout(0.3),
        L.Bidirectional(L.GRU(64)), L.Dropout(0.3),
        L.Dense(32, activation="relu"), L.Dropout(0.2), L.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_transformer_model(input_shape, num_heads: int = 4, ff_dim: int = 128):  # pragma: no cover
    _, keras = _keras()
    L = keras.layers
    inputs = L.Input(shape=input_shape)
    att = L.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(inputs, inputs)
    att = L.Dropout(0.2)(att)
    att = L.LayerNormalization(epsilon=1e-6)(att + inputs)
    ff = L.Dense(ff_dim, activation="relu")(att)
    ff = L.Dropout(0.2)(ff)
    ff = L.Dense(input_shape[1])(ff)
    ff = L.LayerNormalization(epsilon=1e-6)(ff + att)
    pooled = L.GlobalAveragePooling1D()(ff)
    dense = L.Dropout(0.3)(L.Dense(64, activation="relu")(pooled))
    model = keras.Model(inputs=inputs, outputs=L.Dense(1)(dense))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_dl_models(features_df, horizon: int = 1, seq_len: int = 60,
                    feature_list=None, epochs: int = 50, batch_size: int = 32,
                    seed: int = 42) -> dict:  # pragma: no cover - needs TensorFlow
    """Train the three networks; return test metrics per model and for their average."""
    tf, keras = _keras()
    from sklearn.preprocessing import MinMaxScaler

    tf.random.set_seed(seed)
    np.random.seed(seed)
    cols = list(feature_list or STATIONARY_SEQUENCE_FEATURES)
    tcol = f"target_{horizon}d"
    frame = features_df[cols + [tcol]].dropna()
    raw = frame[cols].to_numpy(dtype=float)
    X_raw, y, end_idx = make_sequences(raw, frame[tcol].to_numpy(), seq_len)
    split = sequence_split(len(y), horizon)

    scaler = MinMaxScaler().fit(raw[: end_idx[split.train[-1]] + 1])
    X = scaler.transform(X_raw.reshape(-1, len(cols))).reshape(X_raw.shape)
    Xtr, Xva, Xte = X[split.train], X[split.val], X[split.test]
    ytr, yva, yte = y[split.train], y[split.val], y[split.test]

    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)]
    builders = {"lstm": build_lstm_model, "gru": build_gru_model, "transformer": build_transformer_model}
    preds, val_mse = {}, {}
    for name, build in builders.items():
        model = build((seq_len, len(cols)))
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size,
                  callbacks=cbs, verbose=0)
        val_mse[name] = float(np.mean((model.predict(Xva, verbose=0).ravel() - yva) ** 2))
        preds[name] = model.predict(Xte, verbose=0).ravel()
    preds["dl_average"] = np.mean([preds[k] for k in builders], axis=0)
    preds["baseline:zero"] = np.zeros_like(yte)
    preds["baseline:historical_mean"] = np.full_like(yte, ytr.mean())
    return {
        "features": cols,
        "windows": {"train": len(ytr), "val": len(yva), "test": len(yte)},
        "validation_mse": val_mse,
        "test_metrics": {k: regression_metrics(yte, v) for k, v in preds.items()},
    }
