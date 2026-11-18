"""fints: leakage-aware financial time-series analysis and return forecasting.

Research code only. Nothing in this package is investment advice.

Modules
-------
data        synthetic OHLCV generator, local CSV loader, optional yfinance download
features    backward-looking technical features and forward-return targets
stats       descriptive statistics (risk, normality, Hurst, Ljung-Box, GARCH,
            Markov switching, PELT change points; heavy deps are optional)
validation  chronological train/validation/test splits with purging,
            walk-forward folds, evaluation metrics
baselines   naive forecasts (zero, previous return, historical mean)
models      candidate regressors, time-series stacking, validation-fitted weights,
            validation-only model selection
pipeline    hold-out and walk-forward experiments
backtest    research backtest of a threshold long/short rule vs buy-and-hold
dl          optional LSTM/GRU/Transformer models (needs TensorFlow)
"""

__version__ = "0.2.0"
