# Data

No market data is stored in this repository.

* **Synthetic (default, offline):** `PYTHONPATH=src python3 -m fints generate-data --seed 42 --out data/synthetic_prices.csv`
  writes a deterministic OHLCV series (Markov-switching drift, GARCH(1,1) variance,
  Student-t shocks; see `src/fints/data.py`). The same seed always gives the same file.
* **Your own CSV:** columns `Date, Open, High, Low, Close, Volume`, one row per trading day.
  Use `--csv path --start YYYY-MM-DD --end YYYY-MM-DD` to fix the sample.
* **Yahoo Finance (optional, needs `yfinance` and network):**
  `PYTHONPATH=src python3 -m fints download --ticker SPY --start 2015-01-01 --end 2024-12-31 --out data/spy.csv`.
  Yahoo data is subject to Yahoo's terms of use. Keep it local; `data/*.csv` is git-ignored.
