"""Command-line interface (replaces the interactive ``input()`` prompts of the original).

    python -m fints generate-data --seed 42 --out data/synthetic_prices.csv
    python -m fints forecast --synthetic --seed 42 --out results/synthetic_demo
    python -m fints forecast --csv data/my_prices.csv --start 2015-01-01 --end 2024-12-31
    python -m fints describe --synthetic --seed 42 --out results/synthetic_describe

Research code only. The outputs are forecast evaluations, not trading signals or
investment advice.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .backtest import backtest
from .data import download_yfinance, generate_synthetic_prices, load_price_csv
from .features import build_dataset, create_features
from .models import resolve_model_names
from .pipeline import ExperimentConfig, fold_summary, run_holdout, run_walk_forward
from .significance import diebold_mariano
from .stats import describe

DISCLAIMER = (
    "Research/educational output only. These are evaluations of statistical forecasts, "
    "not trading signals and not investment advice."
)



def _dm_line(dm: dict) -> list[str]:
    """Markdown line for a Diebold-Mariano result (selected model vs zero forecast)."""
    if dm["statistic"] != dm["statistic"]:  # NaN
        return [f"Diebold-Mariano (selected vs zero, squared error): not defined ({dm.get('note', '')})."]
    verdict = "lower" if dm["statistic"] < 0 else "higher"
    return [f"Diebold-Mariano (selected vs zero, squared error, HLN-corrected, h={dm['horizon']}): "
            f"DM = {dm['statistic']:.3f}, p = {dm['p_value']:.4f}, n = {dm['n']} "
            f"(the selected model has {verdict} mean loss; p does not adjust for model selection)."]

def _load_prices(args):
    if args.csv:
        return load_price_csv(args.csv, start=args.start, end=args.end), f"csv:{Path(args.csv).name}"
    return (
        generate_synthetic_prices(n_days=args.n_days, seed=args.seed, ar_coef=args.ar_coef),
        f"synthetic(n_days={args.n_days}, seed={args.seed}, ar_coef={args.ar_coef})",
    )


def _run_info(args, source: str) -> dict:
    import pandas, scipy, sklearn
    return {
        "command": "python -m fints " + " ".join(shlex.quote(a) for a in args.argv_used),
        "seed": args.seed,
        "data_source": source,
        "fints_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pandas.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "utc_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": DISCLAIMER,
    }


def _fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def _table(metrics: dict, keys: list[str]) -> list[str]:
    lines = ["| forecast | n | RMSE | MAE | R² | R² vs zero | direction acc. | up share |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        m = metrics[k]
        lines.append(f"| {k} | {m['n']} | {m['rmse']:.6f} | {m['mae']:.6f} | {_fmt(m['r2'])} | "
                     f"{_fmt(m['r2_vs_zero'])} | {m['direction_accuracy']:.4f} | {m['up_share']:.4f} |")
    return lines


def _plot(path: Path, index, y, preds: dict, bt: dict | None, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 2 if bt else 1
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n), squeeze=False)
    ax = axes[0, 0]
    ax.plot(index, y, lw=0.8, alpha=0.6, label="realised")
    for k, v in preds.items():
        ax.plot(index, v, lw=1.2, label=k)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if bt:
        ax2 = axes[1, 0]
        ax2.plot(index, bt["equity_curve"], label="threshold rule (research)")
        ax2.plot(index, bt["buy_and_hold_curve"], label="buy and hold")
        ax2.set_title("Growth of 1 unit, out-of-sample period")
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_generate(args):
    df = generate_synthetic_prices(n_days=args.n_days, seed=args.seed, ar_coef=args.ar_coef)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, float_format="%.6f")
    print(f"wrote {len(df)} rows to {out}")


def cmd_download(args):  # pragma: no cover - network
    df = download_yfinance(args.ticker, args.start, args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"wrote {len(df)} rows to {out} (keep local; do not commit Yahoo data)")


def cmd_forecast(args):
    prices, source = _load_prices(args)
    feats = create_features(prices, horizons=tuple(args.horizons))
    cfg = ExperimentConfig(
        model_names=tuple(resolve_model_names(args.models)), n_iter=args.n_iter,
        top_k=args.top_k, use_stacking=not args.no_stacking, seed=args.seed,
        n_folds=args.folds, n_jobs=args.n_jobs,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"run_info": _run_info(args, source), "config": {
        "models": list(cfg.model_names), "n_iter": cfg.n_iter, "top_k": cfg.top_k,
        "stacking": cfg.use_stacking, "holdout_fracs": [cfg.train_frac, cfg.val_frac, 1 - cfg.train_frac - cfg.val_frac],
        "walk_forward_folds": cfg.n_folds, "initial_train_frac": cfg.initial_train_frac,
        "horizons": list(args.horizons), "feature_set": args.feature_set, "backtest_threshold": args.threshold, "backtest_cost": args.cost,
    }, "results": {}}
    md = [f"# Forecast evaluation ({source})", "", f"> {DISCLAIMER}", "",
          f"Command: `{report['run_info']['command']}`  ", f"Seed: {args.seed}", ""]

    for h in args.horizons:
        ds = build_dataset(prices, h, features=feats, feature_set=args.feature_set)
        res = {"rows": len(ds), "start": str(ds.y.index[0].date()), "end": str(ds.y.index[-1].date())}
        print(f"[h={h}] {len(ds)} usable rows")
        if args.mode in ("holdout", "both"):
            fold = run_holdout(ds, cfg)
            res["holdout"] = fold_summary(fold)
            md += [f"## Horizon {h} day(s): hold-out test ({res['holdout']['test_start']} to {res['holdout']['test_end']})",
                   "", f"Selected on validation: **{fold.selected}**", ""]
            keys = ["selected"] + sorted(k for k in fold.test_metrics if k != "selected")
            md += _table(fold.test_metrics, keys) + [""]
            dm = diebold_mariano(fold.y_test, fold.test_predictions["selected"],
                                 fold.test_predictions["baseline:zero"], horizon=h)
            res["holdout"]["dm_selected_vs_zero"] = dm
            md += _dm_line(dm) + [""]
            print(f"  hold-out: selected {fold.selected}; test R² vs zero "
                  f"{fold.test_metrics['selected']['r2_vs_zero']:.4f}")
        if args.mode in ("walk-forward", "both"):
            wf = run_walk_forward(ds, cfg)
            res["walk_forward"] = {
                "metrics": wf.metrics,
                "folds": [fold_summary(f) for f in wf.folds],
            }
            md += [f"## Horizon {h} day(s): walk-forward ({cfg.n_folds} expanding folds, "
                   f"{wf.index[0].date()} to {wf.index[-1].date()})", "",
                   "Selected per fold (on that fold's validation block): "
                   + ", ".join(f.selected for f in wf.folds), ""]
            md += _table(wf.metrics, list(wf.metrics)) + [""]
            dm = diebold_mariano(wf.y, wf.predictions["selected"], wf.predictions["baseline:zero"], horizon=h)
            res["walk_forward"]["dm_selected_vs_zero"] = dm
            md += _dm_line(dm) + [""]
            print(f"  walk-forward: R² vs zero {wf.metrics['selected']['r2_vs_zero']:.4f}")
            bt = None
            if h == 1:
                bt = backtest(wf.predictions["selected"], wf.y, args.threshold, args.cost)
                res["walk_forward"]["backtest"] = {k: v for k, v in bt.items() if not k.endswith("curve")}
                s, b = bt["strategy"], bt["buy_and_hold"]
                md += ["### Research backtest of the walk-forward forecasts (1-day)", "",
                       f"Rule: long if forecast > {args.threshold}, short if < -{args.threshold}, else flat; "
                       f"cost {args.cost} per unit of position change.", "",
                       "| | total return | Sharpe (ann.) | max drawdown | exposure |", "|---|---|---|---|---|",
                       f"| threshold rule | {s['total_return']:.4f} | {s['sharpe_annualised']:.3f} | {s['max_drawdown']:.4f} | {s['exposure']:.3f} |",
                       f"| buy and hold | {b['total_return']:.4f} | {b['sharpe_annualised']:.3f} | {b['max_drawdown']:.4f} | 1.000 |", ""]
            if not args.no_plot:
                _plot(out / f"walk_forward_h{h}.png", wf.index, wf.y,
                      {k: v for k, v in wf.predictions.items() if k in ("selected", "baseline:historical_mean")},
                      bt, f"Walk-forward out-of-sample forecasts, horizon {h}d ({source})")
        report["results"][f"h{h}"] = res

    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'metrics.json'} and {out / 'summary.md'}")
    if args.with_dl:  # pragma: no cover - needs TensorFlow
        from .dl import train_dl_models
        dl = {f"h{h}": train_dl_models(feats, horizon=h, seed=args.seed) for h in args.horizons}
        (out / "dl_metrics.json").write_text(json.dumps(dl, indent=2, default=str))


def cmd_describe(args):
    prices, source = _load_prices(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"run_info": _run_info(args, source), "stats": describe(prices)}
    (out / "stats.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["stats"], indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fints", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--csv", help="local OHLCV CSV (Date, Open, High, Low, Close, Volume)")
        g.add_argument("--synthetic", action="store_true", help="use the deterministic synthetic generator")
        sp.add_argument("--start", help="first date (inclusive) when using --csv")
        sp.add_argument("--end", help="last date (inclusive) when using --csv")
        sp.add_argument("--n-days", type=int, default=2000)
        sp.add_argument("--ar-coef", type=float, default=0.0,
                        help="planted AR(1) signal in synthetic data (0 = none)")
        sp.add_argument("--seed", type=int, default=42)

    g = sub.add_parser("generate-data", help="write a synthetic OHLCV CSV")
    g.add_argument("--n-days", type=int, default=2000)
    g.add_argument("--ar-coef", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("download", help="(optional, needs yfinance + network) fixed-date download")
    d.add_argument("--ticker", required=True)
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_download)

    f = sub.add_parser("forecast", help="hold-out and walk-forward forecast evaluation")
    data_args(f)
    f.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    f.add_argument("--models", default="rf,gb,et,ridge",
                   help="comma list: rf,gb,et,ridge[,xgb,lgbm,cat if installed]")
    f.add_argument("--n-iter", type=int, default=0,
                   help="RandomizedSearchCV iterations per model (0 = fixed defaults; original used 20)")
    f.add_argument("--top-k", type=int, default=40)
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--mode", choices=["holdout", "walk-forward", "both"], default="both")
    f.add_argument("--no-stacking", action="store_true")
    f.add_argument("--feature-set", choices=["all", "stationary"], default="all",
                   help="all = original ~59 features; stationary = drop price/volume-level columns")
    f.add_argument("--n-jobs", type=int, default=1, help="threads for tree models (-1 = all cores)")
    f.add_argument("--threshold", type=float, default=0.01)
    f.add_argument("--cost", type=float, default=0.001)
    f.add_argument("--no-plot", action="store_true")
    f.add_argument("--with-dl", action="store_true", help="also train LSTM/GRU/Transformer (needs tensorflow)")
    f.add_argument("--out", default="results/run")
    f.set_defaults(func=cmd_forecast)

    s = sub.add_parser("describe", help="descriptive statistics (risk, normality, Hurst, GARCH, regimes)")
    data_args(s)
    s.add_argument("--out", default="results/describe")
    s.set_defaults(func=cmd_describe)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.argv_used = list(argv) if argv is not None else sys.argv[1:]
    args.func(args)


if __name__ == "__main__":
    main()
