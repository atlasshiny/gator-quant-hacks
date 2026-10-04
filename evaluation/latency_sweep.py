"""
latency_sweep.py

Sensitivity of the out-of-sample result to the assumed order latency.

Each latency runs in its own Python process, because nautilus_backtest.py reads
LATENCY_MS at import time and Nautilus keeps global state between runs.

Usage:
    python latency_sweep.py                        # default: 0 5 20 50 ms
    python latency_sweep.py --latencies 0 2 10 100
s
Outputs (in ./latency_sweep/):
    latency_sweep.csv       one row per latency: entries + all performance stats
    latency_sweep.tex       LaTeX table (subset of columns) for the paper
    latency_<ms>ms/         that run's tearsheet, equity curve and signal log

How to use it honestly:
    * Fix your headline latency BEFORE looking at results, and report ALL rows.
    * This is a robustness check of an assumption, not a tuning knob. Never pick
      the latency, or any other setting, because it looks best on OOS.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path("latency_sweep")
ARTIFACTS = ("oos_tearsheet.html", "equity_curve.png", "signal_log_oos.csv")


def _py(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def worker(latency_ms: float, out_json: str) -> None:
    """Runs ONE OOS replay. LATENCY_MS must be set before importing nautilus_backtest."""
    os.environ["LATENCY_MS"] = str(latency_ms)
    import backtest.nautilus_backtest as nb
    from metrics import performance_stats

    _, oos_daily = nb.run_oos(also_run_is=False)
    stats = performance_stats(oos_daily, nb.COSTS)

    n_entries = 0
    log_path = Path("signal_log_oos.csv")
    if log_path.exists():
        n_entries = max(sum(1 for _ in open(log_path)) - 1, 0)  # minus header

    row = {"latency_ms": latency_ms, "n_entries": n_entries}
    row.update({k: _py(v) for k, v in dict(stats).items()})
    Path(out_json).write_text(json.dumps(row))


def run_sweep(latencies: list[float]) -> pd.DataFrame:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for ms in latencies:
        tag = f"latency_{ms:g}ms"
        run_dir = OUT_DIR / tag
        run_dir.mkdir(exist_ok=True)
        out_json = run_dir / "stats.json"
        print(f"\n=== Latency {ms:g} ms ===", flush=True)

        proc = subprocess.run(
            [sys.executable, __file__, "--worker", str(ms), "--out", str(out_json)]
        )
        if proc.returncode != 0 or not out_json.exists():
            print(f"[ERROR] run at {ms:g} ms failed (exit {proc.returncode}); recorded as failed.")
            rows.append({"latency_ms": ms, "failed": True})
            continue

        for name in ARTIFACTS:                     # keep each run's outputs
            if Path(name).exists():
                shutil.move(name, run_dir / name)
        rows.append(json.loads(out_json.read_text()))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "latency_sweep.csv", index=False)

    keys = ("latency", "entries", "sharpe", "return", "drawdown", "turnover")
    cols = [c for c in df.columns if any(k in c.lower() for k in keys)]
    (OUT_DIR / "latency_sweep.tex").write_text(
        df[cols].to_latex(index=False, float_format="%.3f", na_rep="--")
    )
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latencies", type=float, nargs="+", default=[0, 5, 20, 50])
    ap.add_argument("--worker", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=str, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker is not None:
        worker(args.worker, args.out)
    else:
        pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
        table = run_sweep(args.latencies)
        print("\nLatency sensitivity (out-of-sample, net of fees):")
        print(table.to_string(index=False))
        print(f"\nSaved {OUT_DIR / 'latency_sweep.csv'} and {OUT_DIR / 'latency_sweep.tex'}")