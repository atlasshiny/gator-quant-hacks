#!/bin/bash
#SBATCH --job-name=xgb_only_backtest
#SBATCH --output=logs/xgb_only_backtest_%j.out
#SBATCH --error=logs/xgb_only_backtest_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=48:00:00

# XGBoost-only baseline (no GA) on the one month of ES MBP-10 data, replayed in Nautilus.
#
#   sbatch scripts/xgb_only_backtest.sbatch
#
# 1. Train XGBoost on the in-sample days only (CPU), using the 21 features Nautilus can
#    compute from depth snapshots. The existing output/models/xgboost_january.json cannot
#    be used: it takes 4 event-only inputs (cancel_add_ratio, vpin, cancel_trade_ratio,
#    message_intensity) the strategy refuses, and it was fit on the OOS days too.
# 2. Rule: buy if P(buy)-P(sell) > tau, sell if < -tau. tau = TAU_Q quantile of |score| on
#    the last in-sample days (held out from fitting). Fixed before the OOS replay.
# 3. Build the Nautilus catalog from the January .dbn.zst files (skipped if it exists).
# 4. Replay OOS once (and IS, to show the gap). Results land in output/xgb_only_<jobid>/.

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p logs

module purge
module load conda
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv

export FIRST_DAY="${FIRST_DAY:-2024-01-08}"
export LAST_DAY="${LAST_DAY:-2024-01-31}"
export OOS_FRACTION="${OOS_FRACTION:-0.3}"
export EMBARGO="${EMBARGO:-5min}"
export TAU_Q="${TAU_Q:-0.99}"
export LATENCY_MS="${LATENCY_MS:-5}"
export ALSO_RUN_IS="${ALSO_RUN_IS:-True}"
export DATA_DIR="${DATA_DIR:-$REPO/data/GLBX.MDP3}"
export CATALOG_DIR="${CATALOG_DIR:-$REPO/catalog_january}"
export INSTRUMENT_ID="${INSTRUMENT_ID:-ESH4.GLBX}"
export OUT="$REPO/output/xgb_only_${SLURM_JOB_ID:-local}"
export BUNDLE_PATH="$OUT/frozen_bundle.json"
export MODEL_PATH="$OUT/xgboost_is.json"
export PYTHONPATH="$REPO:$REPO/evaluation:$REPO/evaluation/backtest${PYTHONPATH:+:$PYTHONPATH}"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT"
echo "Days $FIRST_DAY -> $LAST_DAY | OOS fraction $OOS_FRACTION | tau quantile $TAU_Q | latency $LATENCY_MS ms"
echo "Output: $OUT"

# ---------------------------------------------------------------- 1-2. train + freeze
uv run python - <<'EOF'
import json, os
import numpy as np, pandas as pd, polars as pl, xgboost as xgb
from python.config import load_config
from split_sample import SampleSplitter
from nautilus_strategy import EVENT_ONLY_FEATURES, SNAPSHOT_FEATURES

c = load_config().xgboost
H, fee, out = c.target_horizon_events, c.fee_threshold_bps / 1e4, os.environ["OUT"]

# Same labels as train_xgb.py: 100-event forward mid return vs the 1.5 bps barrier.
df = pl.read_parquet(c.features_path)
mid = (pl.col("ask_px_00") + pl.col("bid_px_00")) / 2.0
df = df.with_columns(((mid.shift(-H) - mid) / mid).alias("fr")).drop_nulls("fr")
fr = df["fr"].to_numpy()
y = np.select([fr < -fee, fr > fee], [0, 2], 1).astype(np.int32)

ignore = {"ts_event", "ts_recv", "mid_price", "ask_px_00", "bid_px_00", "fr"}
feats = [n for n in df.columns if n not in ignore and n not in EVENT_ONLY_FEATURES]
assert len(feats) == 21 and set(feats) <= set(SNAPSHOT_FEATURES), feats

# CME trading day (17:00 CT roll), same rule as split_sample.add_trading_day.
ts = df["ts_event"]
ns = ts.dt.epoch("ns") if ts.dtype.is_temporal() else ts.cast(pl.Float64).cast(pl.Int64)
ts = pd.to_datetime(ns.to_numpy(), utc=True)
assert ts[0].year == 2024, f"unexpected ts_event encoding: {ts[0]}"
day = (ts.tz_convert("America/Chicago") + pd.Timedelta(hours=7)).normalize().tz_localize(None)

# Same day split the backtest uses; last ~20% of IS days are held out for early stopping + tau.
all_days = pd.bdate_range(os.environ["FIRST_DAY"], os.environ["LAST_DAY"])
is_days, oos_days = SampleSplitter(float(os.environ["OOS_FRACTION"])).split_day_list(all_days)
n_val = max(1, round(0.2 * len(is_days)))
fit_days, val_days = is_days[:-n_val], is_days[-n_val:]
fit = np.flatnonzero(day.isin(fit_days))[:-H]   # drop labels that look past the window
val = np.flatnonzero(day.isin(val_days))[:-H]
print(f"fit {fit_days[0].date()}..{fit_days[-1].date()} {len(fit):,} rows | "
      f"val {val_days[0].date()}..{val_days[-1].date()} {len(val):,} rows | "
      f"OOS {oos_days[0].date()}..{oos_days[-1].date()} (never touched here)")

X = df.select(feats).to_numpy().astype(np.float32)
w = np.where(y != 1, c.directional_class_weight, 1.0).astype(np.float32)
del df
params = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
              tree_method="hist", device="cpu", nthread=int(os.environ["POLARS_MAX_THREADS"]),
              max_depth=c.max_depth, learning_rate=c.learning_rate,
              subsample=c.subsample, colsample_bytree=c.colsample_bytree)
dfit = xgb.DMatrix(X[fit], label=y[fit], weight=w[fit])
dval = xgb.DMatrix(X[val], label=y[val], weight=w[val])
model = xgb.train(params, dfit, c.num_boost_round, evals=[(dval, "val")],
                  early_stopping_rounds=c.early_stopping_rounds, verbose_eval=100)
best_it = model.best_iteration
model = model[: best_it + 1]

p = model.predict(xgb.DMatrix(X[val]))
score = p[:, 2] - p[:, 0]
tau = float(np.quantile(np.abs(score), float(os.environ["TAU_Q"])))
model.save_model(os.environ["MODEL_PATH"])

# Bundle: only the xgb_score column is on, unstandardized (mean 0, std 1), so S = P(buy)-P(sell).
names = feats + ["xgb_score"] + ["pad"] * (28 - len(feats) - 1)
json.dump({"feature_names_28": names, "xgb_input_names": feats,
           "mask": [int(n == "xgb_score") for n in names],
           "mean": [0.0] * 28, "std": [1.0] * 28, "tau": tau, "lookback": 0},
          open(os.environ["BUNDLE_PATH"], "w"), indent=2)
json.dump({"tau": tau, "tau_quantile": float(os.environ["TAU_Q"]),
           "best_iteration": int(best_it),
           "fit_rows": int(len(fit)), "val_rows": int(len(val)),
           "val_class_share": np.bincount(y[val], minlength=3).tolist(),
           "val_accuracy": float((p.argmax(1) == y[val]).mean()),
           "is_days": [str(d.date()) for d in is_days], "oos_days": [str(d.date()) for d in oos_days],
           "features": feats}, open(f"{out}/train_summary.json", "w"), indent=2)
print(f"tau = {tau:.4f} ({float(os.environ['TAU_Q']):.0%} quantile of |P(buy)-P(sell)| on val)")
EOF
test -s "$MODEL_PATH"
test -s "$BUNDLE_PATH"

# ---------------------------------------------------------------- 3. Nautilus catalog
if [[ ! -d "$CATALOG_DIR" ]]; then
    ls "$DATA_DIR"/*mbp-10*2024-01-*.dbn.zst > /dev/null   # fail early if the raw files are elsewhere
    if ! ls "$DATA_DIR"/*definition*.dbn.zst > /dev/null 2>&1; then
        echo "No ESH4 definition file in $DATA_DIR; pulling it from Databento (tiny)."
        uv run python - <<'EOF'
import os
import databento as db
from dotenv import load_dotenv
load_dotenv()
db.Historical(os.environ["DATABENTO_API_KEY"]).timeseries.get_range(
    dataset="GLBX.MDP3", schema="definition", symbols=["ESH4"], stype_in="raw_symbol",
    start="2024-01-08", end="2024-01-09",
    path=os.path.join(os.environ["DATA_DIR"], "GLBX.MDP3_definition_ESH4.dbn.zst"))
EOF
    fi
    uv run python -c "
import nautilus_catalog as nc
nc.build_catalog(depth_pattern='*mbp-10*2024-01-*.dbn.zst')
"
fi

# ---------------------------------------------------------------- 4. OOS replay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$OUT"
uv run --no-sync --project "$REPO" python -c "
import nautilus_backtest as nb
nb.run_oos(first_day='$FIRST_DAY', last_day='$LAST_DAY', oos_fraction=$OOS_FRACTION,
           embargo='$EMBARGO', also_run_is=$ALSO_RUN_IS)
" 2>&1 | tee backtest_stdout.txt

test -s equity_curve.png
test -s signal_log_oos.csv
echo "Done: $OUT  (train_summary.json, oos_tearsheet.html, equity_curve.png, signal_log_oos.csv, backtest_stdout.txt)"
