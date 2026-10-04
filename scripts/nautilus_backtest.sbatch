#!/bin/bash
#SBATCH --job-name=nautilus_backtest
#SBATCH --output=logs/nautilus_backtest_%j.out
#SBATCH --error=logs/nautilus_backtest_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=06:00:00

# OOS replay of the frozen strategy (evaluation/backtest/nautilus_backtest.py).

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export REPO
cd "$REPO"
mkdir -p logs

module purge
module load conda
source "$REPO/.venv/bin/activate"

# Inputs are absolute so each run can write into its own folder.
export CATALOG_DIR="${CATALOG_DIR:-$REPO/catalog_january}"
export MODEL_PATH="${MODEL_PATH:-$REPO/output/models/xgboost_january.json}"
export EVENT_FEATURES_PATH="${EVENT_FEATURES_PATH:-$REPO/output/features/stationary_features_january.parquet}"
export GA_BEST_PATH="${GA_BEST_PATH:-$REPO/output/ga_results_january/best_strategy.json}"
export BUNDLE_PATH="${BUNDLE_PATH:-$REPO/output/ga_results_january/frozen_bundle.json}"
export DATA_DIR="${DATA_DIR:-$REPO/data/GLBX.MDP3}"
export PUBLISHERS_JSON="${PUBLISHERS_JSON:-$REPO/data/example.json}"
export INSTRUMENT_ID="${INSTRUMENT_ID:-ESH4.GLBX}"
export PYTHONPATH="$REPO/evaluation:$REPO/evaluation/backtest${PYTHONPATH:+:$PYTHONPATH}"

# Pre-execution checks
test -s "$GA_BEST_PATH" || { echo "Error: Missing GA best strategy file at $GA_BEST_PATH" >&2; exit 1; }

test -s "$BUNDLE_PATH" || {
    uv run --no-sync --project "$REPO" python - <<'PY'
import json
import os
from pathlib import Path
import polars as pl

best_path = Path(os.environ["GA_BEST_PATH"])
features_path = Path(os.environ["EVENT_FEATURES_PATH"])
bundle_path = Path(os.environ["BUNDLE_PATH"])
metadata = {"ts_event", "ts_recv", "mid_price", "ask_px_00", "bid_px_00"}
columns = pl.read_parquet_schema(features_path).names()
feature_names = [name for name in columns if name not in metadata]
if len(feature_names) != 25:
    raise ValueError(f"Expected 25 trained feature columns, found {len(feature_names)}")
best = json.loads(best_path.read_text())["strategy"]
bundle = {
    "feature_names_28": feature_names + ["xgb_p_short", "xgb_p_flat", "xgb_p_long"],
    "xgb_input_names": feature_names,
    "mask": [int(v) for v in best["feature_flags"]],
    "mean": [0.0] * 28,
    "std": [1.0] * 28,
    "tau": best["threshold_param"] / 1000.0,
    "lookback": int(best["lookback_window"]),
}
if len(bundle["mask"]) != 28:
    raise ValueError("GA chromosome does not contain 28 feature flags")
bundle_path.write_text(json.dumps(bundle, indent=2))
print(f"Built GA bundle: {bundle_path}")
PY
}

test -s "$MODEL_PATH" || { echo "Error: Missing XGBoost model at $MODEL_PATH" >&2; exit 1; }
test -s "$EVENT_FEATURES_PATH" || { echo "Error: Missing event features at $EVENT_FEATURES_PATH" >&2; exit 1; }

# Download publishers.json directly from Databento metadata API if missing
if [[ ! -s "$PUBLISHERS_JSON" ]]; then
    mkdir -p "$(dirname "$PUBLISHERS_JSON")"
    curl --fail --silent --show-error --location \
        "https://hist.databento.com/v0/metadata.list_publishers" \
        -o "$PUBLISHERS_JSON"
fi
test -s "$PUBLISHERS_JSON" || { echo "Error: Failed to fetch publishers.json" >&2; exit 1; }

if [[ ! -d "$CATALOG_DIR" ]] || ! find "$CATALOG_DIR" -type f -name '*.parquet' -print -quit | grep -q .; then
    if ! find "$DATA_DIR" -maxdepth 1 -type f -name '*definition*.dbn.zst' -print -quit | grep -q .; then
        uv run --no-sync --project "$REPO" python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv
import databento as db

load_dotenv(dotenv_path=Path(os.environ["REPO"]) / ".env")
key = os.environ.get("DATABENTO_API_KEY")
if not key:
    raise RuntimeError("DATABENTO_API_KEY is required to download the ES definition file")
out = Path(os.environ["DATA_DIR"]) / "GLBX.MDP3_definition_ESH4.dbn.zst"
db.Historical(key).timeseries.get_range(
    dataset="GLBX.MDP3", schema="definition", symbols=["ESH4"],
    stype_in="raw_symbol", start="2024-01-08", end="2024-01-09", path=str(out),
)
print(f"Downloaded definition: {out}")
PY
    fi
    DATA_DIR="$DATA_DIR" CATALOG_DIR="$CATALOG_DIR" PUBLISHERS_JSON="$PUBLISHERS_JSON" INSTRUMENT_ID="$INSTRUMENT_ID" \
        uv run --no-sync --project "$REPO" python "$REPO/evaluation/backtest/nautilus_catalog.py"
fi

# Threading setup
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LATENCIES=(0 5 20 50)
    export LATENCY_MS="${LATENCIES[$SLURM_ARRAY_TASK_ID]}"
    ALSO_RUN_IS=False
    RUN_DIR="$REPO/output/backtest/latency_${LATENCY_MS}ms_${SLURM_ARRAY_JOB_ID}"
else
    export LATENCY_MS="${LATENCY_MS:-5}"
    ALSO_RUN_IS="${ALSO_RUN_IS:-False}"
    RUN_DIR="$REPO/output/backtest/run_${SLURM_JOB_ID:-local}"
fi

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
echo "Host $(hostname) | cpus ${SLURM_CPUS_PER_TASK} | latency ${LATENCY_MS} ms | IS too: ${ALSO_RUN_IS}"
echo "Output: $RUN_DIR"

uv run --no-sync --project "$REPO" python -c "
import nautilus_backtest as nb
nb.run_oos(also_run_is=${ALSO_RUN_IS})
" 2>&1 | tee backtest_stdout.txt

test -s equity_curve.png
test -s signal_log_oos.csv
echo "Backtest completed successfully: $RUN_DIR"
echo "Check peak memory with: seff ${SLURM_JOB_ID:-<jobid>}"