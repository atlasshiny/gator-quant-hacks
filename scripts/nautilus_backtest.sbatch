#!/bin/bash
#SBATCH --job-name=nautilus_backtest
#SBATCH --output=logs/nautilus_backtest_%A_%a.out
#SBATCH --error=logs/nautilus_backtest_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256gb
#SBATCH --time=12:00:00

# OOS replay of the frozen strategy (evaluation/backtest/nautilus_backtest.py).
# Submit from the repo root, like the other scripts/:
#
#   sbatch scripts/nautilus_backtest.sbatch                      # OOS + IS, LATENCY_MS=5
#   LATENCY_MS=20 sbatch scripts/nautilus_backtest.sbatch        # one run, other latency
#   sbatch --array=0-3 --mem=128gb scripts/nautilus_backtest.sbatch
#                                                                # latency sweep, OOS only,
#                                                                # one latency per task, in parallel
#
# One Nautilus backtest is single-threaded: the engine replays events on one core.
# Extra cores only help Parquet decoding and the OS, so 4 is plenty. To use more
# cores, run more backtests at once (the --array form above), not one bigger job.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p logs

module purge
module load conda
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv

# Inputs are absolute so each run can write into its own folder.
export CATALOG_DIR="${CATALOG_DIR:-$REPO/catalog}"
export BUNDLE_PATH="${BUNDLE_PATH:-$REPO/data/ga_inputs/frozen_bundle.json}"
export MODEL_PATH="${MODEL_PATH:-$REPO/data/ga_inputs/xgboost_final.json}"
export INSTRUMENT_ID="${INSTRUMENT_ID:-ESH4.GLBX}"

test -d "$CATALOG_DIR"   # build it first: python evaluation/backtest/nautilus_catalog.py
test -s "$BUNDLE_PATH"
test -s "$MODEL_PATH"

# XGBoost scores one row per book update; threading that only adds overhead and
# would fight the engine for cores. Arrow/Polars may use the rest for reading Parquet.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# metrics.py and split_sample.py live in evaluation/, the strategy in evaluation/backtest/
export PYTHONPATH="$REPO/evaluation:$REPO/evaluation/backtest${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LATENCIES=(0 5 20 50)
    export LATENCY_MS="${LATENCIES[$SLURM_ARRAY_TASK_ID]}"
    ALSO_RUN_IS=False
    RUN_DIR="$REPO/output/backtest/latency_${LATENCY_MS}ms_${SLURM_ARRAY_JOB_ID}"
else
    export LATENCY_MS="${LATENCY_MS:-5}"
    ALSO_RUN_IS=True
    RUN_DIR="$REPO/output/backtest/run_${SLURM_JOB_ID:-local}"
fi

# The backtest writes tearsheet, equity curve and signal logs to the working directory.
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
