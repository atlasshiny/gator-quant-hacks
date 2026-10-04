#!/bin/bash
#SBATCH --job-name=raw_unifier
#SBATCH --output=logs/raw_unifier_%j.out
#SBATCH --error=logs/raw_unifier_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load conda
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv
mkdir -p logs

export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"

uv run python -m python.data_pipeline.data_loader

test -s data/unified_mbp10.bin
echo "Raw data unification completed successfully."
echo "Unified cache: data/unified_mbp10.bin"