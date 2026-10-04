#!/bin/bash
#SBATCH --job-name=raw_unifier
#SBATCH --output=logs/raw_unifier_%j.out
#SBATCH --error=logs/raw_unifier_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=01:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load conda cuda/12.8.1
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv
mkdir -p logs

uv run python -m python.data_pipeline.data_loader

test -s data/unified_mbp10.bin
echo "Raw data unification completed successfully."
echo "Unified cache: data/unified_mbp10.bin"