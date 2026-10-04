#!/bin/bash
#SBATCH --job-name=fetch_databento
#SBATCH --output=logs/download_%A_%a.out
#SBATCH --error=logs/download_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --account=ai-workshop
#SBATCH --qos=ai-workshop

set -euo pipefail

RANGES=(
  "2024-01-08T00:00:00Z 2024-02-01T00:00:00Z"
  "2024-02-01T00:00:00Z 2024-03-01T00:00:00Z"
  "2024-03-01T00:00:00Z 2024-04-01T00:00:00Z"
  "2024-04-01T00:00:00Z 2024-05-01T00:00:00Z"
  "2024-05-01T00:00:00Z 2024-06-01T00:00:00Z"
  "2024-06-01T00:00:00Z 2024-07-08T00:00:00Z"
)

read -r START_DATE END_DATE <<< "${RANGES[$SLURM_ARRAY_TASK_ID]}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv

mkdir -p logs

uv run python -m python.data_pipeline.download_databento \
  --symbol ES.c.0 \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --dataset GLBX.MDP3 \
  --schema mbp-10