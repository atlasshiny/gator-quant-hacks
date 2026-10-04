#!/bin/bash
#SBATCH --job-name=features_xgb
#SBATCH --output=logs/features_xgb_%j.out
#SBATCH --error=logs/features_xgb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs output/features output/ga_inputs output/models

module purge
module load conda cuda/12.8.1
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv

test -s data/unified_mbp10.bin

uv run python -m python.prepare_xgb_features

echo "Feature preparation completed successfully."

uv run python -m python.model.train_xgb

test -s output/features/stationary_features.parquet
test -s output/ga_inputs/features.bin
test -s output/ga_inputs/returns.bin
test -s output/models/xgboost_final.json

echo "Feature engineering and XGBoost training completed successfully."
echo "Stationary features: output/features/stationary_features.parquet"
echo "GA features: output/ga_inputs/features.bin"
echo "GA returns: output/ga_inputs/returns.bin"
echo "XGBoost model: output/models/xgboost_final.json"
