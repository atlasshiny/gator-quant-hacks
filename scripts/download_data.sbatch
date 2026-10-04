#!/bin/bash
#SBATCH --job-name=fetch_databento
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --account=ai-workshop
#SBATCH --qos=ai-workshop

# ---------------------------------------------------------------------
# Parameter Handling
# Default values are used if positional arguments are not provided.
# ---------------------------------------------------------------------
SYMBOL="${1:-ES.c.0}"
START_DATE="${2:-2024-01-08T00:00:00Z}"
END_DATE="${3:-2024-01-12T23:59:59Z}"
DATASET="${4:-GLBX.MDP3}"
SCHEMA="${5:-mbp-10}"

echo "=================================================="
echo "          HiPerGator Slurm Data Ingestion         "
echo "=================================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Symbol : $SYMBOL"
echo "Start Date : $START_DATE"
echo "End Date : $END_DATE"
echo "Dataset : $DATASET"
echo "Schema : $SCHEMA"

# Change to the repository root directory
cd $SLURM_SUBMIT_DIR

# Load Conda / Modules if required on HiPerGator
module load conda

# Activate your virtual environment
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv

# Ensure logs directory exists
mkdir -p logs

# Run Python data ingestion script passing positional arguments
uv run python/download_databento.py \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --dataset "$DATASET" \
    --schema "$SCHEMA"