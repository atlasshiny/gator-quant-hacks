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

set -euo pipefail

# ---------------------------------------------------------------------
# Parameter Handling
# Symbol, dates, dataset, and schema default to config.yaml when omitted.
# ---------------------------------------------------------------------
SYMBOL="${1:-}"
START_DATE="${2:-}"
END_DATE="${3:-}"
DATASET="${4:-}"
SCHEMA="${5:-}"

# Change to the repository root directory
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Load Conda / Modules if required on HiPerGator
module load conda

# Activate your virtual environment
source activate /blue/ai-workshop/osoriom/gator-quant-hacks/.venv

# Resolve omitted arguments from the validated application configuration.
CONFIG_VALUES="$(uv run python -c '
from python.config import load_config

c = load_config()
print(c.data.symbol)
print(c.data.start)
print(c.data.end)
print(c.data.dataset)
print(c.data.schema_name)
')"
mapfile -t CONFIG_VALUES <<< "$CONFIG_VALUES"

SYMBOL="${SYMBOL:-${CONFIG_VALUES[0]}}"
START_DATE="${START_DATE:-${CONFIG_VALUES[1]}}"
END_DATE="${END_DATE:-${CONFIG_VALUES[2]}}"
DATASET="${DATASET:-${CONFIG_VALUES[3]}}"
SCHEMA="${SCHEMA:-${CONFIG_VALUES[4]}}"

echo "=================================================="
echo "          HiPerGator Slurm Data Ingestion         "
echo "=================================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Symbol : $SYMBOL"
echo "Start Date : $START_DATE"
echo "End Date : $END_DATE"
echo "Dataset : $DATASET"
echo "Schema : $SCHEMA"

# Ensure logs directory exists
mkdir -p logs

# Run Python data ingestion script passing positional arguments
uv run python -m python.data_pipeline.download_databento \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --dataset "$DATASET" \
    --schema "$SCHEMA"