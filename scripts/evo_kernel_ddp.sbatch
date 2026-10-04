#!/bin/bash
#SBATCH --job-name=evo_kernel_ddp
#SBATCH --output=logs/evo_run_%j.out   # Standard output log (%j inserts job ID)
#SBATCH --error=logs/evo_run_%j.err    # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL     # Receive email notifications
#SBATCH --nodes=1                      # DDP localhost restriction requires 1 node
#SBATCH --ntasks=1                     # 1 Master task for mp.spawn
#SBATCH --cpus-per-task=16             # Allocate enough CPU cores for fast memmap loading
#SBATCH --mem=64gb                     # System RAM
#SBATCH --partition=hpg-b200           # Target the DGX B200 partition
#SBATCH --gpus=b200:5                  # Request exactly 5 GPUs
#SBATCH --time=01:00:00                # Wall clock time limit (HH:MM:SS)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
test -s data/ga_inputs/features.bin
test -s data/ga_inputs/returns.bin

# Load the necessary environment modules
module purge
module load conda cuda/12.8.1

# Optimize NCCL (NVIDIA Collective Communications Library) for PCIe/NVLink
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0 
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME='=bond0'
export NCCL_NET_MERGE_LEVEL=LOC
# Execute the driver script
echo "Starting PyTorch DDP Evolution on $SLURM_JOB_NODELIST"
uv run python python/model/distributed_ga.py

test -s data/ga_outputs/best_strategy.json
test -s data/ga_outputs/fitness_history.csv
echo "GA completed successfully."
echo "Best strategy: data/ga_outputs/best_strategy.json"
echo "Fitness history: data/ga_outputs/fitness_history.csv"