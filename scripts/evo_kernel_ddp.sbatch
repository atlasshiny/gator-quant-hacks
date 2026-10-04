#!/bin/bash
#SBATCH --job-name=evo_kernel_ddp
#SBATCH --output=logs/evo_run_january_%j.out
#SBATCH --error=logs/evo_run_january_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # Receive email notifications
#SBATCH --nodes=1                      # DDP localhost restriction requires 1 node
#SBATCH --ntasks=1                     # 1 Master task for mp.spawn
#SBATCH --cpus-per-task=16             # Allocate enough CPU cores for fast memmap loading
#SBATCH --mem=128gb                    # System RAM for two spawned loaders
#SBATCH --partition=hpg-b200           # Target the DGX B200 partition
#SBATCH --gpus=b200:2                  # Request exactly 2 GPUs
#SBATCH --time=02:00:00                # Wall clock time limit (HH:MM:SS)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs output/ga_results_january
test -s output/ga_inputs_january/features.bin
test -s output/ga_inputs_january/returns.bin

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
uv run python -m python.model.distributed_ga

test -s output/ga_results_january/best_strategy.json
test -s output/ga_results_january/stage1_survivors.json
test -s output/ga_results_january/fitness_history.csv
echo "GA completed successfully."
echo "Best strategy: output/ga_results_january/best_strategy.json"
echo "Stage 1 survivors: output/ga_results_january/stage1_survivors.json"
echo "Fitness history: output/ga_results_january/fitness_history.csv"