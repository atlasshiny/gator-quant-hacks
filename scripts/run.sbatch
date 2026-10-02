#!/bin/bash
#SBATCH --job-name=gator-quant-hacks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:30:00

hostname; date; pwd

# Modify entry point file/executable here
# python -m main.py
# source ./main