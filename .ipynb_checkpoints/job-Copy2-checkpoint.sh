#!/bin/bash --login
#SBATCH --job-name=roberta
#SBATCH --account=data-machine
#SBATCH --time=00-01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus=a100:1

module purge
module load CUDA/12.1.1
module load Miniforge3

conda activate active_learning

python classify_selected.py