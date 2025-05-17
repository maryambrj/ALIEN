#!/bin/bash --login
#SBATCH --job-name=agent
#SBATCH --account=data-machine
#SBATCH --time=01-00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G   
#SBATCH --gpus=0

module purge
module load CUDA/12.1.1
module load Miniforge3

conda activate active_learning

# python two_agent_selector.py
python resume.py