#!/bin/bash --login
#SBATCH --job-name=roberta
#SBATCH --time=00-01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
###SBATCH --cpus-per-task=16
###SBATCH --mem=100G   
#SBATCH --account=data-machine
#SBATCH --gpus=a100:1
###SBATCH --nodelist=lac-[000-023,032-191,200-223,254,255,276,277,318-341,350-369,372,372-445]
###nal-[000-010],nif-[000-005]

module purge
module load CUDA/12.1.1
module load Miniforge3

conda activate active_learning

python classify_selected.py