#!/bin/bash

#SBATCH --job-name=msl_final
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=2-0
#SBATCH --partition batch_grad
#SBATCH -o slurm.out


. /data/dkgud111/anaconda3/etc/profile.d/conda.sh
conda activate daformer



bash ./scripts/MSL.sh


