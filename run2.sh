#!/bin/bash

# job name
#SBATCH -J ttsr2
# output file (%j expands to jobID)
#SBATCH -o out/%A.out
# total nodes
#SBATCH -N 1
# total cores
#SBATCH -n 1
#SBATCH --requeue
#SBATCH --mem=100G
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --partition=ellis
#SBATCH --time=72:00:00

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate ttsr

"$@"

