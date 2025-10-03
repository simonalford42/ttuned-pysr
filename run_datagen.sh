#!/bin/bash

# job name
#SBATCH -J datagen
# output file (%j expands to jobID)
#SBATCH -o out/datagen_%A.out
# total nodes
#SBATCH -N 1
# total cores
#SBATCH -n 1
#SBATCH --requeue
#SBATCH --mem=16G
#SBATCH --partition=ellis
#SBATCH --time=4:00:00

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate ttsr

python -u "$@"
