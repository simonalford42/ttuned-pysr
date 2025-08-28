#!/bin/bash

 # job name
#SBATCH -J ttsr
 # output file (%j expands to jobID)
#SBATCH -o out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis
#SBATCH --time=24:00:00

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate ttsr

python -u "$@"
