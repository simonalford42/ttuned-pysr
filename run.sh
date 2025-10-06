#!/bin/bash

# job name
#SBATCH -J ttsr2
#SBATCH -o out/%A.out
# total nodes
#SBATCH -N 1
# total tasks
#SBATCH -n 1
# cpus per task
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem-per-cpu=50G
# disable #SBATCH --gres=gpu:nvidia_rtx_a6000:1
# disable #SBATCH --partition=ellis
#SBATCH --partition=default_partition
#SBATCH --time=72:00:00

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate ttsr

"$@"
