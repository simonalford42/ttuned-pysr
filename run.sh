#!/bin/bash

# job name
#SBATCH -J ttsr2
#SBATCH -o out/%A.out
# total nodes
#SBATCH -N 1
# total tasks
#SBATCH -n 1
# cpus per task
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --mem=50G
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --partition=ellis
#SBATCH --time=72:00:00

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate ttsr

# Ensure single-node DDP rendezvous is isolated per job
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((29500 + (RANDOM % 1000)))}

# Safer defaults for common single-node clusters without IB
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}

"$@"
