#!/bin/bash

# 8/12/25
sbatch -J ttsr --partition ellis run.sh training/train_one_step.py --config training/onestep-s.json

# 8/11/25
# sbatch -J ttsr --partition ellis run.sh training/train_one_step.py --config training/onestep-config.json

