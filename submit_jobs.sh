#!/bin/bash

# 8/26
sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/onestep-tiny.json
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/onestep-tiny.json --resume --ckpt training/checkpoints/onestep-full_20250811_145316/checkpoint-75000

# 8/20
# sbatch -J ttsr2 --partition ellis run2.sh accelerate launch --config_file training/accelerate.yaml training/train_one_step.py --config training/onestep-s.json

# 8/13
# sbatch -J ttsr2 --partition ellis run2.sh
# sbatch -J ttsr2 --partition ellis run2.sh training/train_one_step.py --config training/onestep-s.json

# 8/12/25
# sbatch -J ttsr --partition ellis run.sh training/train_one_step.py --config training/onestep-s.json

# 8/11/25
# sbatch -J ttsr --partition ellis run.sh training/train_one_step.py --config training/onestep-config.json

