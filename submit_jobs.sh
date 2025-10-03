#!/bin/bash

# 10/2 - Generate 1M expression datasets for each config
# Estimated time: ~40 minutes each, ~2.7 GB per dataset

# Config 1: add,sub,mul with complexity 0.5
sbatch -J dg_asm05 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=1000000 --binary_ops=add,sub,mul --unary_ops= --complexity=0.5 --seed=42

# Config 2: add,sub,mul with complexity 1.0
sbatch -J dg_asm10 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=1000000 --binary_ops=add,sub,mul --unary_ops= --complexity=1.0 --seed=43

# Config 3: all operators with complexity 0.3
sbatch -J dg_all03 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=1000000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.3 --seed=44

# Config 4: all operators with complexity 0.6
sbatch -J dg_all06 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=1000000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.6 --seed=45

# Config 5: all operators with complexity 1.0
sbatch -J dg_all10 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=1000000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=1.0 --seed=46

# Config 1: add,sub,mul with complexity 0.5
sbatch -J dg_asm05 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=100000 --binary_ops=add,sub,mul --unary_ops= --complexity=0.5 --seed=42

# Config 2: add,sub,mul with complexity 1.0
sbatch -J dg_asm10 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=100000 --binary_ops=add,sub,mul --unary_ops= --complexity=1.0 --seed=43

# Config 3: all operators with complexity 0.3
sbatch -J dg_all03 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=100000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.3 --seed=44

# Config 4: all operators with complexity 0.6
sbatch -J dg_all06 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=100000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.6 --seed=45

# Config 5: all operators with complexity 1.0
sbatch -J dg_all10 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=100000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=1.0 --seed=46

# 9/9 - Overfitting test with tiny pythagorean dataset
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/configs/onestep-tiny-overfit.json

# 8/26
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/configs/onestep-tiny.json
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/onestep-tiny-debug.json
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

