#!/bin/bash

# 10/30
sbatch -J direct --partition ellis --gres=gpu:nvidia_rtx_a6000:1 run.sh accelerate launch --config_file training/configs/accelerate1.yaml train_one_step.py --config training/configs/direct.json
sbatch -J direct --partition ellis --gres=gpu:nvidia_rtx_a6000:1 run.sh accelerate launch --config_file training/configs/accelerate1.yaml train_one_step.py --config training/configs/direct2.json
# sbatch -J ttsr --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml train_one_step.py --config training/configs/onestep-tiny.json

# 10/27/25
# sbatch -J dagger --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml dagger.py --num_iterations 5 --checkpoint training/checkpoints/tiny_208822/final_model --original_dataset datasets/training/gen1k_arith_1k_c05_20251016_214231_10.jsonl --expressions_file datasets/expressions/arith_10k_c05_20251017_105805.pkl.gz --num_expressions 50 --num_generations 1000
# sbatch -J dagger --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml dagger.py --num_iterations 5 --checkpoint training/checkpoints/tiny_227520/final_model --original_dataset datasets/training/gen1k_arith_50_c05_20251023_215035.jsonl --expressions_file datasets/expressions/arith_50_c05_20251023_215035.pkl.gz --num_generations 1000 --num_expressions 10

# 10/16/25
# sbatch -J ttsr --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml train_one_step.py --config training/configs/onestep-tiny.json
# sbatch -J ttsr --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml train_one_step.py --config training/configs/onestep-tiny2.json
# sbatch -J 10k_conv --partition ellis run_cpu.sh python -u one_step_conversion.py --input datasets/traces/gen10k_arith_1k_c05_20251016_214231.pkl.gz --ancestors_only
# sbatch -J 10k --partition ellis run_cpu.sh python -u generate_traces.py --expressions_file datasets/expressions/arith_1k_c05_20251016_214231.pkl.gz  --operator_set arith --num_generations 10000
#

# 10/6/26
# sbatch -J tiny --partition ellis run.sh python -u train_one_step.py --config training/configs/onestep-tiny.json
# sbatch -J ttsr --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml train_one_step.py --config training/configs/onestep-tiny.json
# sbatch -J ttsr --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml train_one_step.py --config training/configs/onestep-tiny2.json

# Multi-GPU variants using Accelerate (override Slurm gres and processes)
# 4 GPUs on a single node
# PORT4=$((29500 + (RANDOM % 1000)))
# sbatch -J ttsr4 --partition ellis --gres=gpu:nvidia_rtx_a6000:4 run.sh \
#   accelerate launch --config_file training/configs/accelerate.yaml --num_processes 4 --num_machines 1 --main_process_port ${PORT4} \
#   train_one_step.py --config training/configs/onestep-tiny.json

# 8 GPUs on a single node
# PORT8=$((29500 + (RANDOM % 1000)))
# sbatch -J ttsr8 --partition ellis --gres=gpu:nvidia_rtx_a6000:8 run.sh \
#   accelerate launch --config_file training/configs/accelerate.yaml --num_processes 8 --num_machines 1 --main_process_port ${PORT8} \
#   train_one_step.py --config training/configs/onestep-tiny.json

# sbatch -J dg_asm05 --partition ellis run.sh python generate_expressions.py --n_expressions=1000 --binary_ops=add,sub,mul --complexity=0.5 --seed=42 --constants=1.0
# sbatch -J dg_asm10 --partition ellis run.sh python generate_expressions.py --n_expressions=1000 --binary_ops=add,sub,mul --complexity=1.0 --seed=43 --constants=1.0
# sbatch -J dg_all03 --partition ellis run.sh python generate_expressions.py --n_expressions=1000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.3 --seed=44 --constants=1.0
# sbatch -J dg_all06 --partition ellis run.sh python generate_expressions.py --n_expressions=1000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.6 --seed=45 --constants=1.0
# sbatch -J dg_all10 --partition ellis run.sh python generate_expressions.py --n_expressions=1000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=1.0 --seed=46 --constants=1.0

# submit trace-generation jobs for a list of expression files and generation counts
# NUM_GENERATIONS=(1000 10000) # add more values if needed, e.g. (1000 10000)
# NUM_GENERATIONS=(1000 10000)
# FILES=(
#     "arith_1k_c05_20251003_135753.pkl.gz"
#     "arith_1k_c10_20251003_135910.pkl.gz"
#     "full_1k_c03_20251003_135908.pkl.gz"
#     "full_1k_c06_20251003_135909.pkl.gz"
#     "full_1k_c10_20251003_135911.pkl.gz"
# )

# for num_generations in "${NUM_GENERATIONS[@]}"; do
#     for filename in "${FILES[@]}"; do
#         expr_path="datasets/expressions/${filename}"

#         if [[ ! -f "$expr_path" ]]; then
#             echo "Warning: expressions file not found: $expr_path" >&2
#             continue
#         fi

#         # make a safe job name from the filename (remove extension, replace non-alnum with _)
#         base="${filename%%.*}"
#         jobname="${base}_${num_generations}"
#     jobname="${jobname//[^a-zA-Z0-9_-]/_}"
#         # either 'full' or 'arith'; get the substring before the first underscore
#         operators="${base%%_*}"
#         sbatch -J "$jobname" run.sh python -u generate_traces.py --expressions_file="$expr_path" --num_generations="$num_generations" --create_one_step --operator_set="$operators"
#     done
# done

# 10/2 - 10k expression datasets (archived - using 1k instead)
# sbatch -J dg_asm05 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=10000 --binary_ops=add,sub,mul --unary_ops= --complexity=0.5 --seed=42 --constants=1.0
# sbatch -J dg_asm10 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=10000 --binary_ops=add,sub,mul --unary_ops= --complexity=1.0 --seed=43 --constants=1.0
# sbatch -J dg_all03 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=10000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.3 --seed=44 --constants=1.0
# sbatch -J dg_all06 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=10000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=0.6 --seed=45 --constants=1.0
# sbatch -J dg_all10 --partition ellis run_datagen.sh generate_large_dataset.py --n_expressions=10000 --binary_ops=add,sub,mul,div,pow --unary_ops=abs,sqrt,sin,cos,tan,inv --complexity=1.0 --seed=46 --constants=1.0

# 9/9 - Overfitting test with tiny pythagorean dataset
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/configs/onestep-tiny-overfit.json

# 8/26 - Training experiments
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/configs/onestep-tiny.json
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/onestep-tiny-debug.json
# sbatch -J ft --partition ellis run2.sh python -u training/train_one_step.py --config training/onestep-tiny.json --resume --ckpt training/checkpoints/onestep-full_20250811_145316/checkpoint-75000

# 8/20
# sbatch -J ttsr2 --partition ellis run2.sh accelerate launch --config_file training/accelerate.yaml training/train_one_step.py --config training/onestep-s.json
