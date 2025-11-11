#!/bin/bash

# sbatch -J ttsr --partition ellis --time=72:00:00 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/onestep-tiny.json
# sbatch -J dagger --partition ellis --time=72:00:00 run.sh accelerate launch --config_file configs/accelerate1.yaml dagger.py --original_dataset datasets/training/gen100_20251106_123123_arith_1k_c05_20251016_214231_ancs.jsonl --num_expressions 40 --num_generations 50

# 11/10 – Third round: FeatureSetEmbedder experiments (features × pool)
# Base config: configs/direct-featurepool-small.json
# Feature sets:
#   RAW      = -o embedder_use_raw_xy=true  -o embedder_use_x_phases=false -o embedder_use_logx_phases=false
#   PHASES   = -o embedder_use_raw_xy=false -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true
#   COMBINED = -o embedder_use_raw_xy=true  -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true
# Pools: encoder | dsum | xattn (set via -o embedder_pool_type=...)

# 9 core runs (3 features × 3 pools)
# sbatch -J fe_raw_enc   run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=encoder  -o embedder_use_raw_xy=true  -o embedder_use_x_phases=false -o embedder_use_logx_phases=false
# sbatch -J fe_raw_dsum  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=dsum     -o embedder_use_raw_xy=true  -o embedder_use_x_phases=false -o embedder_use_logx_phases=false
sbatch -J fe_raw_xattn run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=xattn    -o embedder_use_raw_xy=true  -o embedder_use_x_phases=false -o embedder_use_logx_phases=false

# sbatch -J fe_phs_enc   run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=encoder  -o embedder_use_raw_xy=false -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true
sbatch -J fe_phs_dsum  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=dsum     -o embedder_use_raw_xy=false -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true
sbatch -J fe_phs_xattn run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=xattn    -o embedder_use_raw_xy=false -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true

sbatch -J fe_cmb_enc   run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=encoder  -o embedder_use_raw_xy=true  -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true
sbatch -J fe_cmb_dsum  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=dsum     -o embedder_use_raw_xy=true  -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true
sbatch -J fe_cmb_xattn run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=xattn    -o embedder_use_raw_xy=true  -o embedder_use_x_phases=true  -o embedder_use_logx_phases=true

# +1 sanity: add poly block to combined+encoder
sbatch -J fe_cmb_enc_poly run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-featurepool-small.json -o embedder_pool_type=encoder -o embedder_use_raw_xy=true -o embedder_use_x_phases=true -o embedder_use_logx_phases=true -o embedder_include_poly=true

# 11/9 – Direct SetPointEmbedder experiment suite (schedules, prefix, capacity, normalization)

# Best-so-far baseline (normalize=false, prefix=16). Scheduler comparisons + longer training
# sbatch -J ds4c run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o training_args.lr_scheduler_type=constant_with_warmup -o training_args.num_train_epochs=80
# sbatch -J ds4r run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o training_args.lr_scheduler_type=cosine_with_restarts -o training_args.num_train_epochs=80

# Seed stability (best config seeds)
# sbatch -J ds4s43 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o seed=43

# Prefix budget (no-normalize)
# sbatch -J ds4p8  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_prefix_len=8
# sbatch -J ds4p24 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_prefix_len=24

# Capacity sweeps (no-normalize)
# d_model
# sbatch -J ds4d64 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_d_model=64
# num_layers
# sbatch -J ds4l1  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_num_layers=1
# sbatch -J ds4l3  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_num_layers=3
# num_heads
# sbatch -J ds4h4  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_num_heads=4

# Fourier features revisit (no-normalize)
# sbatch -J ds4f2 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_fourier_features=2
# sbatch -J ds4f4 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false -o embedder_fourier_features=4

# Normalization ablations that preserve scale info
# Mean-only (demean)
# sbatch -J ds_norm_center run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=true -o embedder_norm_center_only=true -o embedder_norm_scale_only=false
# Scale-only (divide by std)
# sbatch -J ds_norm_scale  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=true -o embedder_norm_center_only=false -o embedder_norm_scale_only=true
# Normalize + append stats tokens (means/stds)
# sbatch -J ds_norm_stats  run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=true -o embedder_append_stats=true

# 11/3
# sbatch -J ttsr --partition ellis run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/onestep-tiny.json

### First round of embedder architecture experiments
# Direct prediction: E2E baseline,
# sbatch -J e2e run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-e2e-small.json
# default direct set (prefix_len=16, fourier_features=0, normalize=True)
# sbatch -J ds1 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json
# prefix length 8 or 24
# sbatch -J ds2 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_prefix_len=8
# sbatch -J ds3 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_prefix_len=24
# # no normalization
# sbatch -J ds4 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_normalize=false
# # fourier features 2 or 4
# sbatch -J ds5 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_fourier_features=2
# sbatch -J ds6 run.sh accelerate launch --config_file configs/accelerate1.yaml train.py --config configs/direct-set-small.json -o embedder_fourier_features=4


# sbatch -J ttsr --partition gpu run.sh accelerate launch --config_file configs/accelerate.yaml train.py --config configs/onestep-tiny2.json
# sbatch -J direct --partition gpu --gres=gpu:nvidia_rtx_a6000:1 run.sh accelerate launch --config_file training/configs/accelerate1.yaml train.py --config training/configs/direct.json

# 10/30
# sbatch -J direct --partition gpu --gres=gpu:nvidia_rtx_a6000:1 run.sh accelerate launch --config_file training/configs/accelerate1.yaml train.py --config training/configs/direct.json
# sbatch -J direct --partition gpu --gres=gpu:nvidia_rtx_a6000:1 run.sh accelerate launch --config_file training/configs/accelerate1.yaml train.py --config training/configs/direct2.json
# sbatch -J ttsr --partition ellis run.sh accelerate launch --config_file training/configs/accelerate.yaml train.py --config training/configs/onestep-tiny.json

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
