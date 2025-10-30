# Batched Neural SR and DAgger Implementation

## Summary

Implemented batched neural symbolic regression and DAgger-style training data generation. This allows:
1. Processing multiple expressions simultaneously with neural SR for better efficiency
2. Generating DAgger-style training data where the neural model does the rollout but expert labels are used

## New Files Created

### 1. `batched_neural_sr.py`
- **`BatchedNeuralSR`**: Main class for batched neural SR
  - Processes multiple expressions in parallel
  - Shares the neural model across batch members for efficiency
  - Returns final models, MSEs, and trajectories for each expression

- **`batched_fit()`**: Convenience function for running batched neural SR
  - Takes lists of X and y data
  - Returns final models, MSEs, and trajectories

**Key features:**
- Lazy initialization of neural model (loaded once, shared across batch)
- Separate populations and evolution for each expression
- Tracks trajectories for all expressions

### 2. `test_batched_sr.py`
Test script demonstrating batched neural SR functionality.

**Usage:**
```bash
python test_batched_sr.py --checkpoint training/checkpoints/tiny_221861/checkpoint-210000 \
  --num_problems 3 --population_size 20 --num_generations 50
```

**Options:**
- `--checkpoint`: Path to neural model checkpoint
- `--num_problems`: Number of test problems to generate (default: 3)
- `--population_size`: Population size (default: 20)
- `--num_generations`: Number of generations (default: 50)
- `--operator_set`: arith or full (default: arith)
- `--autoregressive`: Use autoregressive model

## Modified Files

### `generate_traces.py`

Added support for neural SR (both regular and batched):

**New arguments:**
- `--checkpoint`: Path to neural model checkpoint
- `--batch_size`: Number of expressions to process in parallel

**New helper functions:**

1. **`_fit_and_extract_trajectory()`**
   - Shared function to fit model and extract trajectory data
   - Used by both BasicSR and NeuralSR paths

2. **`_process_batched_neural()`**
   - Processes expressions in batches using `BatchedNeuralSR`
   - Divides expressions into batches and processes each batch in parallel

### `dagger_relabel.py` (New)

Separate script for creating DAgger-style training data by relabeling neural SR traces with expert actions.

**Key functions:**

1. **`relabel_trajectory_with_expert()`**
   - Takes a neural SR trajectory
   - Replays the population states
   - Computes what expert (BasicSR) would do at each generation
   - Returns relabeled trajectory with expert actions

2. **`relabel_traces_with_expert()`**
   - Main function that processes an entire traces file
   - Loads neural SR traces
   - Relabels each trajectory with expert actions
   - Saves as DAgger traces marked with `dagger: True`

**How DAgger workflow works:**
1. Generate traces using neural SR (normal `generate_traces.py`)
2. Run `dagger_relabel.py` to relabel with expert actions
3. Result: Neural rollout states paired with expert optimal actions

## Testing Results

### Test 1: Normal Neural SR Traces
```bash
python generate_traces.py \
  --expressions_file datasets/expressions/arith_10_c05_20251016_220146.pkl.gz \
  --output_dir datasets/traces/test \
  --max_expressions 10 \
  --population_size 20 \
  --num_generations 10 \
  --operator_set arith \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000
```

**Results:**
- ✓ Successfully generated 10 traces
- Time: 20.2s (avg 2.0s per expression)
- Output: `datasets/traces/test/gen10_arith_10_c05_20251016_220146.pkl.gz` (0.03 MB)
- 4 expressions solved perfectly (MSE=0)
- Other expressions had reasonable MSE values

### Test 2: DAgger-Style Traces (Two-Step Workflow)

Step 1: Generate neural SR traces
```bash
python generate_traces.py \
  --expressions_file datasets/expressions/arith_10_c05_20251016_220146.pkl.gz \
  --output_dir datasets/traces/test \
  --max_expressions 10 \
  --population_size 20 \
  --num_generations 10 \
  --operator_set arith \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000
```

Step 2: Relabel with expert actions
```bash
python dagger_relabel.py \
  --input datasets/traces/test/gen10_arith_10_c05_20251016_220146.pkl.gz \
  --output datasets/traces/test/gen10_arith_10_c05_20251016_220146_dagger.pkl.gz
```

**Results:**
- ✓ Successfully generated and relabeled 10 traces
- Neural generation time: 18.5s (avg 1.85s per expression)
- Relabeling time: ~1s total (very fast)
- Output marked with `dagger: True` and `original_source: "neural_sr"`
- Expert provides optimal actions for states visited by neural rollout

## Usage Examples

### Generate traces with neural SR (single at a time):
```bash
python generate_traces.py \
  --expressions_file datasets/expressions/arith_1k_c05.pkl.gz \
  --output_dir datasets/traces \
  --population_size 20 \
  --num_generations 50 \
  --operator_set arith \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000
```

### Generate traces with batched neural SR:
```bash
python generate_traces.py \
  --expressions_file datasets/expressions/arith_1k_c05.pkl.gz \
  --output_dir datasets/traces \
  --population_size 20 \
  --num_generations 50 \
  --operator_set arith \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000 \
  --batch_size 10
```

### Generate DAgger-style traces (two-step workflow):
```bash
# Step 1: Generate neural SR traces
python generate_traces.py \
  --expressions_file datasets/expressions/arith_1k_c05.pkl.gz \
  --output_dir datasets/traces \
  --population_size 20 \
  --num_generations 50 \
  --operator_set arith \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000

# Step 2: Relabel with expert actions
python dagger_relabel.py \
  --input datasets/traces/gen50_arith_1k_c05.pkl.gz \
  --output datasets/traces/gen50_arith_1k_c05_dagger.pkl.gz
```

### Test batched SR directly:
```bash
python test_batched_sr.py \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000 \
  --num_problems 5 \
  --population_size 20 \
  --num_generations 100
```

## Key Design Decisions

1. **Model Sharing**: In batched mode, the neural model is loaded once and shared across all expressions in the batch. This saves memory and initialization time.

2. **DAgger Two-Step Workflow**: DAgger is implemented as a separate relabeling step rather than integrated into generate_traces.py. This provides:
   - Cleaner separation of concerns (generation vs. relabeling)
   - Ability to relabel existing neural traces without regenerating
   - Simpler code maintenance
   - Flexibility to experiment with different relabeling strategies

3. **DAgger Format**: DAgger traces use the same format as regular traces but are marked with `dagger: True` and `original_source: "neural_sr"`. This allows them to be processed by the same downstream code.

4. **Ancestry Tracking**: Both neural and expert modes properly track ancestry information by calling `update_ancestry_info()` after evolution completes.

## Future Improvements

1. Add progress bars for long batch processing
2. Add option to save intermediate results during batch processing
3. Optimize memory usage for very large batches
4. Experiment with different DAgger variants (e.g., mixing neural and expert actions)
