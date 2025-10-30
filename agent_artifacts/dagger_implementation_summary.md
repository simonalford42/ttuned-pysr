# DAgger Implementation - Final Version

## Overview

DAgger (Dataset Aggregation) creates training data where neural model rollout states are paired with expert (BasicSR) optimal actions. This implementation uses a clean in-place modification approach.

## How It Works

### Step 1: Generate Neural SR Traces
```bash
python generate_traces.py \
  --expressions_file datasets/expressions/arith_10_c05.pkl.gz \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000 \
  --max_expressions 10 --num_generations 10
```

This creates traces with neural SR populations in the `expressions` field.

### Step 2: Add Expert Labels
```bash
python dagger_relabel.py \
  --input datasets/traces/test/gen10_arith_10_c05_20251016_220146.pkl.gz
```

This:
- Loads the neural traces
- For each generation, computes what expert (BasicSR) would generate
- Adds `dagger_expressions` field to each generation
- Keeps original neural `expressions` intact
- Modifies the file **in-place**
- Marks metadata with `dagger: True`

### Step 3: Create Training Data
```bash
python one_step_conversion.py \
  --input datasets/traces/test/gen10_arith_10_c05_20251016_220146.pkl.gz \
  --dagger \
  --split
```

This creates training examples where:
- **Input population**: Neural SR's `expressions` (what neural model actually generated)
- **Target**: Expert's `dagger_expressions` (what expert would have generated)

## Key Design

### Data Structure

**Before DAgger relabeling:**
```python
generation = {
    'expressions': ['x0', '(x0 + x1)', ...],  # Neural's population
    'fitnesses': [...],
    ...
}
```

**After DAgger relabeling:**
```python
generation = {
    'expressions': ['x0', '(x0 + x1)', ...],     # Neural's population (unchanged)
    'dagger_expressions': ['x1', '(x0 * x1)', ...],  # Expert's population (added)
    'fitnesses': [...],
    ...
}
```

### Training Example Format

```json
{
    "context": "generation: 0 | 2 variables | ops: +,-,* | constants: 1.0",
    "population": "x0 <FITNESS>-1.0 (x0 + x1) <FITNESS>-0.5 ...",
    "target": "(x0 * x1)",  // Expert's choice
    "metadata": {...}
}
```

The training data pairs:
- **Input**: Neural model's rollout state (context + neural population)
- **Output**: Expert's optimal action at that state

## Benefits

1. **Correct Implementation**: Neural population as input, expert as target
2. **In-Place Modification**: No duplicate files, original neural data preserved
3. **Clean Separation**: DAgger is a pure relabeling step, not mixed with trace generation
4. **Backward Compatible**: Regular training data creation unchanged
5. **No Ambiguity**: Cannot combine `--dagger` with `--ancestors_only`

## Files Modified

### `dagger_relabel.py`
- `add_expert_labels_to_trajectory()`: Adds `dagger_expressions` to each generation
- `add_expert_labels_to_traces()`: Processes entire file in-place
- Modifies original file, no output file needed

### `one_step_conversion.py`
- Added `--dagger` flag
- When set, uses `dagger_expressions` as targets instead of `expressions`
- Population input always uses `expressions` (neural rollout)
- Prevents combining with `--ancestors_only`

## Test Results

✓ **Relabeling**: Added dagger_expressions to 10 trajectories in ~1s
✓ **Structure**: Each generation has both `expressions` and `dagger_expressions`
✓ **Training data**: Created 1800 examples (1620 train, 180 val)
✓ **Format**: Correct pairing of neural input with expert targets

## Usage Example

```bash
# Full workflow
python generate_traces.py \
  --expressions_file datasets/expressions/arith_1k_c05.pkl.gz \
  --checkpoint training/checkpoints/tiny_221861/checkpoint-210000 \
  --num_generations 50

python dagger_relabel.py \
  --input datasets/traces/gen50_arith_1k_c05.pkl.gz

python one_step_conversion.py \
  --input datasets/traces/gen50_arith_1k_c05.pkl.gz \
  --dagger \
  --split

# This creates:
# - datasets/training/gen50_arith_1k_c05_dagger_train.jsonl
# - datasets/training/gen50_arith_1k_c05_dagger_val.jsonl
```

## Why This Design

The initial implementation created separate files and replaced entire trajectories. This was wrong because:
1. Lost the neural rollout states (needed as input)
2. Created duplicate data storage
3. Mixed concerns (generation + relabeling)

The correct design:
1. Keep neural populations as input (what model actually saw)
2. Add expert actions as targets (what model should have done)
3. In-place modification is efficient and clear
4. Training conversion has explicit `--dagger` flag for clarity
