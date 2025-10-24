# Using Autoregressive Models with NeuralSR

## Overview
The NeuralSR class now supports both one-step and autoregressive models via the `autoregressive` parameter.

## Quick Start

### Running neural_comparison.py with autoregressive model

```bash
# One-step model (default)
python neural_comparison.py \
    --checkpoint training/checkpoints/onestep-tiny/final_model \
    --problem 0

# Autoregressive model
python neural_comparison.py \
    --checkpoint training/checkpoints/tiny_218153/checkpoint-50000 \
    --autoregressive \
    --problem 0
```

### Running fine_grained_comparison.py with autoregressive model

```bash
# One-step model (default)
python fine_grained_comparison.py \
    --checkpoint training/checkpoints/onestep-tiny/final_model \
    --problem 0 \
    --max-generations 20

# Autoregressive model
python fine_grained_comparison.py \
    --checkpoint training/checkpoints/tiny_218153/checkpoint-50000 \
    --autoregressive \
    --problem 0 \
    --max-generations 20
```

### Direct usage in Python

```python
from basic_sr import NeuralSR
import numpy as np

# Generate some test data
X = np.random.randn(50, 1)
y = X[:, 0]  # Simple y = x problem

# One-step model
onestep_model = NeuralSR(
    model_path="training/checkpoints/onestep-tiny/final_model",
    autoregressive=False,  # default
    population_size=20,
    num_generations=10
)
onestep_model.fit(X, y, verbose=True)

# Autoregressive model
autoreg_model = NeuralSR(
    model_path="training/checkpoints/tiny_218153/checkpoint-50000",
    autoregressive=True,
    population_size=20,
    num_generations=10
)
autoreg_model.fit(X, y, verbose=True)
```

## Command-line Arguments

### neural_comparison.py
- `--checkpoint`: Path to model checkpoint (required)
- `--autoregressive`: Use autoregressive model (flag, default: False)
- `--problem`: Problem index to test (default: 0)
- `--max-generations`: Maximum generations (default: 1000)
- `--operator_set`: "arith" or "full" (default: arith)
- `--collect-trajectories`: Collect trajectory data
- `--save-trajectories`: Save trajectories to files

### fine_grained_comparison.py
- `--checkpoint`: Path to model checkpoint (required)
- `--autoregressive`: Use autoregressive model (flag, default: False)
- `--problem`: Problem index to test (default: 0)
- `--max-generations`: Maximum generations (default: 1000)
- `--operator_set`: "arith" or "full" (default: arith)

## Model Differences

### One-Step Models
- **Training**: Each training example = context + population → single expression
- **Inference**: Generates population_size-1 expressions in parallel
- **Output**: Individual expressions generated independently
- **Checkpoint examples**:
  - `training/checkpoints/onestep-tiny/final_model`
  - `training/checkpoints/onestep-full_*/final_model`

### Autoregressive Models
- **Training**: Each training example = context + population → entire next population
- **Inference**: Single generation pass for entire population
- **Output**: Space-separated expressions ("x0 (x0+x0) (x0*x0) ...")
- **Checkpoint examples**:
  - `training/checkpoints/tiny_218153/checkpoint-50000`
  - Any checkpoint from `train_autoreg.py`

## Expected Output

### One-Step Model
```
PREDICTED TARGETS: x0 (x0+1.0) (x0*x0) x0 (x0-1.0) ...
```
Each expression generated separately.

### Autoregressive Model
```
AUTOREG PREDICTED: x0 (x0 + x0) ((x0 * x0) + x0) x0 1.0 ...
Parsed 19/25 expressions, filled 0 with evolution
```
Entire population generated as one sequence, then parsed.

## Statistics Tracking

Both modes track the same statistics:
- `neural_suggestions_total`: Total expressions attempted
- `neural_suggestions_well_formed`: Successfully parsed expressions
- `get_well_formed_percentage()`: Success rate

## Troubleshooting

### Issue: "Model generates nonsense"
- **Solution**: Check that checkpoint matches the mode (one-step vs autoregressive)
- One-step checkpoints trained with `train_one_step.py`
- Autoregressive checkpoints trained with `train_autoreg.py`

### Issue: "All expressions filled with evolution"
- **Solution**: Model may not be well-trained, or temperature too high
- Check training loss in wandb
- Try different checkpoint (earlier/later in training)

### Issue: "Population size mismatch"
- **Solution**: For autoregressive, if model generates too few expressions, remaining slots filled with evolution
- This is expected behavior and allows graceful degradation
