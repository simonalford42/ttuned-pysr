## Set Up SLURM Jobs for Trace Generation (10/3/2025)

Added automated trace generation pipeline for SLURM:

### New files:
1. **`run_tracegen.sh`**: SLURM script for trace generation jobs
   - 32GB memory, 12-hour time limit
   - Suitable for long-running trace generation

2. **`generate_traces_batch.py`**: CLI wrapper for batch trace generation
   - Command-line interface for all trace generation parameters
   - Works with SLURM job submission

### Updated `submit_jobs.sh`:
Added trace generation jobs for all 5 configs with 1k and 10k generations:
- **1k generation traces**: ~2-4 hours each, suitable for initial experiments
- **10k generation traces**: ~20-40 hours each, for deeper learning

Job naming convention: `tg1k_asm05` (trace gen, 1k generations, arith+simple, complexity 0.5)

Output files will be: `gen1k_train_1k_arith_c05.pkl.gz`, etc.

All trace jobs are commented out by default - uncomment to submit.

## Optimized Trace Dataset Storage (10/3/2025)

Updated `generate_traces.py` to use same storage optimizations as `generate_expressions.py`:

### Changes:
1. **Switched to gzipped pickle format** (`.pkl.gz` instead of `.json`):
   - ~9x compression vs JSON
   - Faster loading/saving
   - Native numpy array support

2. **Use float32 for fitness arrays**:
   - 50% smaller than float64
   - Sufficient precision for fitness values

3. **Keep fitnesses as numpy arrays**:
   - More efficient than Python lists
   - Easier analysis and manipulation

4. **Improved filename format**: `gen{N}_{expression_filename}.pkl.gz`
   - Example: `gen10_train_100_arith_c05.pkl.gz` (10 generations)
   - Example: `gen1k_train_100_arith_c05.pkl.gz` (1000 generations)
   - Makes it easy to identify trace generation parameters
   - Preserves connection to source expression dataset

5. **Updated inspect_dataset.py**:
   - Now supports both expression and trace datasets
   - Auto-detects dataset type
   - Shows relevant stats for each type

6. **Updated training/convert_trajectories.py**:
   - Now supports both `.pkl.gz` and `.json` trace files
   - Handles numpy arrays in fitnesses
   - Works with new flattened trajectory format

### Storage improvements:
- Trace datasets now significantly smaller
- Faster to load and process
- Consistent format across expression and trace datasets
- Full backward compatibility with old JSON format

## Fixed Constants Generation in generate_expressions.py (10/3/2025)

Fixed bug where constants were not being generated in expressions despite being specified:

### Problem:
1. **Type mismatch**: `extra_constants` was passed as a list instead of comma-separated string
2. **Missing probability**: `prob_const` was not set, defaulting to 0.0 (no constants generated)

### Solution:
- Convert constants list to comma-separated string: `",".join(str(c) for c in constants)`
- Set `prob_const = 0.2` when constants are specified (20% probability of generating constant in leaf positions)

### Result:
Now `--constants="1.0,2.5,3.14"` correctly generates expressions like:
- `((x_1 mul x_4) mul 3.14)`
- `(x_2 sub (2.5 sub x_2))`

## Cleaned Up Training Data Format (9/15/2025)

Removed unnecessary fields from training data to reduce file size and focus on essential information:

### Changes Made:
1. **Removed `fitnesses` array**: The individual fitness values for each expression in a generation are no longer stored in training data
2. **Removed `avg_fitness`**: Average fitness per generation is no longer computed or stored
3. **Kept essential fields**: Retained `best_fitness`, `best_expression`, `expressions`, `population_diversity` for analysis
4. **Preserved analysis capability**: BasicSR still computes fitnesses internally for analysis but filters them out when creating training datasets

### Implementation:
- Modified `generate_traces.py` to filter out `fitnesses` and `avg_fitness` when creating training datasets
- Updated `convert_trajectories.py` to handle missing fitnesses gracefully with dummy values
- Kept full fitness computation in `basic_sr.py` for runtime analysis and debugging

This reduces training file sizes significantly while preserving all information needed for neural model training.

## Updated Training Data Generation API (9/15/2025)

Updated the training data generation system to use simplified JSON formats as requested:

### Changes Made:
1. **Expressions format**: Changed from full objects with metadata to simple array of expression strings
   - Before: `{"expressions": [{"id": 0, "expression": "x+1", "variables": ["x0"], ...}, ...]}`
   - After: `{"expressions": ["x+1", "x*2", "x**2"]}`

2. **Traces format**: Changed from nested object structure to flat array
   - Before: `{"trajectories": {"expr_0": [trajectory_data], "expr_1": [...]}}`
   - After: `{"trajectories": [trajectory_data1, trajectory_data2, ...]}`

3. **Updated processing pipeline**: Modified `convert_trajectories.py` to handle the new flat array formats, including support for raw arrays extracted by dataset_manager.

4. **Verified end-to-end**: Successfully tested complete pipeline generating expressions → traces → training dataset with new formats.

### Files Updated:
- `generate_expressions.py`: Simplified expression output format
- `generate_traces.py`: Updated traces output to flat array, modified expression loading
- `training/convert_trajectories.py`: Added support for new flat array format detection and processing
- `dataset_manager.py`: Works correctly with new formats

## Implemented Training Data Generation System (9/12/2025)

Successfully implemented a complete training data generation system with three main components:

### 1. Expression Generator (`generate_expressions.py`)
- Generates synthetic polynomial expressions for training
- Configurable parameters: max degree, max variables, constants
- Supports reproducible generation with seed control
- Saves expressions with metadata and data generation functions

### 2. Trace Generator (`generate_traces.py`) 
- Runs BasicSR on expression datasets to collect evolutionary trajectories
- Integrates with existing `collect_trajectories.py` infrastructure
- Configurable BasicSR parameters (population size, generations, time limits)
- Saves complete trajectory data with metadata

### 3. Dataset Manager (`dataset_manager.py`)
- Manages end-to-end pipeline from expressions → traces → training data
- Converts traces to one-step prediction format using existing `convert_trajectories.py`
- Creates train/validation splits with configurable ratios
- Handles dataset versioning, validation, and metadata tracking

### Test Results
Created and validated test dataset `test_small`:
- 5 expressions, 10 generations each
- Generated 20 training examples (16 train, 4 val)
- Traces correctly capture population evolution between generations
- Training data format compatible with existing model training pipeline
- All data validation checks passed

The system successfully builds on existing infrastructure while providing proper dataset management and reproducibility.

## Enhanced Context System for Neural SR (9/12/2025)

Implemented rich context functionality to provide neural models with more informative data about the symbolic regression problems:

### Context Types
1. **Basic Context** - Traditional format: `x0,x1 | +,-,*,/ | 1.0,2.0` (22 chars)
2. **Rich Context** - Adds data statistics: basic + `STATS: range=[...] | mean=... | var=... | skew=... | monotonic=... | zeros=... | complexity=...` (~134 chars)
3. **SuperRich Context** - Rich + ASCII text plot: rich + `PLOT: [ASCII visualization]` (~568 chars)

### Rich Context Features
- **Data moments**: mean, variance, skewness of target values
- **Range information**: min/max values for inputs and outputs
- **Function characteristics**: monotonicity detection, zero crossings count
- **Complexity measure**: normalized total variation metric
- **ASCII plots**: Text-based visualization for 1D functions (superrich only)

### Implementation
- Enhanced `format_utils.py` with `compute_data_statistics()` and contextual formatting
- Updated `convert_trajectories.py` to support `--context_type` argument
- Added data context loading for problems requiring rich statistics
- Maintained backward compatibility with existing basic context

### Test Results
- Successfully converted trajectories with all three context types
- Context length scales appropriately: 22 → 134 → 568 characters
- ASCII plots render correctly for constant/linear/nonlinear functions
- Rich context provides meaningful statistical summaries of data

This enhancement allows neural models to incorporate problem-specific information during search, potentially leading to more intelligent expression suggestions.

## Refactored MinimalSR to BasicSR

Successfully refactored the MinimalSR class with the following changes:

1. **Renamed class**: Changed from `MinimalSR` to `BasicSR` in `minimal_sr.py`
2. **Added time limit functionality**:
   - Added `time_limit` parameter to `__init__` method (defaults to None for no limit)
   - Added time tracking in `fit` method that stops evolution when time limit is reached
3. **Added MSE early stopping**:
   - Added check for MSE <= 1e-31 to stop evolution when near-zero error is achieved
   - Prints stopping message when early termination occurs
4. **Updated all references**: Updated imports and class references in:
   - `time_comparison_simple.py`
   - `initial_vs_evolved.py`
   - `time_comparison.py`
   - `collect_trajectories.py`

**Testing results**:
- BasicSR correctly stops when MSE reaches zero on simple problems (y=x)
- BasicSR correctly respects time limits and finds good solutions within 15 seconds for harder problems (y=x²+1)
- Found expression `(1.0 * (1.0 + (x0 * x0)))` which is equivalent to x²+1 with MSE=0.000000

All functionality works as requested.

## File Reorganization and Updated Analysis

Successfully reorganized the codebase and ran updated analysis:

1. **File cleanup**:
   - Deleted old `time_comparison.py`
   - Renamed `time_comparison_simple.py` to `time_comparison.py`
   - Renamed `minimal_sr.py` to `basic_sr.py`
   - Updated all import statements from `minimal_sr` to `basic_sr`
   - Deleted `minimal_sr_results.md`

2. **Updated time_comparison.py**:
   - Now uses BasicSR.fit() directly with time_limit parameter
   - Leverages built-in MSE early stopping (≤ 3e-16)
   - Population size set to 30 for good balance of speed and exploration
   - Updated to run each problem for 60 seconds (vs. previous 15 seconds)

3. **Analysis results (60 seconds per problem)**:
   - **pythagorean_3d**: Solved perfectly in 0.3s (MSE ≈ 7.45e-31)
   - **quadratic_formula_discriminant**: Solved perfectly in 1.1s (MSE ≈ 3.21e-31)
   - **polynomial_product**: Solved perfectly in 0.0s (MSE ≈ 1.49e-31)
   - **compound_fraction**: Could not solve (MSE = 3.11e-02 after 8.9s)
   - **surface_area_sphere_approx**: Partial solution (MSE = 4.65e-01 after 19.5s)

4. **Key findings**:
   - Early stopping works effectively - most problems solved in <2 seconds
   - BasicSR successfully finds exact solutions for simpler mathematical relationships
   - More complex relationships (compound fractions, surface area approximations) remain challenging
   - The updated implementation properly uses both time limits and MSE thresholds

## Trajectory Collection System Implementation

Successfully implemented comprehensive trajectory collection system for BasicSR:

1. **Fixed collect_trajectories.py**:
   - Rewrote to use BasicSR.fit() instead of manual evolution loop
   - Now leverages built-in record_population_state() method for consistent data format
   - Generates rich trajectory data with full population information per generation
   - Replaced compound_fraction with mixed_polynomial to avoid local optima issues

2. **Enhanced BasicSR trajectory collection**:
   - Added collect_trajectory parameter to enable/disable trajectory recording
   - record_population_state() captures complete population state including:
     - All expressions and their fitnesses
     - Population diversity metrics
     - Best individual per generation
     - Average fitness statistics

3. **Configured for harder problems**:
   - Set up collection for 5 harder problems: pythagorean_3d, quadratic_formula_discriminant, polynomial_product, mixed_polynomial, surface_area_sphere_approx
   - Parameters: 60 seconds per problem, 2 runs per problem
   - Population size 20 for good trajectory granularity
   - Generates comprehensive metadata and summary statistics

4. **Data format standardization**:
   - Trajectory data now matches test_single_trajectory format
   - Contains complete population evolution history
   - Includes problem metadata, timing information, and final results
   - Saves both combined and individual problem trajectory files

5. **Verification results**:
   - Simple problems (y=x) solve in generation 0 due to MSE early stopping (correct behavior)
   - Complex problems show proper multi-generation evolution
   - Trajectory collection system generates rich training data for future ML models

The system is now ready to collect comprehensive evolutionary trajectory data for symbolic regression research and training purposes.

## 8/11 - Neural SR Framework Implementation
- Modified train_one_step.py to enable wandb via config file instead of command line flag
- Refactored BasicSR class to abstract population generation into generate_new_population() method
- Created NeuralSR class that uses trained transformer models for neural population generation
- Implemented robust fallback mechanisms when neural generation fails, allowing seamless comparison between BasicSR and NeuralSR

## 8/12 - Parallel Population Generation for NeuralSR
- Rewrote NeuralSR.generate_new_population() to generate all population members in parallel instead of sequentially
- Changed from generating one expression at a time in a loop to sampling N outputs simultaneously using num_return_sequences parameter
- Processing now happens in batch: single model.generate() call creates all needed expressions, then processes each generated output
- This significantly improves efficiency by reducing the number of model inference calls from population_size-1 to just 1 per generation

## 8/12 - Code Refactoring: Reusable Basic Evolution Method
- Created generate_child_via_evolution() method in BasicSR to eliminate code duplication
- Extracted the common pattern of tournament selection + crossover/mutation into a single reusable function
- Updated BasicSR.generate_new_population() to use the new method, making it more concise
- Updated all 3 fallback locations in NeuralSR to use the same method, ensuring consistent behavior
- The method accepts a configurable crossover_prob parameter (default 0.7) for flexibility

## 8/12 - NeuralSR Size Constraint Removal
- Updated NeuralSR well-formed checking logic to accept any successfully parsed neural-generated expression (no size constraints)
- NeuralSR still accepts max_size and max_depth parameters which apply to fallback evolution operations (crossover/mutation)
- This allows the neural model to generate expressions of any size while maintaining user control over fallback operation constraints

## 8/12 - Neural Comparison Parameter Alignment
- Updated neural_comparison.py to use the same max_size and max_depth parameters as collect_trajectories.py
- Changed from max_depth=3, max_size=10 to max_depth=4, max_size=15
- This ensures fair comparison between neural and basic SR using the same constraints that were used during trajectory collection

## 8/12 - Training Configuration Setup for Larger Model
- Renamed onestep-config.json to onestep-tiny.json for the existing tiny model configuration
- Created new onestep-s.json configuration for training a larger model using gpt-neo-s.json
- Key differences in onestep-s.json:
  - Uses gpt-neo-s.json model config (1024 hidden size, 16 layers vs 256 hidden size, 4 layers)
  - Reduced batch size from 4 to 2 per device to accommodate larger model memory requirements
  - Added gradient_accumulation_steps=2 to maintain effective batch size
  - Reduced learning rate from 5e-4 to 3e-4 for more stable training of larger model
  - Updated output directory to "training/checkpoints/onestep-s"

## 8/12 - Research Documentation Created
- Created comprehensive research-style markdown document (symbolic_regression_approach.md) explaining the symbolic regression approach
- Document includes formal problem definition with mathematical notation
- Details BasicSR algorithm with population initialization, evolution operations, and fitness function
- Explains NeuralSR algorithm including neural architecture, training data format, and neural population generation
- Provides experimental validation framework and key contributions of the hybrid approach

## 9/15 - Enhanced Context with Generation Numbers
- Modified format_context() in format_utils.py to include generation parameter as first argument
- Updated function documentation to describe the new generation parameter
- Updated all call sites across the codebase to pass generation numbers:
  - basic_sr.py: Updated format_context_and_population() and generate_new_population() methods to accept and pass generation
  - superrich_demo.py and test_context_system.py: Updated calls to use generation 0 for demo/test purposes
  - convert_trajectories.py: Updated to use loop index (generation number) when creating context
  - fine_grained_comparison.py: Updated to pass correct generation numbers
- This allows the neural model to understand which generation it's working on during training and inference

## 2025-10-03: Constants Support and Metadata Optimization

### Changes Made

**1. Added constants parameter to BasicSR** (`basic_sr.py`)
- Added `constants` parameter to `__init__` (default: `[1.0, 2.0]`)
- Modified `create_terminal()` to handle empty constants list (only creates variables when no constants available)

**2. Enhanced generate_expressions.py with constants control**
- Added `use_constants` parameter to `generate_e2e_expressions()` and `generate_training_expressions()`
- Added `--no_constants` CLI flag to disable constant generation
- When `use_constants=False`, sets `prob_const=0.0` in E2EDataGenerator
- Stores `use_constants` flag in expression metadata

**3. Updated generate_traces.py for automatic constant detection**
- Added `constants` parameter (default: `[1.0, 2.0]`)
- Added `--constants` CLI flag for custom constant list
- Reads `use_constants` from expression metadata and automatically sets BasicSR constants:
  - If expressions were generated without constants (`use_constants=False`), sets `constants=[]` for BasicSR
  - If expressions have constants, uses provided/default constants
- Stores operator configuration and constants in trace metadata:
  - `binary_operators`, `unary_operators`, `constants` at top level
  - Enables downstream tools to read configuration from metadata

**4. Optimized convert_trajectories.py to read from metadata**
- Now reads `binary_operators` and `constants` from trace file metadata instead of parsing every expression
- Only extracts variables per-trajectory (since they vary by expression)
- Falls back to parsing if metadata not available (backward compatibility)
- Added `extract_variables_from_trajectory()` helper function in `format_utils.py`

### Pipeline Integration

The full pipeline now automatically handles constants:

```bash
# Generate expressions WITHOUT constants
python generate_expressions.py --n_expressions 100 --no_constants

# Generate traces (automatically detects no constants from metadata)
python generate_traces.py --expressions_file datasets/expressions/train_*.pkl.gz --operator_set full

# Convert to training format (reads ops/constants from metadata)
python training/convert_trajectories.py --input datasets/traces/traces_*.json --output data/training.jsonl
```

The constants configuration flows through the entire pipeline via metadata:
1. `generate_expressions.py` → stores `use_constants` in expression metadata
2. `generate_traces.py` → reads from expression metadata, passes to BasicSR, stores in trace metadata  
3. `convert_trajectories.py` → reads from trace metadata for efficient context formatting

### Benefits
- **Consistency**: BasicSR search space automatically matches the generated expressions
- **Efficiency**: Avoids parsing every expression to extract operators/constants
- **Flexibility**: Support both constant-free and constant-enabled workflows
- **Backward compatible**: Falls back to parsing if metadata unavailable


## 2025-10-03b: Changed Default Constants to [1.0]

### Changes Made

**Changed default constants from [1.0, 2.0] to [1.0] across the entire pipeline:**

1. **basic_sr.py**: Default `constants=[1.0]`
2. **generate_expressions.py**: 
   - Changed parameter from `use_constants: bool` to `constants: List[float] = [1.0]`
   - Changed CLI from `--no_constants` flag to `--constants` arg (default: "1.0")
   - Empty string for no constants
3. **generate_traces.py**: 
   - Default `constants=[1.0]`  
   - CLI `--constants` default changed to "1.0"
   - Automatically reads constants from expression metadata and uses those if default not overridden
4. **submit_jobs.sh**: All jobs now pass `--constants=1.0`

### Rationale
- Simplifies search space by using only one constant instead of two
- Still allows flexibility via command line (can pass "1.0,2.0" or "" for no constants)
- Automatic metadata propagation ensures consistency across pipeline

### Usage Examples

```bash
# Generate with default (1.0)
python generate_expressions.py --n_expressions 100

# Generate with multiple constants
python generate_expressions.py --n_expressions 100 --constants "1.0,2.0,3.0"

# Generate with no constants
python generate_expressions.py --n_expressions 100 --constants ""

# Trace generation auto-detects constants from expression metadata
python generate_traces.py --expressions_file datasets/expressions/train_*.pkl.gz
```


## 2025-10-20: Created Autoregressive Training Pipeline

### Summary
Created two new files to enable autoregressive training for neural symbolic regression, complementing the existing one-step prediction approach:

1. **autoreg_conversion.py**: Converts BasicSR trajectory traces into autoregressive training format
2. **train_autoreg.py**: Training script for autoregressive population generation

### Key Differences from One-Step Approach

#### One-Step (Existing)
- **Data format**: Multiple examples per generation transition, one for each expression in next population
- **Training objective**: Given context + current population → predict single next expression
- **Sampling**: Can sample a fraction of expressions from each generation (via `sample_fraction`)
- **Use case**: Model predicts individual expressions, need to sample N times to get full population

#### Autoregressive (New)
- **Data format**: One example per generation transition
- **Training objective**: Given context + current population → predict entire next population autoregressively
- **Sampling**: Can sample a fraction of trajectories (via `sample_fraction`)
- **Use case**: Model generates entire population in one forward pass (more aligned with stream-of-search)

### Implementation Details

**autoreg_conversion.py**:
- Takes same input format as one_step_conversion.py (.pkl.gz trajectory files)
- Creates JSONL with format: `{"context": ..., "population": ..., "target": "expr1 <FITNESS>f1 expr2 <FITNESS>f2 ..."}`
- Target is entire next population formatted as single string to be generated autoregressively
- Supports all three context types: basic, rich, superrich
- `sample_fraction` parameter samples trajectories (not individual expressions)
- Includes train/val split functionality

**train_autoreg.py**:
- Based on train_one_step.py with same model architecture support
- Tokenization trains model to autoregressively complete entire population
- Labels: -100 for context/population input, actual tokens for target population
- Uses same special tokens: `<CONTEXT>`, `<POPULATION>`, `<FITNESS>`, `<TARGET>`
- Default config path: `training/configs/autoreg-s.json`

### Usage Example

```bash
# Convert traces to autoregressive format
python autoreg_conversion.py \
    --input datasets/traces/gen50_train_1k_arith_c05.pkl.gz \
    --context_type basic \
    --split

# Train autoregressive model
python train_autoreg.py --config training/configs/autoreg-s.json
```

### Next Steps
- Create config file at `training/configs/autoreg-s.json` for autoregressive training
- Test conversion and training pipeline
- Compare autoregressive vs one-step performance in neural SR


## 2025-10-21: Added Autoregressive Support to NeuralSR

### Summary
Extended the `NeuralSR` class in `basic_sr.py` to support both one-step and autoregressive models, enabling unified testing and comparison of both approaches.

### Changes Made

**1. Added `autoregressive` parameter to NeuralSR**
- New parameter: `autoregressive=False` (default)
- When `True`, uses autoregressive generation mode
- When `False`, uses existing one-step generation mode

**2. Refactored generation methods**
- Split `generate_new_population()` into dispatch method
- Created `generate_new_population_onestep()` for one-step models (existing logic)
- Created `generate_new_population_autoregressive()` for autoregressive models (new)

**3. Autoregressive generation implementation**
- Single forward pass generates entire population as space-separated expressions
- Longer context window: `max_new_tokens = max(200, 10 * population_size)`
- Parses generated text into individual expressions
- Falls back to evolutionary operators if insufficient well-formed expressions
- Tracks neural suggestion statistics same as one-step mode

**4. Created test script**
- `test_autoreg_neural.py`: Simple test script for autoregressive models
- Tests with checkpoint: `training/checkpoints/tiny_218153/checkpoint-50000`
- Validates autoregressive generation on simple problem (y=x)

### Key Differences

**One-Step Mode** (`autoregressive=False`):
- Generates `population_size-1` expressions in parallel
- Each expression generated independently
- Output format: individual expressions

**Autoregressive Mode** (`autoregressive=True`):
- Single generation pass for entire population
- Model outputs space-separated expressions
- Output format: "expr1 expr2 expr3 ..."
- Parses and validates each expression individually

### Usage Example

```python
# One-step model (existing)
model = NeuralSR(
    model_path="training/checkpoints/onestep-tiny/final_model",
    autoregressive=False,  # default
    population_size=20,
    num_generations=10
)

# Autoregressive model (new)
model = NeuralSR(
    model_path="training/checkpoints/tiny_218153/checkpoint-50000",
    autoregressive=True,  # enables autoreg mode
    population_size=20,
    num_generations=10
)

# Both use the same API
model.fit(X, y, verbose=True)
predictions = model.predict(X)
```

### Testing
- Successfully tested with `test_autoreg_neural.py`
- Model generates expressions like "x0", "(x0 + x0)", "((x0 * x0) + x0)"
- Parsing and fallback mechanisms work correctly
- Well-formed percentage tracking functional

### Benefits
- Unified interface for both model types in NeuralSR
- Easy to test and compare one-step vs autoregressive approaches
- Can use same evaluation scripts (`neural_comparison.py`, `fine_grained_comparison.py`)
- Clean abstraction separates generation logic from core SR algorithm

