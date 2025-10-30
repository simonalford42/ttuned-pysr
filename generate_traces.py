"""
Generate BasicSR traces from expression datasets.
Builds on collect_trajectories.py and basic_sr.py.
"""
import json
import pickle
import gzip
import os
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sr import BasicSR, NeuralSR
import random
import sympy as sp
from utils import get_operators


def load_expressions_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load expressions from generated dataset (supports both .pkl.gz and .json)"""
    # Check file extension
    if filepath.endswith('.pkl.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filepath, 'r') as f:
            data = json.load(f)

    expressions = data['expressions']
    metadata = data['metadata']

    return expressions, metadata


def get_data_from_expr(expr_data: Dict[str, Any], seed: int = 42):
    """Get X, y data from expression data (uses pre-generated data if available)."""
    # If data is already in the expression dict (new format), use it directly
    if 'X_data' in expr_data and 'y_data' in expr_data:
        return expr_data['X_data'], expr_data['y_data']

    # Otherwise generate data (old format)
    expr_str = expr_data['expression']
    expr = sp.sympify(expr_str)
    symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
    if not symbols:
        symbols = [sp.Symbol('x0', real=True)]
    n_vars = len(symbols)

    f = sp.lambdify(symbols, expr, 'numpy')

    rstate = np.random.RandomState(seed)
    target_n = 50
    batch = 300
    max_tries = 8
    X_list = []
    y_list = []
    np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')
    for _ in range(max_tries):
        Xb = rstate.uniform(-3, 3, size=(batch, n_vars))
        try:
            yb = f(*[Xb[:, i] for i in range(n_vars)])
        except Exception:
            continue
        yb = np.asarray(yb)
        if yb.ndim > 1:
            yb = yb.squeeze()
        if yb.shape[0] != Xb.shape[0]:
            continue
        mask = np.isfinite(yb)
        if np.iscomplexobj(yb):
            mask &= (np.abs(np.imag(yb)) < 1e-12)
            yb = np.real(yb)
        X_list.append(Xb[mask])
        y_list.append(yb[mask])
        if sum(len(a) for a in y_list) >= target_n:
            break

    if not y_list:
        # Fallback simple quadratic
        X = rstate.uniform(-3, 3, size=(target_n, n_vars))
        y = X[:, 0] ** 2
    else:
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        if X.shape[0] > target_n:
            X = X[:target_n]
            y = y[:target_n]

    return X, y


def _fit_and_extract_trajectory(model, X, y, expr_str, expr_id, basicsr_params,
                                binary_operators, unary_operators, constants):
    """Fit model and extract trajectory data"""
    # Fit and collect trajectory
    start_time = time.time()
    model.fit(X, y)
    actual_time = time.time() - start_time

    # Get final results
    if model.best_model_:
        y_pred = model.predict(X)
        final_mse = float(np.mean((y - y_pred)**2))
        final_expression = str(model.best_model_)
    else:
        final_mse = float('inf')
        final_expression = None

    print(f"Completed: {len(model.trajectory)} generations, MSE={final_mse:.2e}, time={actual_time:.1f}s")

    # Store trajectory data with numpy arrays (use float32 for smaller size)
    filtered_trajectory = []
    for generation_data in model.trajectory:
        filtered_gen = {
            'generation': generation_data['generation'],
            'population_size': generation_data['population_size'],
            'expressions': generation_data['expressions'],  # Keep as list of strings
            'fitnesses': np.array(generation_data['fitnesses'], dtype=np.float32),  # Convert to numpy array
            'ancestors_of_best': generation_data['ancestors_of_best'],
        }
        filtered_trajectory.append(filtered_gen)

    trajectory_data = {
        'expression_id': int(expr_id),  # Add expression ID for reference
        'target_expression': expr_str,
        'final_mse': np.float32(final_mse),
        'final_expression': final_expression,
        'trajectory': filtered_trajectory,
        # Store basicsr_params per trajectory
        'basicsr_params': {
            'population_size': basicsr_params['population_size'],
            'num_generations': basicsr_params['num_generations'],
            'max_depth': basicsr_params['max_depth'],
            'max_size': basicsr_params['max_size'],
            'tournament_size': basicsr_params['tournament_size'],
        },
        # Store operators and constants per trajectory
        'binary_operators': binary_operators,
        'unary_operators': unary_operators,
        'constants': constants,
        # Store X_data and y_data for the expression
        'X_data': X.astype(np.float32),
        'y_data': y.astype(np.float32),
    }

    return trajectory_data


def _process_batched_neural(expressions, checkpoint, basicsr_params,
                            binary_operators, unary_operators, constants,
                            batch_size, seed):
    """Process expressions in batches using batched neural SR"""
    from sr import BatchedNeuralSR

    all_trajectories = []

    # Create batched neural SR instance
    batched_sr = BatchedNeuralSR(
        model_path=checkpoint,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        constants=constants,
        **basicsr_params
    )

    # Process in batches
    for batch_start in range(0, len(expressions), batch_size):
        batch_end = min(batch_start + batch_size, len(expressions))
        batch_exprs = expressions[batch_start:batch_end]

        print(f"\n=== Processing batch {batch_start//batch_size + 1}: expressions {batch_start+1}-{batch_end} ===")

        # Prepare batch data
        X_batch = []
        y_batch = []
        expr_strs = []
        expr_ids = []

        for expr_data in batch_exprs:
            X, y = get_data_from_expr(expr_data, seed=seed)
            X_batch.append(X)
            y_batch.append(y)
            expr_strs.append(expr_data['expression'])
            expr_ids.append(expr_data['id'])

        # Run batched fit
        final_models, final_mses, trajectories = batched_sr.batched_fit(
            X_batch, y_batch
        )

        # Extract trajectory data for each expression in batch
        for i, (expr_id, expr_str, X, y, final_model, final_mse, trajectory) in enumerate(
            zip(expr_ids, expr_strs, X_batch, y_batch, final_models, final_mses, trajectories)
        ):
            # Convert trajectory to storage format
            filtered_trajectory = []
            for generation_data in trajectory:
                filtered_gen = {
                    'generation': generation_data['generation'],
                    'population_size': generation_data['population_size'],
                    'expressions': generation_data['expressions'],
                    'fitnesses': np.array(generation_data['fitnesses'], dtype=np.float32),
                    'ancestors_of_best': generation_data['ancestors_of_best'],
                }
                filtered_trajectory.append(filtered_gen)

            trajectory_data = {
                'expression_id': int(expr_id),  # Add expression ID for reference
                'target_expression': expr_str,
                'final_mse': np.float32(final_mse),
                'final_expression': str(final_model),
                'trajectory': filtered_trajectory,
                'basicsr_params': {
                    'population_size': basicsr_params['population_size'],
                    'num_generations': basicsr_params['num_generations'],
                    'max_depth': basicsr_params['max_depth'],
                    'max_size': basicsr_params['max_size'],
                    'tournament_size': basicsr_params['tournament_size'],
                },
                'binary_operators': binary_operators,
                'unary_operators': unary_operators,
                'constants': constants,
                'X_data': X.astype(np.float32),
                'y_data': y.astype(np.float32),
            }

            all_trajectories.append(trajectory_data)

    return all_trajectories


def generate_traces_from_expressions(expressions_file: str,
                                   output_dir: str = "datasets/traces",
                                   basicsr_params: Dict[str, Any] = None,
                                   max_expressions: int = None,
                                   seed: int = 42,
                                   operator_set: str = "full",
                                   constants: List[float] = None,
                                   checkpoint: str = None,
                                   batch_size: int = 1,
                                   output_file: str = None) -> str:
    """Generate BasicSR traces from expression dataset

    Args:
        operator_set: Either "full" (all operators) or "arith" (add/sub/mul only)
        constants: List of constants to use (if None uses constants from expression generation)
        checkpoint: Path to neural model checkpoint for NeuralSR (if None, uses BasicSR)
        batch_size: Number of expressions to process in parallel (only used with checkpoint)
    """

    binary_operators, unary_operators = get_operators(operator_set)
    # Default BasicSR parameters
    if basicsr_params is None:
        basicsr_params = {
            'population_size': 20,
            'num_generations': 10,
            'max_depth': 10,
            'max_size': 25,
            'tournament_size': 3,
        }

    print(f"=== Generating traces from {expressions_file} ===")

    # Load expressions
    expressions, expr_metadata = load_expressions_dataset(expressions_file)

    if constants is None:
        constants = expr_metadata['parameters']['constants']

    # Limit number of expressions if specified
    if max_expressions:
        # expressions = expressions[:max_expressions]
        expressions = random.sample(expressions, max_expressions)
        print(f"Limiting to {max_expressions} expressions")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_trajectories = []

    print(f"BasicSR parameters: {basicsr_params}")
    print(f"Operator set: {operator_set} (binary: {binary_operators}, unary: {unary_operators})")
    print(f"Constants: {constants}")
    if checkpoint:
        print(f"Checkpoint: {checkpoint}")
        print(f"Batch size: {batch_size}")

    # Process expressions in batches if using neural SR
    if checkpoint and batch_size > 1:
        all_trajectories = _process_batched_neural(
            expressions, checkpoint, basicsr_params,
            binary_operators, unary_operators, constants,
            batch_size, seed
        )
    else:
        # Process one at a time
        for i, expr_data in enumerate(expressions):
            expr_id = expr_data['id']
            expr_str = expr_data['expression']

            print(f"\nProcessing expression {i+1}/{len(expressions)}: ID={expr_id}")
            print(f"Expression: {expr_str}")

            # Get data (uses pre-generated if available, otherwise generates)
            X, y = get_data_from_expr(expr_data, seed=seed)
            print(f"Data shape: X={X.shape}, y range=[{y.min():.3f}, {y.max():.3f}]")

            # Seed BasicSR stochasticity for reproducibility (vary by expression)
            run_seed = seed + int(expr_id)
            random.seed(run_seed)
            np.random.seed(run_seed)

            # Use NeuralSR if checkpoint provided, otherwise BasicSR
            if checkpoint:
                model = NeuralSR(
                    model_path=checkpoint,
                    collect_trajectory=True,
                    binary_operators=binary_operators,
                    unary_operators=unary_operators,
                    constants=constants,
                    record_heritage=True,
                    **basicsr_params,
                )
                trajectory_data = _fit_and_extract_trajectory(
                    model, X, y, expr_str, expr_id, basicsr_params,
                    binary_operators, unary_operators, constants
                )
            else:
                # Run BasicSR with trajectory collection
                model = BasicSR(
                    collect_trajectory=True,
                    binary_operators=binary_operators,
                    unary_operators=unary_operators,
                    constants=constants,
                    record_heritage=True,
                    **basicsr_params,
                )
                trajectory_data = _fit_and_extract_trajectory(
                    model, X, y, expr_str, expr_id, basicsr_params,
                    binary_operators, unary_operators, constants
                )

            all_trajectories.append(trajectory_data)

    # Save trajectories using gzipped pickle (like generate_expressions.py)
    # Format: gen{N}_{original_expression_filename}.pkl.gz
    num_generations = basicsr_params['num_generations']

    # Format generation count: 1k, 10k, 100k, etc.
    if num_generations >= 1000:
        gen_str = f"{num_generations // 1000}k"
    else:
        gen_str = str(num_generations)

    # Extract base filename from expressions_file (remove path and extension)
    expr_basename = os.path.basename(expressions_file)
    if expr_basename.endswith('.pkl.gz'):
        expr_basename = expr_basename[:-7]  # Remove .pkl.gz
    elif expr_basename.endswith('.json'):
        expr_basename = expr_basename[:-5]  # Remove .json

    trace_name = f"gen{gen_str}"
    if checkpoint is not None:
        trace_name += "_neural"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_name += f"_{timestamp}"

    if output_file is None:
        output_file = f"{output_dir}/{trace_name}_{expr_basename}.pkl.gz"

    # Create full dataset with metadata
    full_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_expressions_file': expressions_file,
            'expressions_metadata': expr_metadata,
            'basicsr_params': basicsr_params,
            'operator_set': operator_set,
            'binary_operators': binary_operators,
            'unary_operators': unary_operators,
            'constants': constants,
            'total_expressions_processed': len(expressions),
            'total_trajectories': len(all_trajectories),
            'total_generations': sum(len(traj['trajectory']) for traj in all_trajectories)
        },
        'trajectories': all_trajectories
    }

    # Use gzipped pickle for ~9x compression vs JSON
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n=== Trace Generation Complete ===")
    print(f"Output file: {output_file} ({file_size_mb:.2f} MB)")
    print(f"Expressions processed: {len(expressions)}")
    print(f"Total trajectories: {len(all_trajectories)}")

    total_gens = sum(len(traj['trajectory']) for traj in all_trajectories)
    print(f"Total generations recorded: {total_gens}")

    return output_file


if __name__ == "__main__":
    import time
    start = time.time()

    import argparse

    parser = argparse.ArgumentParser(description="Generate BasicSR traces from expressions")
    parser.add_argument("--expressions_file", required=True, help="Input expressions JSON file")
    parser.add_argument("--output_dir", default="datasets/traces", help="Output directory")
    parser.add_argument("--max_expressions", type=int, help="Limit number of expressions to process")
    parser.add_argument("--population_size", type=int, default=20, help="BasicSR population size")
    parser.add_argument("--num_generations", type=int, default=50, help="BasicSR num generations")
    parser.add_argument("--operator_set", type=str, default="full", choices=["arith", "full"],
                        help="Operator set: 'arith' (add/sub/mul) or 'full' (all operators)")
    parser.add_argument("--constants", type=str, default=None,
                        help="Comma-separated list of constants (default: uses constants used in expression generation. Empty string for no constants)")
    parser.add_argument("--create_one_step", action="store_true", help="Also convert to one-step format and create train/val splits")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to neural model checkpoint for NeuralSR (if not provided, uses BasicSR)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of expressions to process in parallel (only used with --checkpoint)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Parse constants
    if args.constants:
        constants = [float(c.strip()) for c in args.constants.split(',')]
    else:
        constants = None

    basicsr_params = {
        'population_size': args.population_size,
        'num_generations': args.num_generations,
        'max_depth': 10,
        'max_size': 25,
        'tournament_size': 3,
    }

    output_file = generate_traces_from_expressions(
        args.expressions_file,
        args.output_dir,
        basicsr_params,
        args.max_expressions,
        operator_set=args.operator_set,
        constants=constants,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.create_one_step:
        # Convert to one-step format and create train/val splits
        from one_step_conversion import convert_and_make_split
        print(f"\n=== Converting to one-step format and creating train/val splits ===")
        convert_and_make_split(output_file, context_type='basic')

    print(f"\n✓ Trace generation complete: {output_file}")
    end = time.time()
    print(f"Total time: {end - start:.1f}s")
