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
from typing import List, Dict, Any
from basic_sr import BasicSR
import random
import sympy as sp


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

    print(f"Loaded {len(expressions)} expressions from {filepath}")
    print(f"Generated with: {metadata.get('parameters', {})}")

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


def generate_traces_from_expressions(expressions_file: str,
                                   output_dir: str = "datasets/traces",
                                   basicsr_params: Dict[str, Any] = None,
                                   max_expressions: int = None,
                                   seed: int = 42,
                                   operator_set: str = "full",
                                   constants: List[float] = [1.0]) -> str:
    """Generate BasicSR traces from expression dataset

    Args:
        operator_set: Either "full" (all operators) or "arith" (add/sub/mul only)
        constants: List of constants to use (default: [1.0])
    """

    # Map operator sets to BasicSR operators
    if operator_set == "arith":
        binary_operators = ["+", "-", "*"]
        unary_operators = []
    elif operator_set == "full":
        binary_operators = ["+", "-", "*", "/", "^"]
        unary_operators = ["abs", "sqrt", "sin", "cos", "tan", "inv"]
    else:
        raise ValueError(f"Unknown operator_set: {operator_set}. Use 'arith' or 'full'")

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
    print(f"BasicSR parameters: {basicsr_params}")
    print(f"Operator set: {operator_set} (binary: {binary_operators}, unary: {unary_operators})")
    print(f"Constants: {constants}")

    # Load expressions
    expressions, expr_metadata = load_expressions_dataset(expressions_file)

    # Check if expressions were generated with constants
    expr_constants = expr_metadata.get('parameters', {}).get('constants', [1.0])

    # If expressions don't use constants, BasicSR shouldn't either
    if not expr_constants or len(expr_constants) == 0:
        if constants and len(constants) > 0:
            print(f"⚠ Expression dataset was generated WITHOUT constants")
            print(f"  Ignoring provided constants {constants} and using empty constant set for BasicSR")
        constants = []
        print(f"Expression dataset has no constants; BasicSR will use no constants")
    else:
        # Use constants from expression metadata if not explicitly overridden
        print(f"Expression dataset uses constants: {expr_constants}")
        if constants == [1.0]:  # If using default, prefer expression metadata
            constants = expr_constants
            print(f"Using constants from expression metadata: {constants}")

    # Limit number of expressions if specified
    if max_expressions:
        expressions = expressions[:max_expressions]
        print(f"Limited to first {max_expressions} expressions")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_trajectories = []

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

        # Run BasicSR with trajectory collection
        model = BasicSR(
            population_size=basicsr_params['population_size'],
            num_generations=basicsr_params['num_generations'],
            max_depth=basicsr_params['max_depth'],
            max_size=basicsr_params['max_size'],
            tournament_size=basicsr_params['tournament_size'],
            collect_trajectory=True,
            time_limit=basicsr_params.get('time_limit', 30),
            # Ensure at least 2 generations are recorded before early stopping kicks in
            early_stop=True,
            early_stop_threshold=3e-16,
            min_generations=2,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            constants=constants,
        )

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
            }
            filtered_trajectory.append(filtered_gen)

        trajectory_data = {
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

    output_file = f"{output_dir}/gen{gen_str}_{expr_basename}.pkl.gz"

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
    parser.add_argument("--constants", type=str, default="1.0",
                        help="Comma-separated list of constants (default: 1.0, empty string for no constants)")
    parser.add_argument("--create_one_step", action="store_true", help="Also convert to one-step format and create train/val splits")

    args = parser.parse_args()

    # Parse constants
    if args.constants.strip():
        constants_list = [float(c.strip()) for c in args.constants.split(',')]
    else:
        constants_list = []

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
        constants=constants_list
    )

    if args.create_one_step:
        # Convert to one-step format and create train/val splits
        from one_step_conversion import convert_and_make_split
        print(f"\n=== Converting to one-step format and creating train/val splits ===")
        convert_and_make_split(output_file, context_type='basic')

    print(f"\n✓ Trace generation complete: {output_file}")
    end = time.time()
    print(f"Total time: {end - start:.1f}s")
