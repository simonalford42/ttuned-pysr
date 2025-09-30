"""
Generate BasicSR traces from expression datasets.
Builds on collect_trajectories.py and basic_sr.py.
"""
import json
import os
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from basic_sr import BasicSR
import random
import sympy as sp


def load_expressions_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load expressions from generated dataset"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    expression_strings = data['expressions']
    metadata = data['metadata']

    # Convert expression strings back to dict format for processing
    expressions = []
    for i, expr_str in enumerate(expression_strings):
        expressions.append({
            'id': i,
            'expression': expr_str,
            'variables': [f'x{j}' for j in range(metadata['parameters']['max_vars'])],  # Use max from metadata
            'description': f"Expression: {expr_str}"
        })

    print(f"Loaded {len(expressions)} expressions from {filepath}")
    print(f"Generated with: {metadata.get('parameters', {})}")

    return expressions, metadata


def create_data_function(expr_data: Dict[str, Any]):
    """Create a data generation function from expression data (domain-aware)."""
    expr_str = expr_data['expression']
    # Parse expression and infer variables as real symbols
    expr = sp.sympify(expr_str)
    symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
    if not symbols:
        symbols = [sp.Symbol('x0', real=True)]
    n_vars = len(symbols)

    f = sp.lambdify(symbols, expr, 'numpy')

    def data_func(seed):
        """Generate domain-filtered data for this expression"""
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

    # Add metadata to function
    data_func.__name__ = f"expr_{expr_data['id']}"
    data_func.__doc__ = expr_data.get('description', f"Expression: {expr_str}")

    return data_func


def generate_traces_from_expressions(expressions_file: str,
                                   output_dir: str = "datasets/traces",
                                   basicsr_params: Dict[str, Any] = None,
                                   max_expressions: int = None,
                                   seed: int = 42) -> str:
    """Generate BasicSR traces from expression dataset"""

    # Default BasicSR parameters
    if basicsr_params is None:
        basicsr_params = {
            'population_size': 20,
            'num_generations': 10,  # Small for test
            'max_depth': 4,
            'max_size': 15,
            'tournament_size': 3,
            'time_limit': 30  # 30 seconds per expression
        }

    print(f"=== Generating traces from {expressions_file} ===")
    print(f"BasicSR parameters: {basicsr_params}")

    # Load expressions
    expressions, expr_metadata = load_expressions_dataset(expressions_file)

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

        # Create data function for this expression
        data_func = create_data_function(expr_data)

        # Generate data
        X, y = data_func(seed=seed)
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
            min_generations=2
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

        # Store trajectory data in simplified format (remove fitnesses for training)
        filtered_trajectory = []
        for generation_data in model.trajectory:
            filtered_gen = {
                'generation': generation_data['generation'],
                'population_size': generation_data['population_size'],
                'expressions': generation_data['expressions'],
                'best_fitness': generation_data['best_fitness'],
                'best_expression': generation_data['best_expression'],
                'population_diversity': generation_data['population_diversity']
            }
            filtered_trajectory.append(filtered_gen)

        trajectory_data = {
            'metadata': {
                'target_expression': expr_str,
                'final_mse': final_mse,
                'final_expression': final_expression
            },
            'trajectory': filtered_trajectory
        }

        all_trajectories.append(trajectory_data)

    # Save trajectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/traces_{timestamp}.json"

    # Create full dataset with metadata
    full_data = {
        'generation_metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_expressions_file': expressions_file,
            'expressions_metadata': expr_metadata,
            'basicsr_params': basicsr_params,
            'total_expressions_processed': len(expressions),
            'total_trajectories': len(all_trajectories),
            'total_generations': sum(len(traj['trajectory']) for traj in all_trajectories)
        },
        'trajectories': all_trajectories
    }

    with open(output_file, 'w') as f:
        json.dump(full_data, f, indent=2)

    print(f"\n=== Trace Generation Complete ===")
    print(f"Output file: {output_file}")
    print(f"Expressions processed: {len(expressions)}")
    print(f"Total trajectories: {len(all_trajectories)}")

    total_gens = sum(len(traj['trajectory']) for traj in all_trajectories)
    print(f"Total generations recorded: {total_gens}")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate BasicSR traces from expressions")
    parser.add_argument("--expressions_file", required=True, help="Input expressions JSON file")
    parser.add_argument("--output_dir", default="datasets/traces", help="Output directory")
    parser.add_argument("--max_expressions", type=int, help="Limit number of expressions to process")
    parser.add_argument("--population_size", type=int, default=20, help="BasicSR population size")
    parser.add_argument("--num_generations", type=int, default=50, help="BasicSR num generations")

    args = parser.parse_args()

    basicsr_params = {
        'population_size': args.population_size,
        'num_generations': args.num_generations,
        'max_depth': 4,
        'max_size': 15,
        'tournament_size': 3,
    }

    output_file = generate_traces_from_expressions(
        args.expressions_file,
        args.output_dir,
        basicsr_params,
        args.max_expressions
    )

    print(f"\n✓ Trace generation complete: {output_file}")
