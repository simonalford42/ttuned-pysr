"""
Generate BasicSR traces from expression datasets.
Builds on collect_trajectories.py and basic_sr.py.
"""
import json
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from basic_sr import BasicSR
import random


def load_expressions_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load expressions from generated dataset"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    expressions = data['expressions']
    metadata = data['metadata']
    
    print(f"Loaded {len(expressions)} expressions from {filepath}")
    print(f"Generated with: {metadata.get('parameters', {})}")
    
    return expressions, metadata


def create_data_function(expr_data: Dict[str, Any]):
    """Create a data generation function from expression data"""
    expr_str = expr_data['expression']
    variables = expr_data['variables']
    n_vars = len(variables)
    
    def data_func(seed):
        """Generate data for this expression"""
        rstate = np.random.RandomState(seed)
        X = rstate.uniform(-3, 3, size=(50, n_vars))
        
        try:
            # Build evaluation context
            context = {}
            for i, var in enumerate(variables):
                if i < X.shape[1]:
                    context[var] = X[:, i]
            
            # Add standard functions and constants
            context.update({
                'np': np, 'sqrt': np.sqrt, 'exp': np.exp, 
                'log': np.log, 'sin': np.sin, 'cos': np.cos
            })
            
            # Evaluate expression
            y = eval(expr_str, {"__builtins__": {}}, context)
            y = np.array(y)
            
            # Handle edge cases
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                # Fallback to simple quadratic
                y = X[:, 0] ** 2
                
            return X, y
            
        except Exception as e:
            print(f"Warning: Failed to evaluate {expr_str}, using fallback. Error: {e}")
            # Fallback to simple expression
            return X, X[:, 0] ** 2
    
    # Add metadata to function
    data_func.__name__ = f"expr_{expr_data['id']}"
    data_func.__doc__ = expr_data['description']
    
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
    
    all_trajectories = {}
    
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
        
        # Store trajectory data
        trajectory_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'expression_id': expr_id,
                'target_expression': expr_str,
                'variables': expr_data['variables'],
                'description': expr_data['description'],
                'population_size': model.population_size,
                'num_generations': model.num_generations,
                'max_depth': model.max_depth,
                'max_size': model.max_size,
                'tournament_size': model.tournament_size,
                'time_limit': basicsr_params.get('time_limit', 30),
                'actual_time': actual_time,
                'total_generations_recorded': len(model.trajectory),
                'final_mse': final_mse,
                'final_expression': final_expression
            },
            'trajectory': model.trajectory
        }
        
        all_trajectories[f"expr_{expr_id}"] = [trajectory_data]
    
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
            'total_trajectories': sum(len(runs) for runs in all_trajectories.values()),
            'total_generations': sum(
                len(run['trajectory']) 
                for runs in all_trajectories.values() 
                for run in runs
            )
        },
        'trajectories': all_trajectories
    }
    
    with open(output_file, 'w') as f:
        json.dump(full_data, f, indent=2)
    
    print(f"\n=== Trace Generation Complete ===")
    print(f"Output file: {output_file}")
    print(f"Expressions processed: {len(expressions)}")
    print(f"Total trajectories: {len(all_trajectories)}")
    
    total_gens = sum(
        len(run['trajectory']) 
        for runs in all_trajectories.values() 
        for run in runs
    )
    print(f"Total generations recorded: {total_gens}")
    
    return output_file


def create_test_traces(expressions_file: str, n_expressions: int = 10) -> str:
    """Create test traces with limited expressions and generations"""
    print(f"Creating test traces from {expressions_file}")
    
    basicsr_params = {
        'population_size': 20,
        'num_generations': 10,  # Very limited for testing
        'max_depth': 3,
        'max_size': 10,
        'tournament_size': 3,
        'time_limit': 15  # 15 seconds per expression
    }
    
    return generate_traces_from_expressions(
        expressions_file=expressions_file,
        output_dir="datasets/traces",
        basicsr_params=basicsr_params,
        max_expressions=n_expressions
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate BasicSR traces from expressions")
    parser.add_argument("--expressions_file", required=True, help="Input expressions JSON file")
    parser.add_argument("--output_dir", default="datasets/traces", help="Output directory")
    parser.add_argument("--max_expressions", type=int, help="Limit number of expressions to process")
    parser.add_argument("--population_size", type=int, default=20, help="BasicSR population size")
    parser.add_argument("--num_generations", type=int, default=50, help="BasicSR num generations")
    parser.add_argument("--time_limit", type=int, default=30, help="Time limit per expression (seconds)")
    parser.add_argument("--test", action="store_true", help="Generate small test dataset")
    
    args = parser.parse_args()
    
    basicsr_params = {
        'population_size': args.population_size,
        'num_generations': args.num_generations,
        'max_depth': 4,
        'max_size': 15,
        'tournament_size': 3,
        'time_limit': args.time_limit
    }
    
    if args.test:
        output_file = create_test_traces(args.expressions_file, args.max_expressions or 10)
    else:
        output_file = generate_traces_from_expressions(
            args.expressions_file,
            args.output_dir,
            basicsr_params,
            args.max_expressions
        )
    
    print(f"\n✓ Trace generation complete: {output_file}")
