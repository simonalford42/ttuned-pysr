import numpy as np
import time
from pysr import PySRRegressor
from basic_sr import BasicSR
from simple_problems import SIMPLE_PROBLEMS

def test_with_basicsr(problem_func, problem_name):
    """Test a problem with BasicSR"""
    print(f"\nTesting BasicSR on {problem_name}...")
    
    # Generate data
    X, y = problem_func(seed=42)
    
    # Start timer
    start_time = time.time()
    
    # Create and run BasicSR
    model = BasicSR(
        population_size=200,
        tournament_size=5,
        num_generations=50,
        crossover_prob=0.7,
        mutation_prob=0.3,
        max_depth=5,
        max_size=30,
        parsimony_coefficient=0.005
    )
    
    # Scale y values for better numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1
    y_normalized = (y - y_mean) / y_std
    
    # Fit the model
    model.fit(X, y_normalized)
    
    # End timer
    elapsed_time = time.time() - start_time
    
    # Check if a valid model was found
    if model.best_model_ is None:
        print(f"No valid model found for {problem_name}")
        return {
            'problem': problem_name,
            'algorithm': 'BasicSR',
            'expression': "No valid model found",
            'adjusted_expression': "No valid model found",
            'mse': float('inf'),
            'size': 0,
            'time': elapsed_time
        }
    
    # Evaluate performance
    y_pred_normalized = model.predict(X)
    y_pred = y_pred_normalized * y_std + y_mean  # Convert back to original scale
    mse = np.mean((y - y_pred)**2)
    
    # Format expressions
    expression_str = str(model.best_model_)
    adjusted_expression = f"({expression_str}) * {y_std:.4f} + {y_mean:.4f}"
    
    print(f"BasicSR equation: {expression_str}")
    print(f"Adjusted equation: {adjusted_expression}")
    print(f"MSE: {mse:.6f}")
    print(f"Size: {model.best_model_.size()}")
    print(f"Time: {elapsed_time:.2f} seconds")
    
    return {
        'problem': problem_name,
        'algorithm': 'BasicSR',
        'expression': expression_str,
        'adjusted_expression': adjusted_expression,
        'mse': mse,
        'size': model.best_model_.size(),
        'time': elapsed_time
    }

def test_with_pysr(problem_func, problem_name):
    """Test a problem with PySR"""
    print(f"\nTesting PySR on {problem_name}...")
    
    # Generate data
    X, y = problem_func(seed=42)
    
    # Start timer
    start_time = time.time()
    
    # Create PySR model with simple configuration
    model = PySRRegressor(
        niterations=20,
        populations=1,
        population_size=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        select_k_features=None,
        maxsize=10,
        procs=1,
        timeout_in_seconds=60
    )
    
    try:
        # Fit the model
        model.fit(X, y)
        
        # End timer
        elapsed_time = time.time() - start_time
        
        # Get the best expression
        best_equation = model.sympy()
        
        # Evaluate the model
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred)**2)
        
        print(f"PySR equation: {best_equation}")
        print(f"MSE: {mse:.6f}")
        print(f"Time: {elapsed_time:.2f} seconds")
        
        return {
            'problem': problem_name,
            'algorithm': 'PySR',
            'expression': str(best_equation),
            'mse': mse,
            'time': elapsed_time
        }
    except Exception as e:
        print(f"Error running PySR: {e}")
        return {
            'problem': problem_name,
            'algorithm': 'PySR',
            'expression': f"Error: {str(e)}",
            'mse': float('inf'),
            'time': time.time() - start_time
        }

if __name__ == "__main__":
    all_results = []
    
    for i, problem_func in enumerate(SIMPLE_PROBLEMS):
        problem_name = problem_func.__name__
        
        # Test with BasicSR
        basicsr_result = test_with_basicsr(problem_func, problem_name)
        all_results.append(basicsr_result)
        
        # Test with PySR
        pysr_result = test_with_pysr(problem_func, problem_name)
        all_results.append(pysr_result)
    
    # Print summary
    print("\n=== Summary of Results ===")
    print("Problem\tAlgorithm\tMSE\tTime (s)\tExpression")
    for result in all_results:
        expression = result['expression'] if len(str(result['expression'])) < 60 else str(result['expression'])[:57] + "..."
        print(f"{result['problem']}\t{result['algorithm']}\t{result['mse']:.6f}\t{result['time']:.2f}\t{expression}")
    
    # Save results to file for documentation
    with open('simple_problems_results.txt', 'w') as f:
        f.write("=== Simple Problems Results ===\n\n")
        for result in all_results:
            f.write(f"Problem: {result['problem']}\n")
            f.write(f"Algorithm: {result['algorithm']}\n")
            f.write(f"Expression: {result['expression']}\n")
            if 'adjusted_expression' in result:
                f.write(f"Adjusted expression: {result['adjusted_expression']}\n")
            f.write(f"MSE: {result['mse']:.10f}\n")
            if 'size' in result:
                f.write(f"Size: {result['size']}\n")
            f.write(f"Time: {result['time']:.2f} seconds\n")
            f.write("\n")