import numpy as np
from pysr import PySRRegressor
from problems import PROBLEMS
import time

def test_pysr(problem_index=6):
    # Default: keijzer11 (x1 * x2 + sin((x1 - 1) * (x2 - 1)))
    problem = PROBLEMS[problem_index]
    problem_name = problem.__name__
    
    print(f"Testing PySR on {problem_name}...")
    
    # Generate data
    X, y = problem(seed=42)
    
    # Start timer
    start_time = time.time()
    
    # Create and fit PySR model with simplest possible configuration
    model = PySRRegressor(
        niterations=10,  # Very few iterations for testing
        populations=1,
        population_size=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        select_k_features=None,  # Use all features
        maxsize=10,  # Small trees for simplicity
        procs=1,  # Single process
        timeout_in_seconds=60  # Short timeout
    )
    
    try:
        # Fit the model
        model.fit(X, y)
        
        # End timer
        elapsed_time = time.time() - start_time
        
        # Get the best expression
        best_equation = model.sympy()
        best_score = model.score(X, y)
        best_loss = 1 - best_score  # Approximate MSE from score
        
        print(f"\nResults for {problem_name}:")
        print(f"PySR equation: {best_equation}")
        print(f"R² Score: {best_score:.6f}")
        print(f"Approx MSE: {best_loss:.6f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        # Evaluate on data to verify
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred)**2)
        print(f"Verified MSE: {mse:.6f}")
        
        print("\nGround truth expression:")
        if problem_name == "keijzer11":
            print("x1 * x2 + sin((x1 - 1) * (x2 - 1))")
        
        return {
            'problem': problem_name,
            'expression': str(best_equation),
            'mse': mse,
            'time': elapsed_time
        }
    
    except Exception as e:
        print(f"Error running PySR: {e}")
        return {
            'problem': problem_name,
            'expression': f"Error: {e}",
            'mse': float('inf'),
            'time': time.time() - start_time
        }

if __name__ == "__main__":
    # Test on multiple problems
    results = []
    
    # Try keijzer11 (already tested successfully)
    results.append(test_pysr(6))  # keijzer11
    
    # Try keijzer14 (should be a simple inverse quadratic)
    results.append(test_pysr(9))  # keijzer14
    
    # Try vlad1 (more complex expression)
    results.append(test_pysr(0))  # vlad1
    
    print("\nSummary of Results:")
    for result in results:
        print(f"{result['problem']}: MSE = {result['mse']:.6f}, Equation: {result['expression']}, Time: {result['time']:.2f}s")