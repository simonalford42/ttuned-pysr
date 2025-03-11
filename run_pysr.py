import numpy as np
from pysr import PySRRegressor
from problems import PROBLEMS
import time
import pandas as pd
import sympy

# Run PySR on all the problems
def test_with_pysr():
    results = []
    
    for i, problem in enumerate(PROBLEMS):
        print(f"\nTesting on problem {i+1}: {problem.__name__}")
        
        # Generate data
        X, y = problem(seed=42)
        
        # Start timer
        start_time = time.time()
        
        # Create and fit PySR model
        model = PySRRegressor(
            niterations=40,               # Comparable to our num_generations
            population_size=300,          # Comparable to our population_size
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp"],
            maxsize=40,                   # Comparable to our max_size
            parsimony=0.001,              # Comparable to our parsimony_coefficient
            loss="loss(x, y) = (x - y)^2",  # MSE loss function
            batching=False,              # No batching for small datasets
            warm_start=False,
            turbo=True,
            constraints={
                "/": [
                    # Denominator can't be 0
                    "not(x2 == 0)",
                    # Don't allow x/1
                    "not(x2 == 1)",
                ],
                "exp": [
                    # Don't allow large exp values
                    "x1 < 10",
                    "x1 > -10", 
                ],
            },
            # Explicitly set seeds for reproducibility
            procs=1,
            random_state=42,
            timeout_in_seconds=300  # 5 minute limit per problem
        )
        
        model.fit(X, y)
        
        # End timer
        elapsed_time = time.time() - start_time
        
        # Get the best expression
        best_equation = model.sympy()
        best_complexity = model.complexity()
        best_loss = model.loss_
        
        # Format as a string
        expression_str = str(best_equation)
        
        print(f"PySR equation: {expression_str}")
        print(f"Complexity: {best_complexity}")
        print(f"MSE: {best_loss}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        # Store results
        results.append({
            'problem': problem.__name__,
            'mse': best_loss,
            'expression': expression_str,
            'complexity': best_complexity,
            'time': elapsed_time
        })
    
    # Print summary of results
    print("\n=== Summary of PySR Results ===")
    for result in results:
        print(f"{result['problem']}: MSE = {result['mse']:.6f}, Complexity = {result['complexity']}")
        print(f"  Expression: {result['expression']}")
        print(f"  Time: {result['time']:.2f} seconds")
    
    return results

if __name__ == "__main__":
    pysr_results = test_with_pysr()