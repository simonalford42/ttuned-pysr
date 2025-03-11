import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from problems import PROBLEMS
from basic_sr import BasicSR
from pysr import PySRRegressor
import sympy

# Function to run our BasicSR algorithm and collect results
def run_basic_sr():
    results = []
    
    for i, problem in enumerate(PROBLEMS):
        print(f"\nTesting BasicSR on problem {i+1}: {problem.__name__}")
        
        # Generate data
        X, y = problem(seed=42)
        
        # Start timer
        start_time = time.time()
        
        # Run BasicSR
        model = BasicSR(
            population_size=300,
            tournament_size=7,
            num_generations=100,
            crossover_prob=0.8,
            mutation_prob=0.2,
            max_depth=6,
            max_size=40,
            parsimony_coefficient=0.001
        )
        
        # Scale y values for better numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1
        y_normalized = (y - y_mean) / y_std
        
        model.fit(X, y_normalized)
        
        # End timer
        elapsed_time = time.time() - start_time
        
        # Check if a valid model was found
        if model.best_model_ is None:
            print(f"No valid model found for {problem.__name__}")
            results.append({
                'problem': problem.__name__,
                'mse': float('inf'),
                'expression': "No valid model found",
                'adjusted_expression': "No valid model found",
                'size': 0,
                'time': elapsed_time
            })
            continue
        
        # Evaluate performance
        y_pred_normalized = model.predict(X)
        y_pred = y_pred_normalized * y_std + y_mean  # Convert back to original scale
        mse = np.mean((y - y_pred)**2)
        
        # Get expression details
        expression_str = str(model.best_model_)
        adjusted_expression = f"({expression_str}) * {y_std} + {y_mean}"
        model_size = model.best_model_.size()
        
        print(f"BasicSR equation: {expression_str}")
        print(f"Adjusted equation: {adjusted_expression}")
        print(f"Size: {model_size}")
        print(f"MSE: {mse:.6f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        results.append({
            'problem': problem.__name__,
            'mse': mse,
            'expression': expression_str,
            'adjusted_expression': adjusted_expression,
            'size': model_size,
            'time': elapsed_time
        })
    
    return results

# Function to run PySR algorithm and collect results
def run_pysr():
    results = []
    
    for i, problem in enumerate(PROBLEMS):
        print(f"\nTesting PySR on problem {i+1}: {problem.__name__}")
        
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
            elementwise_loss="loss(x, y) = (x - y)^2",  # MSE loss function
            batching=False,              # No batching for small datasets
            warm_start=False,
            turbo=True,
            # Explicitly set seeds for reproducibility
            procs=1,
            random_state=42,
            timeout_in_seconds=300  # 5 minute limit per problem
        )
        
        try:
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
        except Exception as e:
            print(f"Error running PySR on {problem.__name__}: {e}")
            results.append({
                'problem': problem.__name__,
                'mse': float('inf'),
                'expression': f"Error: {str(e)}",
                'complexity': 0,
                'time': time.time() - start_time
            })
    
    return results

# Function to get the ground truth expressions for each problem
def get_ground_truth():
    ground_truth = [
        {
            'problem': 'vlad1',
            'expression': 'exp(-((x1 - 1) ** 2)) / (1.2 + (x2 - 2.5) ** 2)',
            'latex': r'\frac{e^{-(x_1-1)^2}}{1.2 + (x_2-2.5)^2}'
        },
        {
            'problem': 'vlad2',
            'expression': 'exp(-x1) * x1**3 * (cos(x1) * sin(x1)) * (cos(x1) * sin(x1)**2 - 1)',
            'latex': r'e^{-x_1} \cdot x_1^3 \cdot \cos(x_1) \cdot \sin(x_1) \cdot (\cos(x_1) \cdot \sin^2(x_1) - 1)'
        },
        {
            'problem': 'vlad3',
            'expression': 'exp(-x1) * x1**3 * (cos(x1) * sin(x1)) * (cos(x1) * sin(x1)**2 - 1) * (x2 - 5)',
            'latex': r'e^{-x_1} \cdot x_1^3 \cdot \cos(x_1) \cdot \sin(x_1) \cdot (\cos(x_1) \cdot \sin^2(x_1) - 1) \cdot (x_2 - 5)'
        },
        {
            'problem': 'vlad4',
            'expression': '10 / (5 + np.sum((X - 3) ** 2, axis=1))',
            'latex': r'\frac{10}{5 + \sum_i(x_i - 3)^2}'
        },
        {
            'problem': 'vlad5',
            'expression': '30 * (x1 - 1) * (x3 - 1) / ((x1 - 10) * x2**2)',
            'latex': r'\frac{30 \cdot (x_1 - 1) \cdot (x_3 - 1)}{(x_1 - 10) \cdot x_2^2}'
        },
        {
            'problem': 'keijzer4',
            'expression': 'x1**3 * exp(-x1) * cos(x1) * sin(x1) * (sin(x1)**2 * cos(x1) - 1)',
            'latex': r'x_1^3 \cdot e^{-x_1} \cdot \cos(x_1) \cdot \sin(x_1) \cdot (\sin^2(x_1) \cdot \cos(x_1) - 1)'
        },
        {
            'problem': 'keijzer11',
            'expression': 'x1 * x2 + sin((x1 - 1) * (x2 - 1))',
            'latex': r'x_1 \cdot x_2 + \sin((x_1 - 1) \cdot (x_2 - 1))'
        },
        {
            'problem': 'keijzer12',
            'expression': 'x1**4 - x1**3 + (x2**2 / 2) - x2',
            'latex': r'x_1^4 - x_1^3 + \frac{x_2^2}{2} - x_2'
        },
        {
            'problem': 'keijzer13',
            'expression': '6 * sin(x1) * cos(x2)',
            'latex': r'6 \cdot \sin(x_1) \cdot \cos(x_2)'
        },
        {
            'problem': 'keijzer14',
            'expression': '8 / (2 + x1**2 + x2**2)',
            'latex': r'\frac{8}{2 + x_1^2 + x_2^2}'
        }
    ]
    return ground_truth

# Compare results and create a summary
def compare_and_summarize(basic_sr_results, pysr_results, ground_truth):
    # Create a dataframe to compare results
    comparison = []
    
    for problem in set(item['problem'] for item in basic_sr_results):
        basic_sr_result = next((item for item in basic_sr_results if item['problem'] == problem), None)
        pysr_result = next((item for item in pysr_results if item['problem'] == problem), None)
        truth = next((item for item in ground_truth if item['problem'] == problem), None)
        
        comparison.append({
            'Problem': problem,
            'Ground Truth': truth['expression'] if truth else 'Unknown',
            'Ground Truth LaTeX': truth['latex'] if truth else 'Unknown',
            'BasicSR Expression': basic_sr_result['expression'] if basic_sr_result else 'N/A',
            'BasicSR Adjusted': basic_sr_result['adjusted_expression'] if basic_sr_result and 'adjusted_expression' in basic_sr_result else 'N/A',
            'BasicSR MSE': basic_sr_result['mse'] if basic_sr_result else float('inf'),
            'BasicSR Size': basic_sr_result['size'] if basic_sr_result and 'size' in basic_sr_result else 0,
            'BasicSR Time (s)': basic_sr_result['time'] if basic_sr_result and 'time' in basic_sr_result else 0,
            'PySR Expression': pysr_result['expression'] if pysr_result else 'N/A',
            'PySR MSE': pysr_result['mse'] if pysr_result else float('inf'),
            'PySR Complexity': pysr_result['complexity'] if pysr_result and 'complexity' in pysr_result else 0,
            'PySR Time (s)': pysr_result['time'] if pysr_result and 'time' in pysr_result else 0,
        })
    
    # Create a DataFrame
    comparison_df = pd.DataFrame(comparison)
    
    # Save comparison to CSV
    comparison_df.to_csv('comparison_results.csv', index=False)
    
    # Generate a markdown summary
    generate_markdown_summary(comparison_df)
    
    return comparison_df

def generate_markdown_summary(comparison_df):
    with open('comparison_summary.md', 'w') as f:
        f.write("# Comparison of Symbolic Regression Algorithms\n\n")
        
        f.write("## Summary\n\n")
        f.write("This document compares two symbolic regression approaches:\n\n")
        f.write("1. **BasicSR**: A simple evolutionary algorithm implementation built from scratch\n")
        f.write("2. **PySR**: A state-of-the-art symbolic regression package\n\n")
        
        f.write("Both algorithms were tested on a set of benchmark problems to compare their performance.\n\n")
        
        f.write("## Performance Comparison\n\n")
        
        # Create a table summarizing MSE and runtime
        f.write("| Problem | BasicSR MSE | PySR MSE | BasicSR Size | PySR Complexity | BasicSR Time (s) | PySR Time (s) |\n")
        f.write("|---------|------------|----------|--------------|-----------------|------------------|---------------|\n")
        
        for _, row in comparison_df.iterrows():
            # Format MSE with scientific notation for very small/large values
            basic_mse = f"{row['BasicSR MSE']:.4e}" if abs(row['BasicSR MSE']) < 0.001 or abs(row['BasicSR MSE']) > 1000 else f"{row['BasicSR MSE']:.4f}"
            pysr_mse = f"{row['PySR MSE']:.4e}" if abs(row['PySR MSE']) < 0.001 or abs(row['PySR MSE']) > 1000 else f"{row['PySR MSE']:.4f}"
            
            f.write(f"| {row['Problem']} | {basic_mse} | {pysr_mse} | {row['BasicSR Size']} | {row['PySR Complexity']} | {row['BasicSR Time (s)']:.2f} | {row['PySR Time (s)']:.2f} |\n")
        
        # Calculate and write averages
        avg_basic_mse = comparison_df[comparison_df['BasicSR MSE'] < float('inf')]['BasicSR MSE'].mean()
        avg_pysr_mse = comparison_df[comparison_df['PySR MSE'] < float('inf')]['PySR MSE'].mean()
        avg_basic_time = comparison_df['BasicSR Time (s)'].mean()
        avg_pysr_time = comparison_df['PySR Time (s)'].mean()
        
        f.write(f"| **Average** | {avg_basic_mse:.4e} | {avg_pysr_mse:.4e} | - | - | {avg_basic_time:.2f} | {avg_pysr_time:.2f} |\n\n")
        
        f.write("## Discovered Expressions\n\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"### {row['Problem']}\n\n")
            f.write(f"**Ground Truth**: {row['Ground Truth']}\n\n")
            f.write(f"**BasicSR**: {row['BasicSR Expression']}\n\n")
            if 'BasicSR Adjusted' in row and row['BasicSR Adjusted'] != 'N/A':
                f.write(f"**BasicSR (Adjusted)**: {row['BasicSR Adjusted']}\n\n")
            f.write(f"**PySR**: {row['PySR Expression']}\n\n")
            f.write("---\n\n")
        
        f.write("## Analysis\n\n")
        
        # Overall comparison
        f.write("### Accuracy Comparison\n\n")
        if avg_basic_mse < avg_pysr_mse:
            f.write("On average, BasicSR achieved lower MSE than PySR across the test problems. ")
        else:
            f.write("On average, PySR achieved lower MSE than BasicSR across the test problems. ")
        
        # Performance comparison
        f.write("### Performance Comparison\n\n")
        if avg_basic_time < avg_pysr_time:
            f.write("BasicSR was faster than PySR, taking less time on average to find solutions. ")
        else:
            f.write("PySR was faster than BasicSR, taking less time on average to find solutions. ")
        
        # Expression complexity
        f.write("\n\n### Expression Complexity\n\n")
        f.write("The expressions found by PySR tend to be ")
        if comparison_df['PySR Complexity'].mean() < comparison_df['BasicSR Size'].mean():
            f.write("simpler (lower complexity) ")
        else:
            f.write("more complex (higher complexity) ")
        f.write("than those found by BasicSR. This impacts interpretability and potential for overfitting.\n\n")
        
        # General conclusions
        f.write("## Conclusions\n\n")
        f.write("1. **Implementation Complexity**: PySR is a mature, optimized library while BasicSR is a simple implementation built from scratch.\n\n")
        f.write("2. **Quality of Results**: The quality of expressions found varies by problem, with each algorithm having strengths for different types of problems.\n\n")
        f.write("3. **Performance Tradeoffs**: There's a clear tradeoff between execution time and accuracy of results.\n\n")
        f.write("4. **Areas for Improvement**: BasicSR could be enhanced with more sophisticated mutation operators, better simplification rules, and improved numerical stability.\n\n")

# Main function to run the comparison
def main():
    print("Running BasicSR algorithm...")
    basic_sr_results = run_basic_sr()
    
    print("\nRunning PySR algorithm...")
    pysr_results = run_pysr()
    
    print("\nGetting ground truth expressions...")
    ground_truth = get_ground_truth()
    
    print("\nComparing results and generating summary...")
    comparison_df = compare_and_summarize(basic_sr_results, pysr_results, ground_truth)
    
    print("\nComparison completed. Results saved to comparison_results.csv and comparison_summary.md")

if __name__ == "__main__":
    main()