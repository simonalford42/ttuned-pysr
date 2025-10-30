"""
Thorough evaluation comparing Basic SR vs Neural SR performance.
Runs multiple instances of each algorithm on all harder problems and computes statistics.
"""

import numpy as np
import time
import json
from datetime import datetime
from sr import BasicSR, NeuralSR
from problems import HARDER_PROBLEMS
import argparse
from concurrent.futures import ThreadPoolExecutor
import os


def run_single_instance(algorithm_class, problem_func, seed, checkpoint_path=None, num_generations=1000, population_size=20):
    """Run a single instance of an algorithm on a problem"""
    try:
        X, y = problem_func(seed=seed)

        if algorithm_class == BasicSR:
            sr = BasicSR(
                population_size=population_size,
                num_generations=num_generations,
                max_depth=4,
                max_size=15
            )
        else:  # NeuralSR
            sr = NeuralSR(
                model_path=checkpoint_path,
                population_size=population_size,
                num_generations=num_generations,
                max_depth=4,
                max_size=15
            )

        start_time = time.time()
        sr.fit(X, y)
        end_time = time.time()

        y_pred = sr.predict(X)
        final_mse = np.mean((y - y_pred)**2)

        # Calculate iterations to reach MSE ~0 (threshold 1e-6) using progression tracking
        mse_threshold = 1e-6
        iterations_to_converge = sr.get_iterations_to_convergence(mse_threshold)
        converged = iterations_to_converge is not None

        # Get final MSE from progression (should match calculated MSE)
        progression_final_mse = sr.get_final_mse()

        # Get progression data for detailed analysis
        progression_data = sr.get_progression_data()

        result = {
            'seed': seed,
            'final_mse': float(final_mse),
            'progression_final_mse': float(progression_final_mse),
            'converged': converged,
            'iterations_to_converge': iterations_to_converge,
            'time_seconds': end_time - start_time,
            'expression': str(sr.best_model_),
            'expression_size': sr.best_model_.size(),
            'num_improvements': len(progression_data),
            'progression_data': progression_data
        }

        # Add neural-specific metrics
        if hasattr(sr, 'neural_suggestions_total'):
            result.update({
                'neural_suggestions_total': sr.neural_suggestions_total,
                'neural_suggestions_well_formed': sr.neural_suggestions_well_formed,
                'well_formed_percentage': sr.get_well_formed_percentage()
            })

        return result

    except Exception as e:
        return {
            'seed': seed,
            'error': str(e),
            'final_mse': float('inf'),
            'converged': False,
            'iterations_to_converge': None,
            'time_seconds': 0.0,
            'expression': None,
            'expression_size': 0
        }


def evaluate_algorithm_on_problem(algorithm_class, problem_func, problem_name, n_runs=10, checkpoint_path=None, num_generations=1000, population_size=20):
    """Evaluate an algorithm on a single problem with multiple runs"""
    print(f"\nEvaluating {algorithm_class.__name__} on {problem_name} ({n_runs} runs)...")

    results = []
    seeds = list(range(42, 42 + n_runs))  # Use different seeds for each run

    # Run all instances sequentially for now (parallel might cause GPU memory issues with Neural SR)
    for i, seed in enumerate(seeds):
        print(f"  Run {i+1}/{n_runs} (seed={seed})...")
        result = run_single_instance(algorithm_class, problem_func, seed, checkpoint_path, num_generations, population_size)
        results.append(result)

        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"MSE={result['final_mse']:.2e}, Time={result['time_seconds']:.1f}s")

        print()

    return results


def compute_statistics(results, algorithm_name):
    """Compute statistics from a list of results"""
    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]

    if not valid_results:
        return {
            'algorithm': algorithm_name,
            'total_runs': len(results),
            'valid_runs': 0,
            'error_runs': len(error_results),
            'success_rate': 0.0
        }

    final_mses = [r['final_mse'] for r in valid_results]
    times = [r['time_seconds'] for r in valid_results]
    converged_results = [r for r in valid_results if r['converged']]
    expression_sizes = [r['expression_size'] for r in valid_results]
    num_improvements = [r['num_improvements'] for r in valid_results]

    # Calculate convergence iterations for those that converged
    convergence_iterations = [r['iterations_to_converge'] for r in converged_results if r['iterations_to_converge'] is not None]

    stats = {
        'algorithm': algorithm_name,
        'total_runs': len(results),
        'valid_runs': len(valid_results),
        'error_runs': len(error_results),
        'success_rate': len(valid_results) / len(results) if results else 0.0,

        # MSE statistics
        'final_mse_mean': float(np.mean(final_mses)),
        'final_mse_std': float(np.std(final_mses)),
        'final_mse_median': float(np.median(final_mses)),
        'final_mse_min': float(np.min(final_mses)),
        'final_mse_max': float(np.max(final_mses)),

        # Time statistics
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'time_median': float(np.median(times)),

        # Convergence statistics
        'convergence_count': len(converged_results),
        'convergence_rate': len(converged_results) / len(valid_results) if valid_results else 0.0,
        'convergence_iterations_mean': float(np.mean(convergence_iterations)) if convergence_iterations else None,
        'convergence_iterations_std': float(np.std(convergence_iterations)) if len(convergence_iterations) > 1 else None,
        'convergence_iterations_median': float(np.median(convergence_iterations)) if convergence_iterations else None,

        # Expression size statistics
        'expression_size_mean': float(np.mean(expression_sizes)),
        'expression_size_std': float(np.std(expression_sizes)),
        'expression_size_median': float(np.median(expression_sizes)),

        # Improvement statistics
        'num_improvements_mean': float(np.mean(num_improvements)),
        'num_improvements_std': float(np.std(num_improvements)),
        'num_improvements_median': float(np.median(num_improvements)),

        # Individual results (excluding progression_data to keep file size manageable)
        'individual_results': [{k: v for k, v in r.items() if k != 'progression_data'} for r in valid_results]
    }

    # Add neural-specific statistics if applicable
    neural_results = [r for r in valid_results if 'well_formed_percentage' in r]
    if neural_results:
        well_formed_percentages = [r['well_formed_percentage'] for r in neural_results]
        stats.update({
            'neural_well_formed_mean': float(np.mean(well_formed_percentages)),
            'neural_well_formed_std': float(np.std(well_formed_percentages)),
            'neural_suggestions_total_mean': float(np.mean([r['neural_suggestions_total'] for r in neural_results])),
            'neural_suggestions_well_formed_mean': float(np.mean([r['neural_suggestions_well_formed'] for r in neural_results]))
        })

    return stats


def run_full_evaluation(checkpoint_path, n_runs=10, num_generations=1000, population_size=20):
    """Run full evaluation on all harder problems"""
    print("=== Simple Evaluation: Basic SR vs Neural SR ===")
    print(f"Running {n_runs} instances of each algorithm on {len(HARDER_PROBLEMS)} harder problems")
    print(f"Parameters: {num_generations} generations, population size {population_size}")

    all_results = {}

    for i, problem_func in enumerate(HARDER_PROBLEMS):
        problem_name = problem_func.__name__
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{len(HARDER_PROBLEMS)}: {problem_name}")
        print(f"Description: {problem_func.__doc__}")

        # Test problem to understand data shape
        X_test, y_test = problem_func(seed=42)
        print(f"Data shape: X={X_test.shape}, y={y_test.shape}, y range=[{y_test.min():.3f}, {y_test.max():.3f}]")

        all_results[problem_name] = {}

        # Evaluate Basic SR
        basic_results = evaluate_algorithm_on_problem(
            BasicSR, problem_func, problem_name, n_runs,
            checkpoint_path=None, num_generations=num_generations, population_size=population_size
        )
        all_results[problem_name]['basic_sr'] = compute_statistics(basic_results, 'BasicSR')

        # Evaluate Neural SR
        try:
            neural_results = evaluate_algorithm_on_problem(
                NeuralSR, problem_func, problem_name, n_runs,
                checkpoint_path=checkpoint_path, num_generations=num_generations, population_size=population_size
            )
            all_results[problem_name]['neural_sr'] = compute_statistics(neural_results, 'NeuralSR')
        except Exception as e:
            print(f"Neural SR evaluation failed: {e}")
            all_results[problem_name]['neural_sr'] = {'error': str(e)}

    return all_results


def print_summary(results):
    """Print a summary of the evaluation results"""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    # Problem-by-problem summary
    for problem_name, problem_results in results.items():
        print(f"\n{problem_name.upper()}:")

        if 'basic_sr' in problem_results:
            basic = problem_results['basic_sr']
            convergence_info = ""
            if basic['convergence_iterations_mean'] is not None:
                convergence_info = f", AvgIter={basic['convergence_iterations_mean']:.0f}"
            print(f"  Basic SR:  MSE={basic['final_mse_mean']:.2e}±{basic['final_mse_std']:.2e}, "
                  f"Time={basic['time_mean']:.1f}±{basic['time_std']:.1f}s, "
                  f"Converged={basic['convergence_count']}/{basic['valid_runs']}{convergence_info}")

        if 'neural_sr' in problem_results and 'error' not in problem_results['neural_sr']:
            neural = problem_results['neural_sr']
            convergence_info = ""
            if neural['convergence_iterations_mean'] is not None:
                convergence_info = f", AvgIter={neural['convergence_iterations_mean']:.0f}"
            print(f"  Neural SR: MSE={neural['final_mse_mean']:.2e}±{neural['final_mse_std']:.2e}, "
                  f"Time={neural['time_mean']:.1f}±{neural['time_std']:.1f}s, "
                  f"Converged={neural['convergence_count']}/{neural['valid_runs']}{convergence_info}")
            if 'neural_well_formed_mean' in neural:
                print(f"             Well-formed={neural['neural_well_formed_mean']:.1f}±{neural['neural_well_formed_std']:.1f}%, "
                      f"Improvements={neural['num_improvements_mean']:.1f}±{neural['num_improvements_std']:.1f}")
        elif 'neural_sr' in problem_results:
            print(f"  Neural SR: FAILED - {problem_results['neural_sr']['error']}")

    # Overall summary
    print(f"\n{'='*40}")
    print("OVERALL STATISTICS")
    print(f"{'='*40}")

    basic_mses = []
    neural_mses = []
    basic_times = []
    neural_times = []
    basic_convergence_rates = []
    neural_convergence_rates = []

    for problem_name, problem_results in results.items():
        if 'basic_sr' in problem_results:
            basic = problem_results['basic_sr']
            basic_mses.append(basic['final_mse_mean'])
            basic_times.append(basic['time_mean'])
            basic_convergence_rates.append(basic['convergence_rate'])

        if 'neural_sr' in problem_results and 'error' not in problem_results['neural_sr']:
            neural = problem_results['neural_sr']
            neural_mses.append(neural['final_mse_mean'])
            neural_times.append(neural['time_mean'])
            neural_convergence_rates.append(neural['convergence_rate'])

    if basic_mses:
        print(f"Basic SR  - Avg MSE: {np.mean(basic_mses):.2e}, Avg Time: {np.mean(basic_times):.1f}s, Avg Convergence: {np.mean(basic_convergence_rates)*100:.1f}%")

    if neural_mses:
        print(f"Neural SR - Avg MSE: {np.mean(neural_mses):.2e}, Avg Time: {np.mean(neural_times):.1f}s, Avg Convergence: {np.mean(neural_convergence_rates)*100:.1f}%")

    if basic_mses and neural_mses:
        mse_improvement = np.mean([(b - n) / b for b, n in zip(basic_mses, neural_mses) if b > 0])
        time_ratio = np.mean([n / b for b, n in zip(basic_times, neural_times) if b > 0])
        print(f"Neural vs Basic - MSE improvement: {mse_improvement*100:.1f}%, Time ratio: {time_ratio:.2f}x")


def save_results(results, output_file):
    """Save results to JSON file"""
    # Add metadata
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_problems': len(results),
            'problems': list(results.keys())
        },
        'results': results
    }

    os.makedirs('evaluation_results', exist_ok=True)
    filepath = os.path.join('evaluation_results', output_file)

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Thorough evaluation of Basic SR vs Neural SR")
    parser.add_argument("--checkpoint",
                       default="training/checkpoints/onestep-full_20250811_145316/final_model",
                       help="Path to trained neural model checkpoint")
    parser.add_argument("--n-runs", type=int, default=10,
                       help="Number of runs per algorithm per problem (default: 10)")
    parser.add_argument("--num-generations", type=int, default=1000,
                       help="Number of generations per run (default: 1000)")
    parser.add_argument("--population-size", type=int, default=20,
                       help="Population size (default: 20)")
    parser.add_argument("--output-file", default=None,
                       help="Output JSON file name (default: auto-generated)")

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"simple_evaluation_{timestamp}.json"

    print(f"Starting evaluation with {args.n_runs} runs per algorithm...")
    print(f"Output will be saved to: evaluation_results/{args.output_file}")

    # Run evaluation
    start_time = time.time()
    results = run_full_evaluation(
        args.checkpoint,
        n_runs=args.n_runs,
        num_generations=args.num_generations,
        population_size=args.population_size
    )
    total_time = time.time() - start_time

    # Print summary
    print_summary(results)

    print(f"\nTotal evaluation time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # Save results
    save_results(results, args.output_file)

    print(f"\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
