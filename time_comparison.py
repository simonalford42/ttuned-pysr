import numpy as np
import random
import time
from minimal_sr import Node, MinimalSR
from problems import HARDER_PROBLEMS, ALL_PROBLEMS


def run_timed_experiment(problem, time_limit_seconds, seed=42):
    """Run MinimalSR for a specific time limit"""

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Generate data
    X, y = problem(seed=seed)

    # Create model with parameters that allow longer runs
    model = MinimalSR(
        population_size=100,
        num_generations=1000,  # High number, will be stopped by time limit
        max_depth=5,
        max_size=20,
        tournament_size=5
    )

    # Track progress during evolution
    num_vars = X.shape[1]
    population = model.create_initial_population(num_vars)

    best_fitness = -float('inf')
    best_individual = None
    generation = 0
    start_time = time.time()

    # Store history
    history = []

    print(f"Running {problem.__name__} for {time_limit_seconds}s...")

    while time.time() - start_time < time_limit_seconds:
        # Evaluate fitness
        fitnesses = [model.fitness(ind, X, y) for ind in population]

        # Track best
        current_best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()

            # Calculate actual MSE for tracking
            y_pred = best_individual.evaluate(X)
            mse = np.mean((y - y_pred)**2)
            elapsed = time.time() - start_time

            history.append({
                'generation': generation,
                'time': elapsed,
                'mse': mse,
                'expression': str(best_individual),
                'size': best_individual.size()
            })

            print(f"  Gen {generation} ({elapsed:.1f}s): MSE={mse:.6f}, Size={best_individual.size()}")

        # Create new population
        new_population = []
        new_population.append(best_individual.copy())

        while len(new_population) < model.population_size:
            if random.random() < 0.7:
                parent1 = model.tournament_selection(population, fitnesses)
                parent2 = model.tournament_selection(population, fitnesses)
                child = model.crossover(parent1, parent2)
            else:
                parent = model.tournament_selection(population, fitnesses)
                child = model.mutate(parent, num_vars)

            new_population.append(child)

        population = new_population
        generation += 1

    # Final result
    final_time = time.time() - start_time
    if best_individual:
        y_pred = best_individual.evaluate(X)
        final_mse = np.mean((y - y_pred)**2)
    else:
        final_mse = float('inf')
        best_individual = Node(1.0)  # Fallback

    return {
        'problem': problem.__name__,
        'time_limit': time_limit_seconds,
        'actual_time': final_time,
        'generations': generation,
        'final_mse': final_mse,
        'final_expression': str(best_individual),
        'final_size': best_individual.size(),
        'history': history
    }


def compare_time_limits():
    """Compare algorithm performance across different time limits"""

    time_limits = [60, 300, 600]  # 1 min, 5 min, 10 min

    results = []

    print("=== Time Comparison Experiment ===")
    print("Testing harder problems at different time limits")

    for problem in HARDER_PROBLEMS:
        print(f"\n--- Testing {problem.__name__} ---")
        print(f"Ground truth: {problem.__doc__}")

        problem_results = []

        for time_limit in time_limits:
            result = run_timed_experiment(problem, time_limit)
            problem_results.append(result)
            results.append(result)

            print(f"\n{time_limit}s result:")
            print(f"  Generations: {result['generations']}")
            print(f"  Final MSE: {result['final_mse']:.6f}")
            print(f"  Expression: {result['final_expression']}")
            print(f"  Size: {result['final_size']}")

        print("-" * 60)

    return results


def analyze_time_results(results):
    """Analyze the time comparison results"""

    print("\n" + "="*80)
    print("TIME COMPARISON ANALYSIS")
    print("="*80)

    # Group by problem
    problems = {}
    for result in results:
        if result['problem'] not in problems:
            problems[result['problem']] = []
        problems[result['problem']].append(result)

    analysis = {}

    for problem_name, problem_results in problems.items():
        # Sort by time limit
        problem_results.sort(key=lambda x: x['time_limit'])

        print(f"\n{problem_name}:")

        mse_improvements = []
        gen_scaling = []

        for i, result in enumerate(problem_results):
            time_limit = result['time_limit']
            mse = result['final_mse']
            gens = result['generations']

            print(f"  {time_limit:3d}s: MSE={mse:.2e}, Gens={gens:4d}, Expr={result['final_expression'][:40]}...")

            if i > 0:
                prev_mse = problem_results[i-1]['final_mse']
                if prev_mse > 0:
                    improvement = (prev_mse - mse) / prev_mse * 100
                    mse_improvements.append(improvement)

                prev_gens = problem_results[i-1]['generations']
                gen_ratio = gens / prev_gens if prev_gens > 0 else 0
                gen_scaling.append(gen_ratio)

        analysis[problem_name] = {
            'results': problem_results,
            'mse_improvements': mse_improvements,
            'generation_scaling': gen_scaling
        }

    # Overall analysis
    print(f"\n{'='*50}")
    print("OVERALL PATTERNS:")

    all_improvements = []
    all_scaling = []

    for prob_analysis in analysis.values():
        all_improvements.extend(prob_analysis['mse_improvements'])
        all_scaling.extend(prob_analysis['generation_scaling'])

    if all_improvements:
        avg_improvement = np.mean(all_improvements)
        print(f"Average MSE improvement per time increase: {avg_improvement:.1f}%")

    if all_scaling:
        avg_scaling = np.mean(all_scaling)
        print(f"Average generation scaling per time increase: {avg_scaling:.1f}x")

    # Find diminishing returns
    print(f"\nDiminishing Returns Analysis:")
    for problem_name, prob_analysis in analysis.items():
        if len(prob_analysis['mse_improvements']) >= 2:
            early_improvement = prob_analysis['mse_improvements'][0]  # 1min -> 5min
            late_improvement = prob_analysis['mse_improvements'][1]   # 5min -> 10min

            if early_improvement > late_improvement * 2:
                print(f"  {problem_name}: Strong diminishing returns (early: {early_improvement:.1f}%, late: {late_improvement:.1f}%)")
            elif early_improvement > late_improvement:
                print(f"  {problem_name}: Moderate diminishing returns (early: {early_improvement:.1f}%, late: {late_improvement:.1f}%)")
            else:
                print(f"  {problem_name}: Consistent/increasing returns (early: {early_improvement:.1f}%, late: {late_improvement:.1f}%)")

    return analysis


if __name__ == "__main__":
    results = compare_time_limits()
    analysis = analyze_time_results(results)

    # Save results for markdown generation
    import json

    # Convert to JSON-serializable format
    json_results = []
    for result in results:
        json_result = result.copy()
        # Remove history for cleaner JSON (it's complex nested data)
        json_result['history_length'] = len(result['history'])
        del json_result['history']
        json_results.append(json_result)

    with open('time_comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to time_comparison_results.json")
    print(f"Ready for markdown generation!")
