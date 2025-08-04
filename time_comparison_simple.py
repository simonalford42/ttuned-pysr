import numpy as np
import random
import time
from minimal_sr import Node, MinimalSR
from problems import HARDER_PROBLEMS


def run_timed_experiment(problem, time_limit_seconds, seed=42):
    """Run MinimalSR for a specific time limit and return results"""

    random.seed(seed)
    np.random.seed(seed)

    # Generate data
    X, y = problem(seed=seed)

    # Create model
    model = MinimalSR(
        population_size=100,
        num_generations=10000,  # Will be stopped by time limit
        max_depth=5,
        max_size=20,
        tournament_size=5
    )

    # Initialize
    num_vars = X.shape[1]
    population = model.create_initial_population(num_vars)

    best_fitness = -float('inf')
    best_individual = None
    generation = 0
    start_time = time.time()
    improvements = []  # Track when improvements happen

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

            # Record improvement
            y_pred = best_individual.evaluate(X)
            mse = np.mean((y - y_pred)**2)
            elapsed = time.time() - start_time

            improvements.append({
                'generation': generation,
                'time': elapsed,
                'mse': mse,
                'size': best_individual.size()
            })

            print(f"  Gen {generation:4d} ({elapsed:5.1f}s): MSE={mse:.6f}")

        # Evolution step
        new_population = [best_individual.copy()]  # Elitism

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

    # Return results
    final_time = time.time() - start_time
    if best_individual:
        y_pred = best_individual.evaluate(X)
        final_mse = np.mean((y - y_pred)**2)
    else:
        final_mse = float('inf')
        best_individual = Node(1.0)

    return {
        'problem': problem.__name__,
        'time_limit': time_limit_seconds,
        'actual_time': final_time,
        'generations': generation,
        'final_mse': final_mse,
        'final_expression': str(best_individual),
        'final_size': best_individual.size(),
        'improvements': improvements
    }


def compare_times_single_problem(problem, time_limits=[60, 300, 600]):
    """Compare different time limits on a single problem"""

    print(f"\n{'='*60}")
    print(f"TESTING: {problem.__name__}")
    print(f"Ground truth: {problem.__doc__}")
    print('='*60)

    results = []

    for time_limit in time_limits:
        print(f"\n--- {time_limit} seconds ---")
        result = run_timed_experiment(problem, time_limit)
        results.append(result)

        print(f"Final: {result['generations']} gens, MSE={result['final_mse']:.6f}")
        print(f"Expression: {result['final_expression']}")
        print(f"Improvements made: {len(result['improvements'])}")

    # Quick comparison
    print(f"\n--- COMPARISON ---")
    print(f"{'Time':<6} {'Gens':<6} {'MSE':<12} {'Improvements':<12} {'Expression':<30}")
    print("-" * 70)

    for result in results:
        print(f"{result['time_limit']:<6} {result['generations']:<6} {result['final_mse']:<12.2e} "
              f"{len(result['improvements']):<12} {result['final_expression'][:29]:<30}")

    return results


def quick_test():
    """Quick test on just the first harder problem"""

    print("QUICK TIME COMPARISON TEST")
    print("Testing just the first harder problem with 1min, 3min, 5min")

    # Test just one problem with shorter times for demo
    problem = HARDER_PROBLEMS[0]  # pythagorean_3d
    results = compare_times_single_problem(problem, time_limits=[60, 180, 300])

    return results


def full_comparison():
    """Run each harder problem for 5 minutes and track improvement trajectory"""

    print("IMPROVEMENT TRAJECTORY ANALYSIS")
    print("Running each problem for 1 minute and tracking progress...")
    print(f"Testing {len(HARDER_PROBLEMS)} problems × 1 minute each = ~5 minutes total")

    all_results = []

    for i, problem in enumerate(HARDER_PROBLEMS):
        print(f"\nProblem {i+1}/{len(HARDER_PROBLEMS)}: {problem.__name__}")
        print(f"Ground truth: {problem.__doc__}")

        # Run for 1 minute and track trajectory
        result = run_timed_experiment(problem, time_limit_seconds=60, seed=42)
        all_results.append(result)

        # Show trajectory summary
        improvements = result['improvements']
        print(f"\nTrajectory Summary:")
        print(f"  Total improvements: {len(improvements)}")
        print(f"  Final MSE: {result['final_mse']:.2e}")
        print(f"  Final expression: {result['final_expression']}")

        if len(improvements) >= 2:
            first_mse = improvements[0]['mse']
            last_mse = improvements[-1]['mse']
            total_improvement = (first_mse - last_mse) / first_mse * 100 if first_mse > 0 else 0
            print(f"  Total improvement: {total_improvement:.1f}% MSE reduction")

            # Show key milestones
            print(f"  Key milestones:")
            for j, imp in enumerate(improvements[:5]):  # Show first 5 improvements
                print(f"    {imp['time']:5.1f}s (Gen {imp['generation']:3d}): MSE={imp['mse']:.2e}")
            if len(improvements) > 5:
                print(f"    ... ({len(improvements)-5} more improvements)")

        print("-" * 60)

    # Save results
    import json
    with open('improvement_trajectories.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to improvement_trajectories.json")
    return all_results


def create_markdown_table(results):
    """Create markdown table focusing on improvement trajectories"""

    markdown = "# Improvement Trajectory Analysis\n\n"
    markdown += "Each problem was run for 1 minute to track how the algorithm discovers mathematical structure over time.\n\n"

    for result in results:
        problem_name = result['problem']
        improvements = result['improvements']

        markdown += f"## {problem_name}\n\n"

        # Add problem description if available
        # Note: We'd need to modify the script to pass this through

        markdown += f"**Final Result**: {result['final_mse']:.2e} MSE in {result['generations']} generations  \n"
        markdown += f"**Expression**: `{result['final_expression']}`\n\n"

        if len(improvements) > 0:
            markdown += f"**Improvement Trajectory** ({len(improvements)} improvements):\n\n"
            markdown += "| Time | Generation | MSE | Size | Improvement |\n"
            markdown += "|------|------------|-----|------|-------------|\n"

            prev_mse = None
            for imp in improvements:  # Show ALL improvements
                improvement_pct = ""
                if prev_mse is not None and prev_mse > 0:
                    pct = (prev_mse - imp['mse']) / prev_mse * 100
                    improvement_pct = f"{pct:.1f}%"

                markdown += f"| {imp['time']:.1f}s | {imp['generation']} | {imp['mse']:.2e} | {imp['size']} | {improvement_pct} |\n"
                prev_mse = imp['mse']
        else:
            markdown += "**No improvements recorded** (likely found perfect solution immediately)\n"

        # Add analysis
        if len(improvements) >= 2:
            first_mse = improvements[0]['mse']
            last_mse = improvements[-1]['mse']
            total_improvement = (first_mse - last_mse) / first_mse * 100 if first_mse > 0 else 0

            markdown += f"\n**Analysis**:\n"
            markdown += f"- Total improvement: {total_improvement:.1f}% MSE reduction\n"
            markdown += f"- First improvement at {improvements[0]['time']:.1f}s (Gen {improvements[0]['generation']})\n"
            markdown += f"- Last improvement at {improvements[-1]['time']:.1f}s (Gen {improvements[-1]['generation']})\n"

            # Improvement rate analysis
            early_improvements = [imp for imp in improvements if imp['time'] <= 15]  # First 15 seconds
            late_improvements = [imp for imp in improvements if imp['time'] > 45]    # Last 15 seconds

            markdown += f"- Early phase (0-15s): {len(early_improvements)} improvements\n"
            markdown += f"- Late phase (45-60s): {len(late_improvements)} improvements\n"

            if len(early_improvements) > len(late_improvements) * 2:
                markdown += "- **Pattern**: Front-loaded discovery (most progress early)\n"
            elif len(late_improvements) > len(early_improvements):
                markdown += "- **Pattern**: Late-stage breakthroughs (progress throughout)\n"
            else:
                markdown += "- **Pattern**: Steady progress throughout run\n"

        markdown += "\n---\n\n"

    return markdown


if __name__ == "__main__":
    print("Improvement Trajectory Analysis")
    print("===============================")
    print()
    print("Running full analysis on all 5 harder problems (1 minute each)...")

    # Run full comparison automatically
    results = full_comparison()

    # Generate markdown
    if results:
        markdown = create_markdown_table(results)
        with open('improvement_trajectories.md', 'w') as f:
            f.write(markdown)
        print(f"\nMarkdown analysis saved to improvement_trajectories.md")
