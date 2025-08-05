"""
Trajectory collection functions using the refactored MinimalSR with collect_trajectory flag
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from minimal_sr import MinimalSR
from problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS, ALL_SIMPLE_PROBLEMS


def collect_trajectories_for_problems(problem_list, problem_set_name, num_runs=3):
    """Collect trajectories for a set of problems using MinimalSR with collect_trajectory=True"""
    print(f"=== Collecting trajectories for {problem_set_name} ===")

    all_trajectories = {}

    for i, problem in enumerate(problem_list):
        problem_name = problem.__name__
        print(f"\nCollecting trajectories for problem {i+1}: {problem_name}")

        problem_trajectories = []

        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")

            # Generate data
            X, y = problem(seed=42 + run)  # Different seed for each run

            # Create MinimalSR with trajectory collection enabled
            model = MinimalSR(
                population_size=10,  # Small population size as requested
                num_generations=10000,  # High number, will be stopped by time limit
                max_depth=5,
                max_size=20,
                collect_trajectory=True  # Enable trajectory collection
            )

            # Fit with 15 second time limit using manual evolution loop
            print(f"    Running for 15 seconds...")

            # Initialize evolution manually for time control
            num_vars = X.shape[1]
            population = model.create_initial_population(num_vars)

            start_time = time.time()
            generation = 0

            while time.time() - start_time < 15:  # 15 second time limit
                # Evaluate fitness
                fitnesses = [model.fitness(ind, X, y) for ind in population]

                # Record generation data for trajectory
                if model.collect_trajectory:
                    best_idx = np.argmax(fitnesses)
                    best_individual = population[best_idx]
                    best_fitness = fitnesses[best_idx]

                    generation_data = {
                        'generation': generation,
                        'time': time.time() - start_time,
                        'best_fitness': float(best_fitness),
                        'best_expression': str(best_individual),
                        'best_size': best_individual.size(),
                        'population_diversity': len(set(str(ind) for ind in population))
                    }
                    model.trajectory.append(generation_data)

                # Evolution step
                new_population = []

                # Elitism - keep best individual
                best_idx = np.argmax(fitnesses)
                new_population.append(population[best_idx].copy())

                # Generate rest of population
                while len(new_population) < model.population_size:
                    if np.random.random() < 0.7:  # Crossover
                        parent1 = model.tournament_selection(population, fitnesses)
                        parent2 = model.tournament_selection(population, fitnesses)
                        child = model.crossover(parent1, parent2)
                    else:  # Mutation
                        parent = model.tournament_selection(population, fitnesses)
                        child = model.mutate(parent, num_vars)
                    new_population.append(child)

                population = new_population
                generation += 1

            # Set best model
            final_fitnesses = [model.fitness(ind, X, y) for ind in population]
            best_idx = np.argmax(final_fitnesses)
            model.best_model_ = population[best_idx]

            # Add problem metadata to trajectory
            trajectory_with_metadata = {
                'problem_name': problem_name,
                'problem_doc': problem.__doc__,
                'run_id': run,
                'data_shape': {'X': X.shape, 'y_range': [float(y.min()), float(y.max())]},
                'final_mse': float(np.mean((y - model.predict(X))**2)),
                'trajectory_data': model.trajectory
            }

            problem_trajectories.append(trajectory_with_metadata)

        all_trajectories[problem_name] = problem_trajectories

    # Save all trajectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectories_{problem_set_name}_{timestamp}.json"

    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)

    summary_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'problem_set_name': problem_set_name,
            'num_problems': len(problem_list),
            'num_runs_per_problem': num_runs,
            'total_trajectories': sum(len(trajectories) for trajectories in all_trajectories.values())
        },
        'trajectories': all_trajectories
    }

    with open(filepath, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nAll trajectories saved to {filepath}")
    print(f"Total trajectories collected: {summary_data['metadata']['total_trajectories']}")

    return filepath, all_trajectories


def test_single_trajectory_collection():
    """Test trajectory collection on a single problem"""
    print("Testing single trajectory collection...")

    # Pick one ultra-simple problem
    from problems import single_variable

    X, y = single_variable(seed=42)
    print(f"Problem: {single_variable.__doc__}")
    print(f"Data shape: X={X.shape}, y range=[{y.min():.3f}, {y.max():.3f}]")

    # Create model with trajectory collection
    model = MinimalSR(
        population_size=20,
        num_generations=10,
        max_depth=2,
        max_size=8,
        collect_trajectory=True
    )

    # Fit and collect
    model.fit(X, y)

    # Check trajectory
    print(f"\nTrajectory collected:")
    print(f"- Generations recorded: {len(model.trajectory)}")
    print(f"- Final MSE: {np.mean((y - model.predict(X))**2):.6f}")
    print(f"- Best expression: {model.best_model_}")

    # Save trajectory
    filename = f"test_single_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = model.save_trajectory(filename)

    print(f"✅ Single trajectory test completed successfully!")
    return filepath


def test_trajectory_collection_disabled():
    """Test that MinimalSR works normally when collect_trajectory=False"""
    print("Testing MinimalSR with trajectory collection disabled...")

    from problems import simple_square

    X, y = simple_square(seed=42)

    # Create model WITHOUT trajectory collection (default)
    model = MinimalSR(
        population_size=15,
        num_generations=5,
        max_depth=2,
        max_size=5
    )

    # Fit normally
    model.fit(X, y)

    # Check that trajectory is empty
    print(f"Trajectory length: {len(model.trajectory)} (should be 0)")
    print(f"Final MSE: {np.mean((y - model.predict(X))**2):.6f}")
    print(f"Best expression: {model.best_model_}")

    # Try to save trajectory (should raise error)
    try:
        model.save_trajectory("should_fail.json")
        print("❌ ERROR: save_trajectory should have failed!")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    print("✅ Disabled trajectory test completed successfully!")


def collect_all_simple_trajectories(num_runs=3):
    """Collect trajectories for all 15 problems from all_simple_splits.txt"""
    print("Collecting trajectories for ALL_SIMPLE_PROBLEMS (15 problems total)")
    print("Using 15 second time limit and population size 10")
    print(f"Total estimated time: {15 * 15 * num_runs} seconds = {15 * 15 * num_runs / 60:.1f} minutes")

    # Collect for all 15 problems
    filepath, data = collect_trajectories_for_problems(
        ALL_SIMPLE_PROBLEMS,  # All 15 problems
        "all_simple_15s_n10",
        num_runs=num_runs
    )

    return filepath, data


def main():
    """Collect trajectories for all simple problems for training data"""
    print("Collecting trajectories for training data generation...")
    print("=" * 60)

    # Collect trajectories for all 15 problems
    filepath, data = collect_all_simple_trajectories(num_runs=1)

    print(f"\n✅ Trajectory collection completed!")
    print(f"Data saved to: {filepath}")
    print(f"Ready for conversion to training format.")

    return filepath, data


if __name__ == "__main__":
    main()
