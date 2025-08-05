"""
Trajectory collection functions using BasicSR with proper trajectory collection
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from basic_sr import BasicSR
from problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS, HARDER_PROBLEMS


def collect_trajectories_for_problems(problem_list, problem_set_name, num_runs=3, time_limit=15):
    """Collect trajectories for a set of problems using BasicSR with collect_trajectory=True"""
    print(f"=== Collecting trajectories for {problem_set_name} ===")

    all_trajectories = {}

    for i, problem in enumerate(problem_list):
        problem_name = problem.__name__
        print(f"\nCollecting trajectories for problem {i+1}: {problem_name}")
        print(f"Ground truth: {problem.__doc__}")

        problem_trajectories = []

        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs} - {time_limit}s time limit")

            # Generate data
            X, y = problem(seed=42 + run)  # Different seed for each run

            # Create BasicSR with trajectory collection enabled
            model = BasicSR(
                population_size=20,  # Match the test_single_trajectory format
                num_generations=10000,  # High number, will be stopped by time limit
                max_depth=4,
                max_size=15,
                tournament_size=3,
                collect_trajectory=True,  # Enable trajectory collection
                time_limit=time_limit  # Use built-in time limit
            )

            # Fit the model - this will automatically collect trajectory data
            start_time = time.time()
            model.fit(X, y)
            actual_time = time.time() - start_time

            # Get final results
            if model.best_model_:
                y_pred = model.predict(X)
                final_mse = np.mean((y - y_pred)**2)
            else:
                final_mse = float('inf')

            # Create trajectory data with proper metadata format
            trajectory_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'problem_name': problem_name,
                    'problem_description': problem.__doc__,
                    'run_number': run + 1,
                    'population_size': model.population_size,
                    'num_generations': model.num_generations,
                    'max_depth': model.max_depth,
                    'max_size': model.max_size,
                    'tournament_size': model.tournament_size,
                    'time_limit': time_limit,
                    'actual_time': actual_time,
                    'total_generations_recorded': len(model.trajectory),
                    'final_mse': final_mse,
                    'final_expression': str(model.best_model_) if model.best_model_ else None
                },
                'trajectory': model.trajectory  # This contains the full population data
            }

            problem_trajectories.append(trajectory_data)
            print(f"    Completed in {actual_time:.1f}s, {len(model.trajectory)} generations recorded")

        all_trajectories[problem_name] = problem_trajectories

    return all_trajectories


def save_trajectories(trajectories, filename_prefix):
    """Save trajectories to JSON files"""
    os.makedirs('data', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all trajectories in one file
    all_filename = f"data/{filename_prefix}_all_{timestamp}.json"
    with open(all_filename, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f"All trajectories saved to {all_filename}")
    
    # Also save individual problem trajectories for easier analysis
    for problem_name, problem_trajectories in trajectories.items():
        problem_filename = f"data/{filename_prefix}_{problem_name}_{timestamp}.json"
        with open(problem_filename, 'w') as f:
            json.dump(problem_trajectories, f, indent=2)
        print(f"{problem_name} trajectories saved to {problem_filename}")
    
    return all_filename


def test_single_trajectory():
    """Test trajectory collection on a single problem"""
    print("Testing trajectory collection on single problem...")
    
    from problems import simple_square
    
    # Generate data
    X, y = simple_square(seed=42)
    print(f"Problem: {simple_square.__doc__}")
    print(f"Data shape: X={X.shape}, y range=[{y.min():.3f}, {y.max():.3f}]")
    
    # Create model with trajectory collection
    model = BasicSR(
        population_size=20,
        num_generations=10,  # Small number for testing
        max_depth=2,
        max_size=8,
        tournament_size=3,
        collect_trajectory=True
    )
    
    # Fit and collect trajectory
    model.fit(X, y)
    
    # Create trajectory data
    trajectory_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'population_size': model.population_size,
            'num_generations': model.num_generations,
            'max_depth': model.max_depth,
            'max_size': model.max_size,
            'tournament_size': model.tournament_size,
            'total_generations_recorded': len(model.trajectory)
        },
        'trajectory': model.trajectory
    }
    
    # Save test trajectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/test_single_trajectory_{timestamp}.json"
    os.makedirs('data', exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"Test trajectory saved to {filename}")
    print(f"Trajectory contains {len(model.trajectory)} generations")
    
    return filename


def test_trajectory_disabled():
    """Test that BasicSR works normally when collect_trajectory=False"""
    print("Testing BasicSR with trajectory collection disabled...")
    
    from problems import simple_square
    X, y = simple_square(seed=42)
    
    model = BasicSR(
        population_size=10,
        num_generations=5,
        collect_trajectory=False  # Disabled
    )
    
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred)**2)
    
    print(f"Final MSE: {mse:.6f}")
    print(f"Final expression: {model.best_model_}")
    print(f"Trajectory length: {len(model.trajectory) if model.trajectory else 0}")
    
    assert len(model.trajectory) == 0, "Trajectory should be empty when disabled"
    print("✓ Trajectory collection properly disabled")


if __name__ == "__main__":
    print("Trajectory Collection for Harder Problems")
    print("="*50)
    print(f"Collecting trajectories for {len(HARDER_PROBLEMS)} harder problems")
    print("Parameters: 60 seconds per problem, 2 runs per problem")
    print(f"Estimated total time: ~{len(HARDER_PROBLEMS) * 60 * 2 / 60:.0f} minutes")
    print()
    
    # Collect trajectories for all harder problems
    trajectories = collect_trajectories_for_problems(
        HARDER_PROBLEMS,
        "harder_problems",
        num_runs=2,
        time_limit=60
    )
    
    # Save the collected trajectories
    filename = save_trajectories(trajectories, "harder_problems")
    
    print("\n" + "="*50)
    print("COLLECTION SUMMARY:")
    total_trajectories = 0
    total_generations = 0
    
    for problem_name, runs in trajectories.items():
        print(f"\n{problem_name}:")
        for run_idx, run_data in enumerate(runs):
            num_gens = len(run_data['trajectory'])
            final_mse = run_data['metadata']['final_mse']
            actual_time = run_data['metadata']['actual_time']
            total_trajectories += 1
            total_generations += num_gens
            print(f"  Run {run_idx+1}: {num_gens:4d} generations, {actual_time:5.1f}s, MSE={final_mse:.2e}")
    
    print(f"\nTOTAL: {total_trajectories} trajectory runs, {total_generations} total generations recorded")
    print(f"Average: {total_generations/total_trajectories:.0f} generations per run")
    print(f"\n✓ All trajectories saved to {filename}")