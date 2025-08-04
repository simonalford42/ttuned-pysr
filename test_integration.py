"""
Test integration between trajectory collection and problem splits
"""

from collect_trajectories import collect_trajectories_for_problems
from problem_splits import ULTRA_SIMPLE_SPLITS
import json
import os


def test_trajectory_collection_with_splits():
    """Test that trajectory collection works with problem splits"""
    
    print("Testing trajectory collection with problem splits...")
    
    # Use only train set from ultra-simple problems for quick test
    train_problems = ULTRA_SIMPLE_SPLITS['train']
    
    print(f"Collecting trajectories for {len(train_problems)} training problems:")
    for problem in train_problems:
        print(f"  - {problem.__name__}: {problem.__doc__}")
    
    # Collect trajectories (minimal runs for testing)
    trajectory_file, trajectory_data = collect_trajectories_for_problems(
        train_problems, 
        "train_test", 
        num_runs=1  # Just 1 run for testing
    )
    
    # Verify the data structure
    print(f"\nVerifying trajectory data structure...")
    
    with open(trajectory_file, 'r') as f:
        saved_data = json.load(f)
    
    print(f"✓ File saved successfully: {trajectory_file}")
    print(f"✓ Metadata present: {list(saved_data['metadata'].keys())}")
    print(f"✓ Problems in data: {list(saved_data['trajectories'].keys())}")
    
    # Check one trajectory in detail  
    sample_problem = list(saved_data['trajectories'].keys())[0]
    sample_trajectory = saved_data['trajectories'][sample_problem][0]
    
    print(f"\n✓ Sample trajectory for '{sample_problem}':")
    print(f"  - Generations recorded: {len(sample_trajectory['trajectory_data'])}")
    print(f"  - Final MSE: {sample_trajectory['final_mse']:.6f}")
    print(f"  - Data shape: {sample_trajectory['data_shape']}")
    
    # Check trajectory content
    first_gen = sample_trajectory['trajectory_data'][0]
    print(f"  - First generation population size: {first_gen['population_size']}")
    print(f"  - First generation diversity: {first_gen['population_diversity']}")
    
    print(f"\n✅ Integration test completed successfully!")
    print(f"Ready to proceed with transformer training setup.")
    
    return trajectory_file, saved_data


if __name__ == "__main__":
    test_trajectory_collection_with_splits()