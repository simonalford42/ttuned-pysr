#!/usr/bin/env python3
"""
Test script to generate traces from expression dataset and verify one-step conversion.

This script:
1. Generates traces from a small expression dataset (10 generations per trace)
2. Verifies the trace format
3. Converts traces to one-step training format
4. Verifies the conversion output
"""
import os
import pickle
import gzip
from generate_traces import generate_traces_from_expressions
from convert_data import convert_basicsr_to_one_step_format


def test_trace_generation():
    """Test generating traces from expression dataset"""
    print("=" * 80)
    print("TEST 1: Generate traces from expression dataset")
    print("=" * 80)

    # Use small test dataset
    expressions_file = "datasets/expressions/arith_2_c03_20251003_120124.pkl.gz"

    # Generate traces with 10 generations
    basicsr_params = {
        'population_size': 20,
        'num_generations': 3,
        'max_depth': 10,
        'max_size': 25,
        'tournament_size': 3,
    }

    output_file = generate_traces_from_expressions(
        expressions_file=expressions_file,
        output_dir="datasets/traces",
        basicsr_params=basicsr_params,
        seed=42,
        operator_set="arith",
        constants=[1.0]
    )

    print(f"\n✓ Generated traces: {output_file}")

    # Verify trace format
    print("\n" + "=" * 80)
    print("Verifying trace format...")
    print("=" * 80)

    with gzip.open(output_file, 'rb') as f:
        data = pickle.load(f)

    # Check metadata
    assert 'metadata' in data, "Missing metadata"
    assert 'trajectories' in data, "Missing trajectories"

    metadata = data['metadata']
    trajectories = data['trajectories']

    print(f"✓ Found {len(trajectories)} trajectories")
    print(f"✓ Metadata keys: {list(metadata.keys())}")

    # Check first trajectory structure
    traj = trajectories[0]
    required_keys = ['target_expression', 'trajectory', 'basicsr_params',
                     'binary_operators', 'unary_operators', 'constants',
                     'X_data', 'y_data']

    for key in required_keys:
        assert key in traj, f"Missing key '{key}' in trajectory"

    print(f"✓ Trajectory has all required keys: {required_keys}")
    print(f"✓ Target expression: {traj['target_expression']}")
    print(f"✓ Number of generations: {len(traj['trajectory'])}")
    print(f"✓ BasicSR params: {traj['basicsr_params']}")
    print(f"✓ Binary operators: {traj['binary_operators']}")
    print(f"✓ Constants: {traj['constants']}")
    print(f"✓ X_data shape: {traj['X_data'].shape}")
    print(f"✓ y_data shape: {traj['y_data'].shape}")

    return output_file


def test_one_step_conversion(trace_file):
    """Test converting traces to one-step format"""
    print("\n" + "=" * 80)
    print("TEST 2: Convert traces to one-step training format")
    print("=" * 80)

    output_file = "datasets/training/test_conversion.jsonl"

    converted_data = convert_basicsr_to_one_step_format(
        input_file=trace_file,
        output_file=output_file,
        context_type='basic'
    )

    print(f"\n✓ Converted {len(converted_data)} training examples")

    # Verify conversion format
    print("\n" + "=" * 80)
    print("Verifying conversion format...")
    print("=" * 80)

    # Check first example
    example = converted_data[0]
    required_keys = ['context', 'population', 'target', 'metadata']

    for key in required_keys:
        assert key in example, f"Missing key '{key}' in example"

    print(f"✓ Example has all required keys: {required_keys}")
    print(f"\nExample structure:")
    print(f"  Context: {example['context'][:100]}...")
    print(f"  Population: {example['population'][:100]}...")
    print(f"  Target: {example['target']}")
    print(f"  Metadata: {example['metadata']}")

    # Clean up test file
    # os.remove(output_file)
    # print(f"\n✓ Cleaned up test file: {output_file}")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TRACE GENERATION AND CONVERSION TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Generate traces
        trace_file = test_trace_generation()

        # Test 2: Convert to one-step format
        test_one_step_conversion(trace_file)

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print(f"\nGenerated trace file: {trace_file}")
        print("You can inspect it with:")
        print(f"  python inspect_dataset.py {trace_file}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
