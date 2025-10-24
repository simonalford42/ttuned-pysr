#!/usr/bin/env python3
"""
Inspect a dataset and print examples.

Usage:
    python inspect_dataset.py <dataset_path> [--n_samples=5]

Supports expression datasets, trace datasets, and one-step training datasets (jsonl).
"""
import argparse
import pickle
import gzip
import json
import numpy as np


def inspect_expression_dataset(data: dict, n_samples: int = 5):
    """Inspect an expression dataset."""
    metadata = data['metadata']
    expressions = data['expressions']

    # Print metadata
    print("=" * 80)
    print("EXPRESSION DATASET METADATA")
    print("=" * 80)
    if 'generation_command' in metadata:
        print(f"Generation command: {metadata['generation_command']}")
    print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"Generator version: {metadata.get('generator_version', 'N/A')}")
    print(f"\nParameters:")
    params = metadata.get('parameters', {})
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Dataset stats
    print(f"\n" + "=" * 80)
    print("DATASET STATS")
    print("=" * 80)
    print(f"Total expressions: {len(expressions)}")

    if expressions:
        first_expr = expressions[0]
        print(f"X_data shape: {first_expr['X_data'].shape}")
        print(f"X_data dtype: {first_expr['X_data'].dtype}")
        print(f"y_data shape: {first_expr['y_data'].shape}")
        print(f"y_data dtype: {first_expr['y_data'].dtype}")

    # Print examples
    print(f"\n" + "=" * 80)
    print(f"EXAMPLES (showing {min(n_samples, len(expressions))} of {len(expressions)})")
    print("=" * 80)

    for i in range(min(n_samples, len(expressions))):
        expr = expressions[i]
        print(f"\nExample {i+1}:")
        print(f"  Expression: {expr['expression']}")
        print(f"  X shape: {expr['X_data'].shape}, y shape: {expr['y_data'].shape}")
        print(f"  X sample (first 3 points):")
        for j in range(min(3, len(expr['X_data']))):
            x_vals = ', '.join([f'{v:.3f}' for v in expr['X_data'][j]])
            print(f"    [{x_vals}] -> y={expr['y_data'][j]:.3f}")


def inspect_trace_dataset(data: dict, n_samples: int = 5, stats_sample_size: int = 1000):
    """Inspect a trace dataset."""
    metadata = data['metadata']
    trajectories = data['trajectories']

    # Print metadata
    print("=" * 80)
    print("TRACE DATASET METADATA")
    print("=" * 80)
    print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"Source expressions file: {metadata.get('source_expressions_file', 'N/A')}")
    print(f"Operator set: {metadata.get('operator_set', 'N/A')}")
    print(f"Binary operators: {metadata.get('binary_operators', [])}")
    print(f"Unary operators: {metadata.get('unary_operators', [])}")
    print(f"Constants: {metadata.get('constants', [])}")
    print(f"\nBasicSR parameters:")
    params = metadata.get('basicsr_params', {})
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Dataset stats
    print(f"\n" + "=" * 80)
    print("DATASET STATS")
    print("=" * 80)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Total expressions processed: {metadata.get('total_expressions_processed', 'N/A')}")
    print(f"Total generations: {metadata.get('total_generations', 'N/A')}")

    if trajectories:
        # Calculate average generations per trajectory using a sample
        sample_size = min(stats_sample_size, len(trajectories))
        if sample_size < len(trajectories):
            # Random sample for stats
            indices = np.random.choice(len(trajectories), size=sample_size, replace=False)
            sample_trajs = [trajectories[i] for i in indices]
            print(f"\nStats calculated from random sample of {sample_size} trajectories:")
        else:
            sample_trajs = trajectories
            print(f"\nStats calculated from all {len(trajectories)} trajectories:")

        avg_gens = np.mean([len(t['trajectory']) for t in sample_trajs])
        print(f"Average generations per trajectory: {avg_gens:.1f}")

    # Print examples
    print(f"\n" + "=" * 80)
    print(f"EXAMPLES (showing {min(n_samples, len(trajectories))} of {len(trajectories)})")
    print("=" * 80)

    for i in range(min(n_samples, len(trajectories))):
        traj = trajectories[i]
        print(f"\nTrajectory {i+1}:")
        print(f"  Target expression: {traj['target_expression']}")
        print(f"  Final MSE: {traj['final_mse']:.6e}")
        print(f"  Final expression: {traj['final_expression']}")
        print(f"  Generations: {len(traj['trajectory'])}")

        # Show first and last generation
        if traj['trajectory']:
            first_gen = traj['trajectory'][0]
            last_gen = traj['trajectory'][-1]

            print(f"  First generation (gen {first_gen['generation']}):")
            print(f"    Population size: {first_gen['population_size']}")
            print(f"    Best fitness: {np.min(first_gen['fitnesses']):.6e}")
            print(f"    Best expression: {first_gen['expressions'][np.argmin(first_gen['fitnesses'])]}")

            print(f"  Last generation (gen {last_gen['generation']}):")
            print(f"    Population size: {last_gen['population_size']}")
            print(f"    Best fitness: {np.min(last_gen['fitnesses']):.6e}")
            print(f"    Best expression: {last_gen['expressions'][np.argmin(last_gen['fitnesses'])]}")


def inspect_onestep_dataset(dataset_path: str, n_samples: int = 5):
    """Inspect a one-step training dataset (jsonl format)."""
    print("=" * 80)
    print("ONE-STEP TRAINING DATASET")
    print("=" * 80)

    # Count total lines
    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total training examples: {total_lines}")

    # Parse a few examples to understand structure
    if lines:
        first_example = json.loads(lines[0])
        print(f"\nStructure:")
        print(f"  Keys: {list(first_example.keys())}")
        if 'metadata' in first_example:
            print(f"  Metadata keys: {list(first_example['metadata'].keys())}")

    # Print examples
    print(f"\n" + "=" * 80)
    print(f"EXAMPLES (showing {min(n_samples, total_lines)} of {total_lines})")
    print("=" * 80)

    for i in range(min(n_samples, total_lines)):
        example = json.loads(lines[i])
        print(f"\nExample {i+1}:")
        print(f"  Context: {example['context']}")
        print(f"  Target: {example['target']}")
        if 'metadata' in example:
            meta = example['metadata']
            print(f"  Metadata:")
            print(f"    Target expression: {meta.get('target_expression', 'N/A')}")
            print(f"    Generation: {meta.get('generation', 'N/A')}")
            print(f"    Target index: {meta.get('target_index', 'N/A')}")

        # Show a snippet of the population
        pop = example.get('population', '')
        if pop:
            print(f"  Population: {pop}")


def inspect_dataset(dataset_path: str, n_samples: int = 5, stats_sample_size: int = 1000, interactive: bool = False):
    """Load and inspect a dataset, automatically detecting type."""
    print(f"Loading dataset: {dataset_path}\n")

    # Check file extension to determine format
    if dataset_path.endswith('.jsonl'):
        # One-step training dataset
        inspect_onestep_dataset(dataset_path, n_samples)
    elif dataset_path.endswith('.pkl.gz'):
        # Pickle gzip format (expression or trace dataset)
        with gzip.open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        # Detect dataset type
        if 'expressions' in data:
            inspect_expression_dataset(data, n_samples)
        elif 'trajectories' in data:
            inspect_trace_dataset(data, n_samples, stats_sample_size)
        else:
            print("ERROR: Unknown dataset format!")
            print("Expected either 'expressions' or 'trajectories' key in data.")
            print(f"Found keys: {list(data.keys())}")
    else:
        print(f"ERROR: Unsupported file format: {dataset_path}")
        print("Supported formats: .pkl.gz (pickle gzip), .jsonl (one-step training)")
        return

    # print the structure of the data file
    # from utils import print_structure
    # print_structure(data)

    if interactive:
        import pdb; pdb.set_trace()


def main():
    parser = argparse.ArgumentParser(description="Inspect expression, trace, or one-step training dataset")
    parser.add_argument("dataset_path", help="Path to the dataset file (.pkl.gz or .jsonl)")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of examples to show (default: 5)")
    parser.add_argument("--stats_sample_size", type=int, default=50,
                        help="Number of trajectories to sample for stats calculation (default: 1000)")
    parser.add_argument("--interactive", action="store_true",
                        help="Drop into interactive debugger after inspection")

    args = parser.parse_args()
    inspect_dataset(args.dataset_path, args.n_samples, args.stats_sample_size, args.interactive)


if __name__ == "__main__":
    main()
