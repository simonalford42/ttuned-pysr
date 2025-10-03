#!/usr/bin/env python3
"""
Inspect a dataset and print examples.

Usage:
    python inspect_dataset.py <dataset_path> [--n_samples=5]
"""
import argparse
import pickle
import gzip
import numpy as np


def inspect_dataset(dataset_path: str, n_samples: int = 5):
    """Load and inspect a dataset, printing examples and metadata."""
    print(f"Loading dataset: {dataset_path}\n")

    # Load the dataset
    with gzip.open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    metadata = data['metadata']
    expressions = data['expressions']

    # Print metadata
    print("=" * 80)
    print("METADATA")
    print("=" * 80)
    print(f"Generation command: {metadata['generation_command']}")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Generator version: {metadata['generator_version']}")
    print(f"\nParameters:")
    for key, value in metadata['parameters'].items():
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


def main():
    parser = argparse.ArgumentParser(description="Inspect expression dataset")
    parser.add_argument("dataset_path", help="Path to the dataset file (.pkl.gz)")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of examples to show (default: 5)")

    args = parser.parse_args()
    inspect_dataset(args.dataset_path, args.n_samples)


if __name__ == "__main__":
    main()
