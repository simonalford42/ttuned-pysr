#!/usr/bin/env python3
"""
Generate large datasets (100k - 1M expressions) for training.

Usage:
    python generate_large_dataset.py --n_expressions=1000000 --binary_ops=add,sub,mul --complexity=0.5
"""
import argparse
from generate_expressions import generate_training_expressions


def main():
    parser = argparse.ArgumentParser(description="Generate large expression datasets")
    parser.add_argument("--n_expressions", type=int, default=1_000_000,
                        help="Number of expressions to generate (default: 1M)")
    parser.add_argument("--binary_ops", type=str, default="add,sub,mul",
                        help="Comma-separated binary operators")
    parser.add_argument("--unary_ops", type=str, default="",
                        help="Comma-separated unary operators")
    parser.add_argument("--complexity", type=float, default=0.5,
                        help="Complexity level (0.0 to 1.0)")
    parser.add_argument("--n_input_points", type=int, default=64,
                        help="Number of data points per expression")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", default="datasets/expressions",
                        help="Output directory")

    args = parser.parse_args()

    print(f"Generating {args.n_expressions:,} expressions...")
    print(f"Config: binary_ops={args.binary_ops}, unary_ops={args.unary_ops or 'none'}, "
          f"complexity={args.complexity}")
    print(f"This may take a while...")

    filename = generate_training_expressions(
        n_expressions=args.n_expressions,
        binary_ops=args.binary_ops,
        unary_ops=args.unary_ops,
        complexity=args.complexity,
        n_input_points=args.n_input_points,
        seed=args.seed,
        output_dir=args.output_dir
    )

    print(f"\n✓ Dataset generation complete!")
    print(f"✓ Output: {filename}")


if __name__ == "__main__":
    main()
