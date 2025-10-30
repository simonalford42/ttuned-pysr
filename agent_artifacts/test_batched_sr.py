"""
Test script for batched neural SR.

Demonstrates running batched neural SR on multiple expressions simultaneously
and compares performance with sequential processing.
"""

import numpy as np
import argparse
import time
from sr import NeuralSR, BatchedNeuralSR
from utils import get_operators


def generate_test_problems(num_problems=3, seed=42):
    """Generate simple test problems"""
    np.random.seed(seed)

    problems = []
    for i in range(num_problems):
        # Generate random data
        n_samples = 50
        n_vars = 2
        X = np.random.uniform(-3, 3, size=(n_samples, n_vars))

        # Create different target expressions
        if i == 0:
            # x0 + x1
            y = X[:, 0] + X[:, 1]
            expr = "x0 + x1"
        elif i == 1:
            # x0 * x1
            y = X[:, 0] * X[:, 1]
            expr = "x0 * x1"
        else:
            # x0 + (x0 * x1)
            y = X[:, 0] + X[:, 0] * X[:, 1]
            expr = "x0 + (x0 * x1)"

        problems.append({
            'X': X,
            'y': y,
            'expr': expr
        })

    return problems


def main():
    parser = argparse.ArgumentParser(description="Test batched neural SR")
    parser.add_argument("--checkpoint",
                       default="training/checkpoints/tiny_221861/checkpoint-210000",
                       help="Path to trained model checkpoint")
    parser.add_argument("--num_problems", type=int, default=3,
                       help="Number of test problems (default: 3)")
    parser.add_argument("--population_size", type=int, default=20,
                       help="Population size (default: 20)")
    parser.add_argument("--num_generations", type=int, default=50,
                       help="Number of generations (default: 50)")
    parser.add_argument("--operator_set", type=str, default="arith", choices=["arith", "full"],
                        help="Operator set: 'arith' (add/sub/mul) or 'full' (all operators)")
    parser.add_argument("--compare_sequential", action="store_true",
                        help="Also run sequential version for comparison")

    args = parser.parse_args()

    print("=== Batched Neural SR Test ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of problems: {args.num_problems}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.num_generations}")
    print(f"Operator set: {args.operator_set}")

    # Generate test problems
    print("\n--- Generating test problems ---")
    problems = generate_test_problems(args.num_problems)

    for i, prob in enumerate(problems):
        print(f"Problem {i}: {prob['expr']}")
        print(f"  X shape: {prob['X'].shape}")
        print(f"  y range: [{prob['y'].min():.3f}, {prob['y'].max():.3f}]")

    # Extract X and y batches
    X_batch = [prob['X'] for prob in problems]
    y_batch = [prob['y'] for prob in problems]

    # Get operators
    binary_operators, unary_operators = get_operators(args.operator_set)

    # Run batched fitting
    print("\n--- Running BATCHED neural SR ---")
    start_time = time.time()
    batched_sr = BatchedNeuralSR(
        model_path=args.checkpoint,
        population_size=20,
        num_generations=args.num_generations,
        max_depth=4,
        max_size=15,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        constants=[1.0]
    )

    final_models, final_mses, trajectories = batched_sr.batched_fit(
        X_batch=X_batch,
        y_batch=y_batch,
    )
    batched_time = time.time() - start_time
    print(f"\nBatched processing time: {batched_time:.2f}s")

    # Optionally run sequential version for comparison
    if args.compare_sequential:
        print("\n--- Running SEQUENTIAL neural SR (for comparison) ---")
        start_time = time.time()
        sequential_models = []
        sequential_mses = []

        for i, (X, y) in enumerate(zip(X_batch, y_batch)):
            print(f"\nFitting problem {i}...")
            sr = NeuralSR(
                model_path=args.checkpoint,
                population_size=args.population_size,
                num_generations=args.num_generations,
                max_depth=4,
                max_size=15,
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                constants=[1.0],
                collect_trajectory=True
            )
            sr.fit(X, y, verbose=False)

            # Calculate MSE
            y_pred = sr.best_model_.evaluate(X)
            finite_mask = np.isfinite(y_pred)
            mag_mask = np.abs(y_pred) < 1e6
            valid_mask = finite_mask & mag_mask
            if int(np.sum(valid_mask)) >= max(3, int(0.5 * y.shape[0])):
                mse = float(np.mean((y[valid_mask] - y_pred[valid_mask]) ** 2))
            else:
                mse = float('inf')

            sequential_models.append(sr.best_model_)
            sequential_mses.append(mse)
            print(f"  Found: {sr.best_model_} (MSE: {mse:.6e})")

        sequential_time = time.time() - start_time
        print(f"\nSequential processing time: {sequential_time:.2f}s")

        # Calculate speedup
        speedup = sequential_time / batched_time if batched_time > 0 else float('inf')
        print(f"\n{'='*60}")
        print(f"PERFORMANCE COMPARISON:")
        print(f"  Batched time:    {batched_time:.2f}s")
        print(f"  Sequential time: {sequential_time:.2f}s")
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"{'='*60}")

    # Print results
    print("\n=== Batched Results ===")
    for i, (prob, model, mse, traj) in enumerate(zip(problems, final_models, final_mses, trajectories)):
        print(f"\nProblem {i}: {prob['expr']}")
        print(f"  Found: {model}")
        print(f"  MSE: {mse:.6e}")
        print(f"  Size: {model.size()}")
        print(f"  Generations: {len(traj)}")

        # Check if solution is good
        if mse < 1e-6:
            status = "✓ Excellent"
        elif mse < 1e-3:
            status = "✓ Good"
        elif mse < 0.1:
            status = "~ Fair"
        else:
            status = "✗ Poor"
        print(f"  Status: {status}")

    print("\n✓ Batched neural SR test complete!")


if __name__ == "__main__":
    main()
