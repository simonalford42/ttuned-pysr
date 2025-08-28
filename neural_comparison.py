"""
Compare neural SR vs basic SR performance.
Tests loading a trained checkpoint and running neural evolution.
"""

import numpy as np
import time
from basic_sr import BasicSR, NeuralSR
from problems import HARDER_PROBLEMS
import argparse


def compare_neural_vs_basic(checkpoint_path, problem_idx=0, collect_trajectories=False, save_trajectories=False, num_generations=1000):
    """Compare neural SR vs basic SR on a specific problem"""
    problem = HARDER_PROBLEMS[problem_idx]
    X, y = problem(seed=42)

    print(f"=== Comparing Neural vs Basic SR ===")
    print(f"Problem: {problem.__name__}")
    print(f"Description: {problem.__doc__}")

    # Test BasicSR
    print("\n--- Basic SR ---")
    basic_sr = BasicSR(
        population_size=20,
        num_generations=num_generations,
        max_depth=4,
        max_size=15,
        collect_trajectory=collect_trajectories
    )

    start_time = time.time()
    basic_sr.fit(X, y)
    basic_time = time.time() - start_time

    y_pred_basic = basic_sr.predict(X)
    mse_basic = np.mean((y - y_pred_basic)**2)

    print(f"Basic SR Results:")
    print(f"  Time: {basic_time:.2f}s")
    print(f"  MSE: {mse_basic:.6f}")
    print(f"  Expression: {basic_sr.best_model_}")
    print(f"  Size: {basic_sr.best_model_.size()}")

    # Test NeuralSR
    print("\n--- Neural SR ---")
    try:
        neural_sr = NeuralSR(
            model_path=checkpoint_path,
            population_size=20,
            num_generations=num_generations,
            max_depth=4,
            max_size=15,
            collect_trajectory=collect_trajectories
        )

        start_time = time.time()
        neural_sr.fit(X, y)
        neural_time = time.time() - start_time

        y_pred_neural = neural_sr.predict(X)
        mse_neural = np.mean((y - y_pred_neural)**2)

        print(f"Neural SR Results:")
        print(f"  Time: {neural_time:.2f}s")
        print(f"  MSE: {mse_neural:.6f}")
        print(f"  Expression: {neural_sr.best_model_}")
        print(f"  Size: {neural_sr.best_model_.size()}")
        print(f"  Well-formed suggestions: {neural_sr.neural_suggestions_well_formed}/{neural_sr.neural_suggestions_total} ({neural_sr.get_well_formed_percentage():.1f}%)")

        # Save trajectories if requested
        if save_trajectories and collect_trajectories:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            basic_trajectory_file = f"basic_sr_trajectory_{problem.__name__}_{timestamp}.json"
            neural_trajectory_file = f"neural_sr_trajectory_{problem.__name__}_{timestamp}.json"

            basic_sr.save_trajectory(basic_trajectory_file)
            neural_sr.save_trajectory(neural_trajectory_file)

        # Comparison summary
        print(f"\n--- Comparison ---")
        print(f"MSE improvement: {(mse_basic - mse_neural):.6f}")
        print(f"Time ratio (neural/basic): {neural_time/basic_time:.2f}")

        return {
            'basic_mse': mse_basic,
            'neural_mse': mse_neural,
            'basic_time': basic_time,
            'neural_time': neural_time,
            'basic_expr': str(basic_sr.best_model_),
            'neural_expr': str(neural_sr.best_model_),
            'neural_well_formed_total': neural_sr.neural_suggestions_total,
            'neural_well_formed_count': neural_sr.neural_suggestions_well_formed,
            'neural_well_formed_percentage': neural_sr.get_well_formed_percentage()
        }

    except Exception as e:
        print(f"Neural SR failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare neural vs basic SR")
    parser.add_argument("--checkpoint",
                       default="training/checkpoints/onestep-full_20250811_145316/final_model",
                       help="Path to trained model checkpoint")
    parser.add_argument("--problem", type=int, default=0,
                       help="Problem index to test (default: 0)")
    parser.add_argument("--collect-trajectories", action="store_true",
                       help="Collect trajectories during evolution")
    parser.add_argument("--save-trajectories", action="store_true",
                       help="Save trajectories to files (requires --collect-trajectories)")

    args = parser.parse_args()

    result = compare_neural_vs_basic(
        args.checkpoint,
        args.problem,
        collect_trajectories=args.collect_trajectories,
        save_trajectories=args.save_trajectories
    )
    if result:
        print("\n✓ Comparison completed successfully!")
        if 'neural_well_formed_percentage' in result:
            print(f"Neural well-formed rate: {result['neural_well_formed_percentage']:.1f}%")
    else:
        print("\n❌ Neural comparison failed.")


if __name__ == "__main__":
    main()
