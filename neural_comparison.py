"""
Compare neural SR vs basic SR performance.
Tests loading a trained checkpoint and running neural evolution.
"""

import numpy as np
import time
from sr import BasicSR, NeuralSR
from problems import HARDER_PROBLEMS
import argparse
from utils import get_operators


def compare_neural_vs_basic(checkpoint_path, problem_idx=0, save_trajectories=False, autoregressive=False, sr_params={}):
    """Compare neural SR vs basic SR on a specific problem"""
    problem = HARDER_PROBLEMS[problem_idx]
    X, y = problem(seed=42)

    print(f"=== Comparing Neural vs Basic SR ===")
    print(f"Problem: {problem.__name__}")
    print(f"Description: {problem.__doc__}")
    print(f"Model type: {'Autoregressive' if autoregressive else 'One-step'}")

    # Test BasicSR
    print("\n--- Basic SR ---")
    basic_sr = BasicSR(**sr_params)

    start_time = time.time()
    basic_sr.fit(X, y, verbose=True)
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
        neural_sr = NeuralSR(model_path=checkpoint_path, autoregressive=autoregressive, **sr_params)

        start_time = time.time()
        neural_sr.fit(X, y, verbose=True)
        neural_time = time.time() - start_time

        y_pred_neural = neural_sr.predict(X)
        mse_neural = np.mean((y - y_pred_neural)**2)

        print(f"Neural SR Results:")
        print(f"  Time: {neural_time:.2f}s")
        print(f"  MSE: {mse_neural:.6f}")
        print(f"  Expression: {neural_sr.best_model_}")
        print(f"  Size: {neural_sr.best_model_.size()}")
        print(f"  Well-formed suggestions: {neural_sr.neural_suggestions_well_formed}/{neural_sr.neural_suggestions_total} ({neural_sr.get_well_formed_percentage():.1f}%)")

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
    parser.add_argument("--max-generations", type=int, default=1000,
                       help="Maximum number of generations (default: 1000)")
    parser.add_argument("--operator_set", type=str, default="arith", choices=["arith", "full"],
                        help="Operator set: 'arith' (add/sub/mul) or 'full' (all operators)")
    parser.add_argument("--autoregressive", action="store_true",
                        help="Use autoregressive model (default: one-step model)")

    args = parser.parse_args()

    binary_operators, unary_operators = get_operators(args.operator_set)

    sr_params = {
        'binary_operators': binary_operators,
        'unary_operators': unary_operators,
        'collect_trajectory': args.collect_trajectories,
        'population_size': 20,
        'num_generations': args.max_generations,
        'max_depth': 4,
        'max_size': 15,
        'constants': [1.0],
    }

    result = compare_neural_vs_basic(
        args.checkpoint,
        args.problem,
        save_trajectories=args.save_trajectories,
        autoregressive=args.autoregressive,
        sr_params=sr_params,
    )
    if result:
        print("\n✓ Comparison completed successfully!")
        if 'neural_well_formed_percentage' in result:
            print(f"Neural well-formed rate: {result['neural_well_formed_percentage']:.1f}%")
    else:
        print("\n❌ Neural comparison failed.")


if __name__ == "__main__":
    main()
