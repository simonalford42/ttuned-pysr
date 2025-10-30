"""
Test script for autoregressive NeuralSR model.
Tests the autoreg=True functionality with a trained checkpoint.
"""
import numpy as np
from sr import NeuralSR
from problems import ULTRA_SIMPLE_PROBLEMS

def test_autoreg_model(checkpoint_path="training/checkpoints/tiny_218153/checkpoint-50000"):
    """Test autoregressive NeuralSR on a simple problem"""

    print("=" * 60)
    print("Testing Autoregressive NeuralSR")
    print("=" * 60)

    # Use a simple problem for testing
    problem = ULTRA_SIMPLE_PROBLEMS[0]  # linear function
    X, y = problem(seed=42)

    print(f"\nProblem: {problem.__doc__}")
    print(f"Data shape: X={X.shape}, y range=[{y.min():.3f}, {y.max():.3f}]")

    # Initialize autoregressive model
    print(f"\nLoading autoregressive model from: {checkpoint_path}")
    model = NeuralSR(
        model_path=checkpoint_path,
        autoregressive=True,  # Enable autoregressive mode
        population_size=20,
        num_generations=10,
        max_depth=3,
        max_size=10,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=[],
        constants=[1.0]
    )

    print(f"Model loaded. Autoregressive mode: {model.autoregressive}")
    print(f"Device: {model.device}")

    # Fit the model
    print("\nFitting model...")
    model.fit(X, y, verbose=True)

    # Evaluate
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred)**2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final MSE: {mse:.6f}")
    print(f"Final expression: {model.best_model_}")
    print(f"Expression size: {model.best_model_.size()}")
    print(f"Well-formed percentage: {model.get_well_formed_percentage():.1f}%")
    print(f"Neural suggestions: {model.neural_suggestions_well_formed}/{model.neural_suggestions_total}")

    return model, mse


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test autoregressive NeuralSR")
    parser.add_argument("--checkpoint", type=str,
                       default="training/checkpoints/tiny_218153/checkpoint-50000",
                       help="Path to autoregressive model checkpoint")
    args = parser.parse_args()

    test_autoreg_model(args.checkpoint)
