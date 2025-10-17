"""
Fine-grained comparison of neural SR vs basic SR.
Compares generation by generation with user input to proceed.
"""

import numpy as np
import time
from basic_sr import BasicSR, NeuralSR
from problems import HARDER_PROBLEMS
import argparse
from utils import get_operators


def compare_neural_vs_basic_fine_grained(checkpoint_path, problem_idx=0, max_generations=1000, binary_operators=['add', 'sub', 'mul'], unary_operators=[]):
    """Compare neural SR vs basic SR generation by generation with user input"""
    problem = HARDER_PROBLEMS[problem_idx]
    X, y = problem(seed=42)
    num_vars = X.shape[1]

    print(f"=== Fine-Grained Comparison: Neural vs Basic SR ===")
    print(f"Problem: {problem.__name__}")
    print(f"Description: {problem.__doc__}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Press Enter after each generation to proceed...\n")

    # Initialize both algorithms
    basic_sr = BasicSR(
        population_size=20,
        num_generations=1,  # We'll run one generation at a time
        max_depth=4,
        max_size=15,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
    )

    try:
        neural_sr = NeuralSR(
            model_path=checkpoint_path,
            population_size=20,
            num_generations=1,  # We'll run one generation at a time
            max_depth=4,
            max_size=15,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
        )
    except Exception as e:
        print(f"Failed to initialize Neural SR: {e}")
        return None

    # Initialize populations
    basic_population = basic_sr.create_initial_population(num_vars)
    neural_population = neural_sr.create_initial_population(num_vars)

    basic_best_fitness = -float('inf')
    basic_best_individual = None
    neural_best_fitness = -float('inf')
    neural_best_individual = None

    # Run generation by generation
    for generation in range(max_generations):
        print(f"=== Generation {generation} ===")

        # Evaluate Basic SR population
        basic_fitnesses = np.array([basic_sr.fitness(ind, X, y) for ind in basic_population])
        basic_current_best_idx = np.argmax(basic_fitnesses)
        basic_current_best_fitness = basic_fitnesses[basic_current_best_idx]

        if basic_current_best_fitness > basic_best_fitness:
            basic_best_fitness = basic_current_best_fitness
            basic_best_individual = basic_population[basic_current_best_idx].copy()

        # Evaluate Neural SR population
        neural_fitnesses = np.array([neural_sr.fitness(ind, X, y) for ind in neural_population])
        neural_current_best_idx = np.argmax(neural_fitnesses)
        neural_current_best_fitness = neural_fitnesses[neural_current_best_idx]

        if neural_current_best_fitness > neural_best_fitness:
            neural_best_fitness = neural_current_best_fitness
            neural_best_individual = neural_population[neural_current_best_idx].copy()

        # Format and display individuals
        print("Basic SR individuals:")
        basic_individuals_str = format_individuals_for_display(basic_population)
        print(f"  {basic_individuals_str}")
        basic_best_mse = np.mean((y - basic_best_individual.evaluate(X))**2) if basic_best_individual else float('inf')
        print(f"  Best: {basic_best_individual} (MSE: {basic_best_mse:.6f})")

        print("\nNeural SR individuals:")
        neural_individuals_str = format_individuals_for_display(neural_population)
        print(f"  {neural_individuals_str}")
        neural_best_mse = np.mean((y - neural_best_individual.evaluate(X))**2) if neural_best_individual else float('inf')
        print(f"  Best: {neural_best_individual} (MSE: {neural_best_mse:.6f})")
        print(f"  Well-formed: {neural_sr.neural_suggestions_well_formed}/{neural_sr.neural_suggestions_total}")

        # Check for early stopping
        if basic_best_mse <= 3e-16 or neural_best_mse <= 3e-16:
            print(f"\nNear-zero MSE reached. Stopping evolution.")
            break

        # Wait for user input
        user_input = input("\nPress Enter to continue to next generation (or 'q' to quit): ")
        if user_input.strip().lower() == 'q':
            break

        # Generate new populations for next iteration
        if generation < max_generations - 1:  # Don't generate if this is the last iteration
            basic_population, _ = basic_sr.generate_new_population(basic_population, basic_fitnesses, basic_best_individual, num_vars, generation + 1)
            neural_population, _ = neural_sr.generate_new_population(neural_population, neural_fitnesses, neural_best_individual, num_vars, generation + 1)

    # Final summary
    print(f"\n=== Final Results ===")
    print(f"Basic SR - Best MSE: {basic_best_mse:.6f}, Expression: {basic_best_individual}")
    print(f"Neural SR - Best MSE: {neural_best_mse:.6f}, Expression: {neural_best_individual}")
    print(f"Neural well-formed rate: {neural_sr.get_well_formed_percentage():.1f}%")

    return {
        'basic_best_mse': basic_best_mse,
        'neural_best_mse': neural_best_mse,
        'basic_best_expr': str(basic_best_individual),
        'neural_best_expr': str(neural_best_individual),
        'neural_well_formed_percentage': neural_sr.get_well_formed_percentage(),
        'generations_completed': generation + 1
    }


def format_individuals_for_display(population):
    """Format individuals for display: sort by length then alphabetically, display on one line"""
    # Convert to strings
    individual_strs = [str(ind) for ind in population]

    # Sort by string length first, then alphabetically
    individual_strs.sort(key=lambda x: (len(x), x))

    # Join with spaces
    return " ".join(individual_strs)


def main():
    parser = argparse.ArgumentParser(description="Fine-grained comparison of neural vs basic SR")
    parser.add_argument("--checkpoint",
                       default="training/checkpoints/onestep-full_20250811_145316/final_model",
                       help="Path to trained model checkpoint")
    parser.add_argument("--problem", type=int, default=0,
                       help="Problem index to test (default: 0)")
    parser.add_argument("--max-generations", type=int, default=1000,
                       help="Maximum number of generations (default: 1000)")
    parser.add_argument("--operator_set", type=str, default="arith", choices=["arith", "full"],
                        help="Operator set: 'arith' (add/sub/mul) or 'full' (all operators)")

    args = parser.parse_args()

    binary_operators, unary_operators = get_operators(args.operator_set)

    result = compare_neural_vs_basic_fine_grained(
        args.checkpoint,
        args.problem,
        max_generations=args.max_generations,
        binary_operators=binary_operators,
        unary_operators=unary_operators
    )
    if result:
        print("\n✓ Fine-grained comparison completed!")
        print(f"Generations completed: {result['generations_completed']}")
        print(f"Neural well-formed rate: {result['neural_well_formed_percentage']:.1f}%")
    else:
        print("\n❌ Fine-grained comparison failed.")


if __name__ == "__main__":
    main()
