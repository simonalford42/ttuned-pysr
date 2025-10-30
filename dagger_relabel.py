"""
DAgger Relabeling Script

Takes neural SR traces and adds expert (BasicSR) actions as 'dagger_expressions' field.
This creates DAgger-style training data where the neural model's rollout states
are paired with the expert's optimal actions.

The script modifies the input file in-place, adding a 'dagger_expressions' field to each
generation in each trajectory. The neural population remains in 'expressions' field.

Usage:
    python dagger_relabel.py --input datasets/traces/neural_traces.pkl.gz
"""

import argparse
import pickle
import gzip
import numpy as np
import random
from typing import List, Dict, Any
from sr import BasicSR
from datetime import datetime


def add_expert_labels_to_trajectory(trajectory: Dict[str, Any],
                                   basicsr_params: Dict[str, Any],
                                   binary_operators: List[str],
                                   unary_operators: List[str],
                                   constants: List[float],
                                   seed: int = 42) -> None:
    """
    Add expert actions to a trajectory as 'dagger_expressions' field.

    Takes the population states from neural SR and computes what BasicSR
    would do at each generation, adding the expert's next population as
    'dagger_expressions' field in each generation.

    Args:
        trajectory: Neural SR trajectory (modified in-place)
        basicsr_params: BasicSR parameters
        binary_operators: Binary operators to use
        unary_operators: Unary operators to use
        constants: Constants to use
        seed: Random seed for reproducibility
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Get X, y data
    X = trajectory['X_data']
    y = trajectory['y_data']

    # Create expert model
    expert = BasicSR(
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        constants=constants,
        **basicsr_params
    )

    # Parse expressions from the neural trajectory
    from expression_parser import ExpressionParser
    parser = ExpressionParser()

    # Get initial population from first generation
    first_gen = trajectory['trajectory'][0]
    population = [parser.parse(expr) for expr in first_gen['expressions']]

    # Track best individual
    best_individual = None
    best_fitness = -float('inf')

    num_vars = X.shape[1]

    # Process each generation and add expert labels
    for gen_idx, gen_data in enumerate(trajectory['trajectory']):
        # Evaluate fitness for current population
        fitnesses = np.array([expert.fitness(ind, X, y) for ind in population])

        # Update best
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx].copy()

        # Generate next population using expert
        new_population, heritages = expert.generate_new_population(
            population, fitnesses, best_individual, num_vars, gen_idx
        )

        # Add expert's population to next generation
        if gen_idx + 1 < len(trajectory['trajectory']):
            trajectory['trajectory'][gen_idx + 1]['dagger_expressions'] = [str(ind) for ind in new_population]
            # print(f"New population: {' '.join([str(e) for e in new_population])}")

        # Update population for next iteration
        population = new_population

    # Mark trajectory as having DAgger labels
    trajectory['dagger'] = True


def add_expert_labels_to_traces(input_file: str, seed: int = 42) -> str:
    """
    Add expert actions to all traces in a file (modifies in-place).

    Args:
        input_file: Path to traces file (neural SR traces) - will be modified in-place
        seed: Base random seed

    Returns:
        Path to modified file
    """
    print(f"=== DAgger Relabeling (In-Place) ===")
    print(f"Input file: {input_file}")

    # Load input traces
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    trajectories = data['trajectories']
    metadata = data['metadata']

    print(f"Loaded {len(trajectories)} trajectories")

    # Get operators and constants from metadata (or first trajectory)
    binary_operators = metadata.get('binary_operators')
    unary_operators = metadata.get('unary_operators')
    constants = metadata.get('constants')

    if binary_operators is None:
        # Fall back to first trajectory
        binary_operators = trajectories[0]['binary_operators']
        unary_operators = trajectories[0]['unary_operators']
        constants = trajectories[0]['constants']

    # Get BasicSR params from metadata (or first trajectory)
    basicsr_params = metadata.get('basicsr_params')
    if basicsr_params is None:
        basicsr_params = trajectories[0]['basicsr_params']

    print(f"BasicSR params: {basicsr_params}")
    print(f"Operators: binary={binary_operators}, unary={unary_operators}")
    print(f"Constants: {constants}")

    # Add expert labels to each trajectory (modifies in-place)
    for i, traj in enumerate(trajectories):
        if (i + 1) % 10 == 0:
            print(f"Processing trajectory {i+1}/{len(trajectories)}")

        # Use different seed for each trajectory
        traj_seed = seed + i

        add_expert_labels_to_trajectory(
            traj, basicsr_params, binary_operators, unary_operators, constants, traj_seed
        )

    # Update metadata to indicate DAgger labels have been added
    metadata['dagger'] = True
    metadata['dagger_relabel_timestamp'] = datetime.now().isoformat()

    # Save back to same file
    with gzip.open(input_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n=== Relabeling Complete ===")
    print(f"Added 'dagger_expressions' field to {len(trajectories)} trajectories")
    print(f"File updated: {input_file}")

    return input_file


def main():
    parser = argparse.ArgumentParser(description="Add expert actions to neural SR traces for DAgger training (modifies file in-place)")
    parser.add_argument("--input", required=True, help="Input traces file (neural SR traces) - will be modified in-place")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    output_file = add_expert_labels_to_traces(
        args.input,
        seed=args.seed,
    )

    print(f"\n✓ DAgger relabeling complete: {output_file}")


if __name__ == "__main__":
    main()
