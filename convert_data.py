"""
Convert data to training formats (one-step, autoreg, direct).

Default behavior converts BasicSR trajectory data to one-step prediction format.
Use --autoreg for autoregressive next-generation targets,
or --direct for expression dataset direct prediction.
"""
import json
import argparse
import random
import pickle
import gzip
from pathlib import Path
from format_utils import format_context, format_population_with_fitness, compute_data_statistics
import numpy as np
from tqdm import tqdm
import math
from typing import List, Dict, Any
import datetime
from sr import BasicSR


def convert_basicsr_to_one_step_format(input_file, output_file, context_type='basic', sample_fraction=1.0, ancestors_only=False, expressions_file=None):
    """
    Convert BasicSR trajectory file to one-step prediction format.
    Each trajectory becomes multiple training examples (one for each generation transition).

    Args:
        input_file: Input file with trajectories (.pkl.gz)
        output_file: Output JSONL file
        context_type: 'basic', 'rich', or 'superrich' context level
        sample_fraction: Fraction of data to sample from input file (default 1.0 = all data)
    """
    # Load data from pickle format
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    population_size = data['metadata']['basicsr_params']['population_size']
    if 1/sample_fraction > population_size:
        raise ValueError(f"Sample fraction {sample_fraction} too small for population size {population_size} (we need at least one example per generation). Either increase sample fraction or generate fewer/shorter trajectories.")

    converted_data = []

    # Get source expressions file from metadata (for loading X, y later during training)
    source_expressions_file = data['metadata']['source_expressions_file']
    if expressions_file:
        if source_expressions_file != expressions_file:
            raise ValueError(f"Provided expressions_file {expressions_file} does not match source_expressions_file {source_expressions_file} in trajectory metadata.")

    # Extract trajectories from current format
    trajectories = data['trajectories']

    if expressions_file:
        # load expressions pkl file into dict mapping expr string to expression entry
        with gzip.open(expressions_file, 'rb') as f:
            expressions_data = pickle.load(f)
        expr_dict = {expr['expression']: expr
                     for expr in expressions_data['expressions']}

    # Process each trajectory
    for traj in tqdm(trajectories):
        trajectory_data = traj['trajectory']
        target_expr = traj['target_expression']
        expression_id = traj.get('expression_id', None)
        if expression_id is None:
            # find matching expression from expressions file
            expression_id = expr_dict[target_expr]['id']

        # Extract operators and constants from trajectory
        binary_operators = sorted(traj['binary_operators'])
        unary_operators = sorted(traj.get('unary_operators', []))
        constants = traj['constants']

        X, y = traj['X_data'], traj['y_data']
        num_variables = X.shape[1]

        # Get X_data and y_data for rich/superrich context
        data_stats = None
        if context_type in ['rich', 'superrich']:
            data_stats = compute_data_statistics(X, y)

            # Add text plot for superrich context
            if context_type == 'superrich':
                from format_utils import create_text_plot
                data_stats['text_plot'] = create_text_plot(X, y)

        # Create one training example for each generation transition
        for i in range(len(trajectory_data) - 1):
            # Create context header
            context_header = format_context(i, num_variables, binary_operators + unary_operators, constants, context_type, data_stats)
            current_gen = trajectory_data[i]
            next_gen = trajectory_data[i + 1]

            # Format current generation population
            current_expressions = current_gen["expressions"]
            current_fitnesses = current_gen["fitnesses"]
            # Convert numpy arrays to lists if needed
            if isinstance(current_fitnesses, np.ndarray):
                current_fitnesses = current_fitnesses.tolist()
            population_line = format_population_with_fitness(current_expressions, current_fitnesses)

            # Get target expressions for next generation
            target_expressions = next_gen["expressions"]

            # Create one training example for each target expression
            if sample_fraction < 1.0:
                k = max(1, math.ceil(sample_fraction * population_size))
                expressions = random.sample(target_expressions, k)
            elif ancestors_only:
                ancestor_ixs = next_gen["ancestors_of_best"]
                expressions = [target_expressions[i] for i in ancestor_ixs]
            else:
                expressions = target_expressions

            for j, target in enumerate(expressions):
                example = {
                    "context": context_header,
                    "population": population_line,
                    "target": target,
                    "expression_id": expression_id,  # Store ID reference instead of full X, y data
                    "metadata": {
                        "target_expression": target_expr,
                        "generation": current_gen["generation"],
                        "target_index": j,
                        "source_expressions_file": source_expressions_file  # Add reference to source file
                    }
                }
                converted_data.append(example)

    # Save converted data
    with open(output_file, 'w') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Converted {len(converted_data)} one-step prediction examples to {output_file}")
    return converted_data


def convert_basicsr_to_autoreg_format(input_file, output_file, context_type='basic', sample_fraction=1.0):
    """
    Convert BasicSR trajectory file to autoregressive format.
    Each trajectory becomes multiple training examples (one for each generation transition).
    Each example has: context + current population -> entire next population (autoregressive completion)

    Args:
        input_file: Input file with trajectories (.pkl.gz)
        output_file: Output JSONL file
        context_type: 'basic', 'rich', or 'superrich' context level
        sample_fraction: Fraction of trajectories to sample from input file (default 1.0 = all data)
    """
    # Load data from pickle format
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    converted_data = []

    # Extract trajectories from current format
    trajectories = data['trajectories']

    # Sample trajectories if requested
    if sample_fraction < 1.0:
        num_trajectories = max(1, int(len(trajectories) * sample_fraction))
        trajectories = random.sample(trajectories, num_trajectories)
        print(f"Sampling {num_trajectories}/{len(data['trajectories'])} trajectories")

    # Process each trajectory
    for traj in tqdm(trajectories):
        trajectory_data = traj['trajectory']
        target_expr = traj['target_expression']

        # Extract operators and constants from trajectory
        binary_operators = sorted(traj['binary_operators'])
        unary_operators = sorted(traj.get('unary_operators', []))
        constants = traj['constants']

        X, y = traj['X_data'], traj['y_data']
        num_variables = X.shape[1]

        # Get X_data and y_data for rich/superrich context
        data_stats = None
        if context_type in ['rich', 'superrich']:
            data_stats = compute_data_statistics(X, y)

            # Add text plot for superrich context
            if context_type == 'superrich':
                from format_utils import create_text_plot
                data_stats['text_plot'] = create_text_plot(X, y)

        # Create one training example for each generation transition
        for i in range(len(trajectory_data) - 1):
            # Create context header
            context_header = format_context(i, num_variables, binary_operators + unary_operators, constants, context_type, data_stats)
            current_gen = trajectory_data[i]
            next_gen = trajectory_data[i + 1]

            # Format current generation population
            current_expressions = current_gen["expressions"]
            current_fitnesses = current_gen["fitnesses"]
            # Convert numpy arrays to lists if needed
            if isinstance(current_fitnesses, np.ndarray):
                current_fitnesses = current_fitnesses.tolist()
            population_line = format_population_with_fitness(current_expressions, current_fitnesses)

            # Get target expressions for next generation (all expressions as one autoregressive target)
            target_expressions = next_gen["expressions"]
            target_fitnesses = next_gen["fitnesses"]
            if isinstance(target_fitnesses, np.ndarray):
                target_fitnesses = target_fitnesses.tolist()

            # Format the entire next population as the target
            target_population = ' '.join(target_expressions)

            example = {
                "context": context_header,
                "population": population_line,
                "target": target_population,
                "metadata": {
                    "target_expression": target_expr,
                    "generation": current_gen["generation"],
                }
            }
            converted_data.append(example)

    # Save converted data
    with open(output_file, 'w') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Converted {len(converted_data)} autoregressive examples to {output_file}")
    return converted_data


def convert_expressions_to_direct_format(input_file, output_file):
    """
    Convert expressions file to direct prediction format.
    Each expression becomes one training example.

    Args:
        input_file: Input file with expressions (.pkl.gz)
        output_file: Output JSONL file
    """
    # Load data from pickle format
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    converted_data = []

    # Extract expressions
    expressions = data['expressions']

    # Process each expression
    for expr in tqdm(expressions, desc="Converting expressions"):
        # Get expression data
        expr_id = expr['id']
        expression_str = expr['expression']
        X_data = expr['X_data']  # numpy array (n_points, n_vars)
        y_data = expr['y_data']  # numpy array (n_points,)

        # Create training example
        # Store X, y data directly as lists for JSON serialization
        example = {
            "X_data": X_data.tolist(),
            "y_data": y_data.tolist(),
            "target": expression_str,
            "metadata": {
                "expression_id": expr_id,
                "source_expressions_file": str(input_file),
                "n_variables": X_data.shape[1],
                "n_points": X_data.shape[0]
            }
        }
        converted_data.append(example)

    # Save converted data
    with open(output_file, 'w') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Converted {len(converted_data)} direct prediction examples to {output_file}")
    return converted_data


def split_train_val(input_file, train_file, val_file, val_split=0.1, seed=42):
    """
    Split training data into train/validation sets.

    Args:
        input_file: Input JSONL file with training data
        train_file: Output file for training data
        val_file: Output file for validation data
        val_split: Fraction of data to use for validation (default 0.1 = 10%)
        seed: Random seed for reproducible splits
    """
    print(f"Splitting {input_file} into train/val sets...")

    # Read all data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Shuffle data
    random.seed(seed)
    random.shuffle(data)

    # Split
    val_size = int(len(data) * val_split)
    val_data = data[:val_size]
    train_data = data[val_size:]

    # Save splits
    with open(train_file, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')

    with open(val_file, 'w') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Split complete:")
    print(f"  - Training: {len(train_data)} examples -> {train_file}")
    print(f"  - Validation: {len(val_data)} examples -> {val_file}")
    print(f"  - Val split: {val_split:.1%}")

    return len(train_data), len(val_data)


def convert_and_make_split(input_file, context_type, sample_fraction, split, ancestors_only, expressions_file, mode):
    # Generate output filename from input filename
    input_path = Path(input_file)
    input_filename = input_path.stem  # Remove extension
    if input_filename.endswith('.pkl'):  # Handle .pkl.gz case
        input_filename = input_filename[:-4]

    # Create output directory and filename
    output_dir = Path("datasets/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    frac_str = f"_{int(sample_fraction*100)}" if sample_fraction < 1.0 else ""
    anc_str = "_ancs" if ancestors_only else ""

    if mode == 'autoreg':
        output_filename = f"{input_filename}_autoreg{frac_str}.jsonl"
        output_file = output_dir / output_filename
        convert_basicsr_to_autoreg_format(input_file, str(output_file), context_type, sample_fraction)
    elif mode == 'direct':
        output_filename = f"{input_filename}_direct.jsonl"
        output_file = output_dir / output_filename
        convert_expressions_to_direct_format(input_file, str(output_file))
    else:
        output_filename = f"{input_filename}{frac_str}{anc_str}.jsonl"
        output_file = output_dir / output_filename
        # Convert trajectories to one-step format
        convert_basicsr_to_one_step_format(input_file, str(output_file), context_type, sample_fraction, ancestors_only, expressions_file)

    if split:
        base_name = str(output_file).replace('.jsonl', '')
        train_file = f"{base_name}_train.jsonl"
        val_file = f"{base_name}_val.jsonl"

        split_train_val(str(output_file), train_file, val_file, val_split=0.1, seed=42)

"""
DAgger Relabeling

Takes neural SR traces and adds expert (BasicSR) actions as 'dagger_expressions' field.
This creates DAgger-style training data where the neural model's rollout states
are paired with the expert's optimal actions.

The script modifies the input file in-place, adding a 'dagger_expressions' field to each
generation in each trajectory. The neural population remains in 'expressions' field.

Usage:
    python convert_data.py --input datasets/traces/neural_traces.pkl.gz --dagger_relabel
"""

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
    parser = argparse.ArgumentParser(description="Convert datasets to training formats: one-step (default), autoreg (--autoreg), or direct (--direct)")
    parser.add_argument("--input", required=True, help="Input file: trace (.pkl.gz) for one-step/autoreg, expressions (.pkl.gz) for direct")
    parser.add_argument("--expressions_file", default=None, help="Expressions file to match expression IDs to (required for one-step if IDs missing)")
    parser.add_argument("--context_type", default="basic", choices=["basic", "rich", "superrich"],
                       help="Context type to use for one-step/autoreg: basic (default), rich (with data stats), or superrich (with plot)")
    parser.add_argument("--split", action="store_true", help="If set, split into train/val sets")
    parser.add_argument("--ancestors_only", action="store_true", help="Only include ancestors as training targets (one-step mode)")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of data to sample from input file (default 1.0 = all data). For one-step: per-generation targets; for autoreg: trajectories")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--autoreg", action="store_true", help="Use autoregressive next-generation targets (input must be trajectory file)")
    mode_group.add_argument("--direct", action="store_true", help="Convert expressions to direct prediction format (input must be expressions file)")
    mode_group.add_argument("--dagger_relabel", action="store_true", help="Dagger relabelling")

    args = parser.parse_args()

    if args.dagger_relabel:
        add_expert_labels_to_traces(
            args.input,
            seed=args.seed,
        )

    mode = 'autoreg' if args.autoreg else ('direct' if args.direct else 'one_step')
    convert_and_make_split(
        args.input,
        args.context_type,
        sample_fraction=args.sample_fraction,
        split=args.split,
        ancestors_only=args.ancestors_only,
        expressions_file=args.expressions_file,
        mode=mode,
    )


if __name__ == "__main__":
    main()
