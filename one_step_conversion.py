"""
Convert BasicSR trajectory data to one-step prediction training format.
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


def convert_basicsr_to_one_step_format(input_file, output_file, context_type='basic'):
    """
    Convert BasicSR trajectory file to one-step prediction format.
    Each trajectory becomes multiple training examples (one for each generation transition).

    Args:
        input_file: Input file with trajectories (.pkl.gz)
        output_file: Output JSONL file
        context_type: 'basic', 'rich', or 'superrich' context level
    """
    # Load data from pickle format
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    converted_data = []

    # Extract trajectories from current format
    trajectories = data['trajectories']

    # Process each trajectory
    for traj in tqdm(trajectories):
        trajectory_data = traj['trajectory']
        target_expr = traj['target_expression']

        # Extract operators and constants from trajectory
        binary_operators = sorted(traj['binary_operators'])
        unary_operators = sorted(traj.get('unary_operators', []))
        constants = sorted([str(c) for c in traj['constants']])

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
            for j, target in enumerate(target_expressions):
                example = {
                    "context": context_header,
                    "population": population_line,
                    "target": target,
                    "metadata": {
                        "target_expression": target_expr,
                        "generation": current_gen["generation"],
                        "target_index": j
                    }
                }
                converted_data.append(example)

    # Save converted data
    with open(output_file, 'w') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Converted {len(converted_data)} one-step prediction examples to {output_file}")
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


def convert_and_make_split(input_file, context_type):
    # Generate output filename from input filename
    input_path = Path(input_file)
    input_filename = input_path.stem  # Remove extension
    if input_filename.endswith('.pkl'):  # Handle .pkl.gz case
        input_filename = input_filename[:-4]

    # Create output directory and filename
    output_dir = Path("datasets/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add context_type to the filename
    output_filename = f"{input_filename}_{context_type}.jsonl"
    output_file = output_dir / output_filename

    # Convert trajectories to temporary file
    converted_data = convert_basicsr_to_one_step_format(input_file, str(output_file), context_type)
    print(f"Using {context_type} context type")

    # Generate train/val file names
    base_name = str(output_file).replace('.jsonl', '')
    train_file = f"{base_name}_train.jsonl"
    val_file = f"{base_name}_val.jsonl"

    # Split into train/val (10% validation)
    split_train_val(str(output_file), train_file, val_file, val_split=0.1, seed=42)

    # Remove temporary file
    import os
    os.remove(output_file)
    print(f"Removed temporary file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert BasicSR trajectories to training formats")
    parser.add_argument("--input", required=True, help="Input trace file (.pkl.gz)")
    parser.add_argument("--context_type", default="basic", choices=["basic", "rich", "superrich"],
                       help="Context type to use: basic (default), rich (with data stats), or superrich (with plot)")
    args = parser.parse_args()
    convert_and_make_split(args.input, args.context_type)


if __name__ == "__main__":
    main()
