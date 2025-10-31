"""
Convert expression datasets to direct prediction training format.
For direct baseline: predict target expression directly from input embeddings.
"""
import json
import argparse
import random
import pickle
import gzip
from pathlib import Path
import numpy as np
from tqdm import tqdm


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


def convert_and_make_split(input_file, split):
    # Generate output filename from input filename
    input_path = Path(input_file)
    input_filename = input_path.stem  # Remove extension
    if input_filename.endswith('.pkl'):  # Handle .pkl.gz case
        input_filename = input_filename[:-4]

    # Create output directory and filename
    output_dir = Path("datasets/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{input_filename}_direct.jsonl"
    output_file = output_dir / output_filename

    # Convert expressions to direct format
    convert_expressions_to_direct_format(input_file, str(output_file))

    if split:
        base_name = str(output_file).replace('.jsonl', '')
        train_file = f"{base_name}_train.jsonl"
        val_file = f"{base_name}_val.jsonl"

        split_train_val(str(output_file), train_file, val_file, val_split=0.1, seed=42)


def main():
    parser = argparse.ArgumentParser(description="Convert expressions to direct prediction format")
    parser.add_argument("--input", required=True, help="Input expressions file (.pkl.gz)")
    parser.add_argument("--split", action="store_true", help="If set, split into train/val sets")
    args = parser.parse_args()
    convert_and_make_split(args.input, args.split)


if __name__ == "__main__":
    main()
