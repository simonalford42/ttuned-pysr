"""
Convert BasicSR trajectory data to one-step prediction training format.
"""
import json
import argparse
import random
from pathlib import Path
from format_utils import extract_variables_operators_constants, format_context, format_population_with_fitness


def convert_basicsr_to_one_step_format(input_file, output_file):
    """
    Convert BasicSR trajectory JSON to one-step prediction format.
    Each trajectory becomes multiple training examples (one for each generation transition).
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    converted_data = []

    # Detect format and extract trajectories
    if "trajectories" in data:
        # Old format: data["trajectories"][problem_name] = [runs...]
        trajectories_dict = data["trajectories"]
        trajectory_key = "trajectory_data"
        problem_doc_key = "problem_doc"
        final_mse_key = "final_mse"
        run_id_key = "run_id"
    else:
        # New format: data[problem_name] = [runs...]
        trajectories_dict = data
        trajectory_key = "trajectory"
        problem_doc_key = "problem_description"
        final_mse_key = "final_mse"
        run_id_key = "run_number"

    for problem_name, runs in trajectories_dict.items():
        for run in runs:
            # Handle different metadata structures
            if "metadata" in run:
                # New format: metadata is separate
                metadata = run["metadata"]
                trajectory_data = run[trajectory_key]
                run_id = metadata.get(run_id_key, 0)
            else:
                # Old format: everything is at top level
                trajectory_data = run[trajectory_key]
                run_id = run.get(run_id_key, 0)

            # Skip if trajectory doesn't have new format with full population data
            if not trajectory_data or "expressions" not in trajectory_data[0]:
                continue

            # Extract variables, operators, constants from this trajectory
            variables, operators, constants = extract_variables_operators_constants(trajectory_data)

            # Create context header using shared formatting
            context_header = format_context(variables, operators, constants)

            # Create one training example for each generation transition
            for i in range(len(trajectory_data) - 1):
                current_gen = trajectory_data[i]
                next_gen = trajectory_data[i + 1]

                # Format current generation population using shared formatting
                current_expressions = current_gen["expressions"]
                current_fitnesses = current_gen["fitnesses"]
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
                            "problem": problem_name,
                            "generation": current_gen["generation"],
                            "run_id": run_id,
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


def main():
    parser = argparse.ArgumentParser(description="Convert BasicSR trajectories to training formats")
    parser.add_argument("--input", default="data/harder_problems_all_20250807_163319.json", help="Input file")
    parser.add_argument("--output", default="data/harder_problems_one_step.jsonl", help="Output file (temporary, will be split)")
    args = parser.parse_args()

    # Convert trajectories to temporary file
    converted_data = convert_basicsr_to_one_step_format(args.input, args.output)

    # Generate train/val file names based on output file
    base_name = args.output.replace('.jsonl', '')
    train_file = f"{base_name}_train.jsonl"
    val_file = f"{base_name}_val.jsonl"

    # Split into train/val (10% validation)
    split_train_val(args.output, train_file, val_file, val_split=0.1, seed=42)

    # Remove temporary file
    import os
    os.remove(args.output)
    print(f"Removed temporary file: {args.output}")


if __name__ == "__main__":
    main()
