"""
Convert BasicSR trajectory data to one-step prediction training format.
"""
import json
import argparse
import random
from pathlib import Path
from format_utils import extract_variables_operators_constants, format_context, format_population_with_fitness, compute_data_statistics
import numpy as np


def convert_basicsr_to_one_step_format(input_file, output_file, context_type='basic', data_context=None):
    """
    Convert BasicSR trajectory JSON to one-step prediction format.
    Each trajectory becomes multiple training examples (one for each generation transition).
    
    Args:
        input_file: Input JSON file with trajectories
        output_file: Output JSONL file  
        context_type: 'basic', 'rich', or 'superrich' context level
        data_context: Dict mapping problem_name -> {'X': X_data, 'y': y_data} for rich context
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    converted_data = []

    # Detect format and extract trajectories
    if "trajectories" in data:
        trajectories = data["trajectories"]

        # Check if it's the new flat array format
        if isinstance(trajectories, list):
            # New improved format: data["trajectories"] = [runs...]
            runs = trajectories
            trajectory_key = "trajectory"
            problem_doc_key = "target_expression"
            final_mse_key = "final_mse"
            run_id_key = "run_id"
            is_flat_format = True
        else:
            # Old format: data["trajectories"][problem_name] = [runs...]
            trajectories_dict = trajectories
            trajectory_key = "trajectory_data"
            problem_doc_key = "problem_doc"
            final_mse_key = "final_mse"
            run_id_key = "run_id"

            # Process each problem separately
            problem_runs = []
            for problem_name, runs in trajectories_dict.items():
                problem_runs.extend([(problem_name, run) for run in runs])
            runs = problem_runs
            is_flat_format = False
    elif isinstance(data, list):
        # Raw flat array format (extracted by dataset_manager)
        runs = data
        trajectory_key = "trajectory"
        problem_doc_key = "target_expression"
        final_mse_key = "final_mse"
        run_id_key = "run_id"
        is_flat_format = True
    else:
        # Legacy format: data[problem_name] = [runs...]
        trajectories_dict = data
        trajectory_key = "trajectory"
        problem_doc_key = "problem_description"
        final_mse_key = "final_mse"
        run_id_key = "run_number"

        # Process each problem separately
        problem_runs = []
        for problem_name, runs_list in trajectories_dict.items():
            problem_runs.extend([(problem_name, run) for run in runs_list])
        runs = problem_runs
        is_flat_format = False

    # Process runs based on format
    def process_run(run, problem_name):
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
            return

        # Extract variables, operators, constants from this trajectory
        variables, operators, constants = extract_variables_operators_constants(trajectory_data)

        # Compute data statistics for rich context if needed
        data_stats = None
        if context_type in ['rich', 'superrich'] and data_context and problem_name in data_context:
            problem_data = data_context[problem_name]
            X, y = problem_data['X'], problem_data['y']
            data_stats = compute_data_statistics(X, y)

            # Add text plot for superrich context
            if context_type == 'superrich':
                from format_utils import create_text_plot
                data_stats['text_plot'] = create_text_plot(X, y)

        # Create one training example for each generation transition
        for i in range(len(trajectory_data) - 1):
            # Create context header using shared formatting with context type and current generation
            context_header = format_context(i, variables, operators, constants, context_type, data_stats)
            current_gen = trajectory_data[i]
            next_gen = trajectory_data[i + 1]

            # Format current generation population using shared formatting
            current_expressions = current_gen["expressions"]
            # Handle cases where fitnesses are not present (simplified training format)
            if "fitnesses" in current_gen:
                current_fitnesses = current_gen["fitnesses"]
            else:
                # Use dummy fitnesses for format compatibility (not used in training)
                current_fitnesses = [-1.0] * len(current_expressions)
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

    # Handle different formats for iteration
    if is_flat_format:
        # Flat format without problem names
        for run in runs:
            process_run(run, "training_data")
    else:
        # Format with problem names
        for problem_name, run in runs:
            process_run(run, problem_name)

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


def create_tiny_training_data(input_file):
    """Convert tiny dataset and create train/val files (duplicate for overfitting test)"""
    print(f"Converting tiny dataset from {input_file}")
    
    # Generate output filenames
    base_name = input_file.replace('.json', '').replace('data/', 'data/tiny_training_')
    temp_file = f"{base_name}.jsonl"
    train_file = f"{base_name}_train.jsonl"
    val_file = f"{base_name}_val.jsonl"
    
    # Convert trajectories
    converted_data = convert_basicsr_to_one_step_format(input_file, temp_file)
    
    # For overfitting test, use same data for train and val
    print(f"Creating train/val files (same data for overfitting test)")
    
    import shutil
    shutil.copy(temp_file, train_file)
    shutil.copy(temp_file, val_file)
    
    # Remove temporary file
    import os
    os.remove(temp_file)
    
    print(f"Tiny training dataset created:")
    print(f"  - Training examples: {len(converted_data)}")
    print(f"  - Train file: {train_file}")
    print(f"  - Val file: {val_file}")
    print(f"  - Both files identical (for overfitting test)")
    
    return train_file, val_file, len(converted_data)


def main():
    parser = argparse.ArgumentParser(description="Convert BasicSR trajectories to training formats")
    parser.add_argument("--input", default="data/harder_problems_all_20250807_163319.json", help="Input file")
    parser.add_argument("--output", default="data/harder_problems_one_step.jsonl", help="Output file (temporary, will be split)")
    parser.add_argument("--context_type", default="basic", choices=["basic", "rich", "superrich"], 
                       help="Context type to use: basic (default), rich (with data stats), or superrich (with plot)")
    parser.add_argument("--data_context", help="Optional JSON file with data context for rich/superrich modes")
    args = parser.parse_args()

    # Load data context if provided
    data_context = None
    if args.data_context and args.context_type in ['rich', 'superrich']:
        with open(args.data_context, 'r') as f:
            data_context = json.load(f)
        print(f"Loaded data context from {args.data_context} for {args.context_type} context")

    # Convert trajectories to temporary file
    converted_data = convert_basicsr_to_one_step_format(args.input, args.output, args.context_type, data_context)
    print(f"Using {args.context_type} context type")

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
