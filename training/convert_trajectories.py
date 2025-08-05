"""
Convert MinimalSR trajectory data to stream-of-search training format.
"""
import json
import argparse
from pathlib import Path


def convert_trajectory_to_search_path(trajectory_data):
    """
    Convert a single trajectory to search path format.
    
    Args:
        trajectory_data: List of generation data with populations and fitnesses
    
    Returns:
        search_path: List of strings representing the search trajectory
    """
    search_path = []
    
    for gen_data in trajectory_data:
        generation = gen_data["generation"]
        expressions = gen_data["expressions"]
        fitnesses = gen_data["fitnesses"]
        
        # Format: Generation X: [expr1 (fitness1), expr2 (fitness2), ...]
        expr_fitness_pairs = [f"{expr} ({fitness:.6f})" for expr, fitness in zip(expressions, fitnesses)]
        gen_text = f"Generation {generation}: [{', '.join(expr_fitness_pairs)}]"
        search_path.append(gen_text)
    
    return search_path


def convert_minimalsr_to_sos_format(input_file, output_file):
    """
    Convert MinimalSR trajectory JSON to stream-of-search format.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    
    for problem_name, runs in data["trajectories"].items():
        for run in runs:
            # Convert trajectory to search path
            search_path = convert_trajectory_to_search_path(run["trajectory_data"])
            
            # Create entry in stream-of-search format
            entry = {
                "problem": problem_name,
                "target": run.get("problem_doc", "Unknown target"),
                "search_path": search_path,
                "rating": [1.0] * len(search_path),  # Placeholder ratings
                "final_mse": run.get("final_mse", float('inf')),
                "run_id": run.get("run_id", 0)
            }
            converted_data.append(entry)
    
    # Save converted data
    with open(output_file, 'w') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Converted {len(converted_data)} trajectories to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert MinimalSR trajectories to stream-of-search format")
    parser.add_argument("--input", required=True, help="Input MinimalSR trajectory JSON file")
    parser.add_argument("--output", required=True, help="Output file for stream-of-search format")
    
    args = parser.parse_args()
    
    convert_minimalsr_to_sos_format(args.input, args.output)


if __name__ == "__main__":
    main()