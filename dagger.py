from generate_traces import generate_traces_from_expressions
from convert_data import convert_and_make_split, add_expert_labels_to_traces
import json
import random
import train
from utils import load_jsonl, save_jsonl, Timing
import shutil
import argparse

def run_dagger(
    num_iterations: int,
    expressions_file: str,
    num_expressions: int,
    num_generations: int,
    original_dataset: str,
    seed: int = 42,
    checkpoint: str = None,
):

    # initialize dataset to original_dataset
    shutil.copyfile(original_dataset, "datasets/training/dagger_combined.jsonl")

    # train new network from scratch if no checkpoint provided
    if checkpoint is None:
        print("No checkpoint provided, training initial model from scratch.")
        with Timing("Trained neural network"):
            checkpoint = train.main(
                config="configs/dagger.json",
            )

    for i in range(num_iterations):
        print(f"=== DAgger Iteration {i+1}/{num_iterations} ===")
        with Timing(f"Completed DAgger iteration {i+1}"):

            # Generate traces with current model
            traces_file = f"datasets/traces/traces_dagger_iter_{i+1}.pkl.gz"
            basicsr_params = {
                'population_size': 20,
                'num_generations': num_generations,
                'max_depth': 10,
                'max_size': 25,
                'tournament_size': 3,
            }

            with Timing("Generated neural traces"):
                generate_traces_from_expressions(
                    expressions_file=expressions_file,
                    max_expressions=num_expressions,
                    basicsr_params=basicsr_params,
                    operator_set="arith",
                    output_file=traces_file,
                    checkpoint=checkpoint,
                    batch_size=1,
                    seed=seed,
                )
            add_expert_labels_to_traces(traces_file)

            # create one step dataset from new traces
            new_dataset = convert_and_make_split(
                input_file=traces_file,
                dagger=True,
            )

            combine_datasets(
                old_data_path="datasets/training/dagger_combined.jsonl",
                new_data_path=new_dataset,
                filename="datasets/training/dagger_combined.jsonl",
            )

            # for debugging, save dataset at that step to a separate file
            shutil.copyfile(
                "datasets/training/dagger_combined.jsonl",
                f"datasets/training/dagger_combined_iter_{i+1}.jsonl",
            )

            with Timing("Trained neural network"):
                # continue training model on new dataset
                checkpoint = train.main(
                    config_path="configs/dagger.json",
                    reset=True,
                    checkpoint=checkpoint,
                )

def combine_datasets(old_data_path, new_data_path, filename):
    # combine datasets
    old_data = load_jsonl(old_data_path)
    new_data = load_jsonl(new_data_path)

    needed_size = len(old_data) - len(new_data)
    if needed_size > 0:
        additional_data = random.sample(old_data, needed_size)
        new_data.extend(additional_data)
        print(f"Combined datasets: old size {len(old_data)}, new size {len(new_data)}, added {needed_size} samples from old data.")

    save_jsonl(new_data, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DAgger loop to iteratively generate traces and fine-tune the model.")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of DAgger iterations to run.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to start from.")
    parser.add_argument("--expressions_file", type=str, default="datasets/expressions/arith_1k_c05_20251016_214231.pkl.gz", help="Path to the expressions file used to generate traces.")
    parser.add_argument("--num_expressions", type=int, default=100, help="Number of expressions to generate per iteration.")
    parser.add_argument("--num_generations", type=int, default=100, help="Number of generations for the trace generator.")
    parser.add_argument("--original_dataset", type=str, required=True, help="Path to the original dataset jsonl file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    run_dagger(
        num_iterations=args.num_iterations,
        checkpoint=args.checkpoint,
        expressions_file=args.expressions_file,
        num_expressions=args.num_expressions,
        num_generations=args.num_generations,
        original_dataset=args.original_dataset,
        seed=args.seed,
    )


