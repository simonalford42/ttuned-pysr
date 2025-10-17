"""
Check if loss is being computed correctly on the training data.
"""
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model on {device}")

    # Load some training examples
    with open(args.train_file, 'r') as f:
        examples = [json.loads(line) for line in list(f)[:args.num_examples]]

    print(f"\nAnalyzing {len(examples)} training examples...\n")

    total_loss = 0.0
    total_tokens = 0
    first_token_correct = 0
    first_token_total = 0

    for i, example in enumerate(examples):
        # Format like training
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""

        input_part = bos + "<CONTEXT>" + example['context'] + "<POPULATION>" + example['population']
        target_part = "<TARGET>" + example['target'] + eos

        # Tokenize
        input_tokens = tokenizer(input_part, add_special_tokens=False)
        target_tokens = tokenizer(target_part, add_special_tokens=False)

        full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]
        full_input_ids = torch.tensor([full_input_ids[:args.context_length]]).to(device)

        # Create labels
        labels = [-100] * len(input_tokens["input_ids"])
        labels.extend(target_tokens["input_ids"])
        labels = labels[:args.context_length]
        labels_tensor = torch.tensor([labels]).to(device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(full_input_ids, labels=labels_tensor)
            loss = outputs.loss

        # Compute accuracy on first target token
        target_token_id = tokenizer.encode("<TARGET>", add_special_tokens=False)[0]
        target_positions = (full_input_ids[0] == target_token_id).nonzero(as_tuple=True)[0]

        if len(target_positions) > 0:
            target_pos = target_positions[-1].item()

            # Get prediction for next token
            logits = model(full_input_ids).logits
            next_token_logits = logits[0, target_pos, :]
            predicted_token = torch.argmax(next_token_logits).item()

            # Get expected token (first token of target)
            expected_tokens = tokenizer.encode(example['target'], add_special_tokens=False)
            if expected_tokens:
                expected_token = expected_tokens[0]

                if predicted_token == expected_token:
                    first_token_correct += 1
                first_token_total += 1

                if i < 5:  # Print details for first 5
                    pred_str = tokenizer.decode([predicted_token])
                    exp_str = tokenizer.decode([expected_token])
                    print(f"Example {i+1}:")
                    print(f"  Expected: {example['target']}")
                    print(f"  First token expected: '{exp_str}' (id={expected_token})")
                    print(f"  First token predicted: '{pred_str}' (id={predicted_token})")
                    print(f"  Match: {predicted_token == expected_token}")
                    print(f"  Loss: {loss.item():.4f}")
                    print()

        total_loss += loss.item()
        total_tokens += (labels_tensor != -100).sum().item()

    print(f"{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Average loss: {total_loss / len(examples):.4f}")
    print(f"First token accuracy: {100 * first_token_correct / first_token_total:.2f}% ({first_token_correct}/{first_token_total})")
    print(f"Total training tokens: {total_tokens}")
    print(f"Avg training tokens per example: {total_tokens / len(examples):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="/home/sca63/ttuned-pysr/training/checkpoints/tiny_183867/checkpoint-60000",
                       help="Checkpoint to load")
    parser.add_argument("--train_file", type=str,
                       default="/home/sca63/ttuned-pysr/datasets/training/gen1k_arith_1k_c05_20251003_135753_basic_train.jsonl",
                       help="Training data file")
    parser.add_argument("--num_examples", type=int, default=100,
                       help="Number of examples to check")
    parser.add_argument("--context_length", type=int, default=1024,
                       help="Context length from config")

    args = parser.parse_args()
    main(args)
