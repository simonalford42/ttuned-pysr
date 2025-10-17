"""
Test script to examine model generation behavior in detail.
"""
import json
import argparse
import torch
import sys
sys.path.append('..')
from transformers import AutoTokenizer, AutoModelForCausalLM

def format_inference_input(bos_token, context, population):
    """Format input for model inference (includes TARGET prompt)"""
    bos = bos_token or ""
    return (
        bos +
        "<CONTEXT>" + context +
        "<POPULATION>" + population +
        "<TARGET>"
    )

def main(args):
    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model loaded on {device}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load validation data
    with open(args.val_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"\nLoaded {len(examples)} validation examples")
    print(f"\nTesting {args.num_examples} examples...\n")

    for i in range(min(args.num_examples, len(examples))):
        example = examples[i]

        print(f"\n{'='*80}")
        print(f"Example {i+1}")
        print(f"{'='*80}")
        print(f"Context: {example['context']}")
        print(f"Population (first 150 chars): {example['population'][:150]}...")
        print(f"Expected target: {example['target']}")

        # Create input
        input_text = format_inference_input(tokenizer.bos_token, example['context'], example['population'])

        # Tokenize
        input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(device)

        print(f"\nInput length: {input_ids.shape[1]} tokens")

        # Check what token comes after <TARGET>
        target_token_id = tokenizer.encode("<TARGET>", add_special_tokens=False)[0]
        target_pos = (input_ids[0] == target_token_id).nonzero(as_tuple=True)[0]
        if len(target_pos) > 0:
            target_pos = target_pos[-1].item()
            print(f"<TARGET> token found at position {target_pos}")

        # Generate with different settings
        print(f"\n--- Greedy decoding ---")
        with torch.no_grad():
            output_greedy = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_greedy = tokenizer.decode(output_greedy[0][input_ids.shape[1]:], skip_special_tokens=False)
        print(f"Generated (greedy): {generated_greedy}")

        # Check logits at the <TARGET> position
        print(f"\n--- Logit analysis at <TARGET> position ---")
        with torch.no_grad():
            logits = model(input_ids).logits

        # Get top-k predictions for the token after <TARGET>
        next_token_logits = logits[0, target_pos, :]
        top_k = 20
        top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), k=top_k)

        print(f"Top {top_k} predictions for next token after <TARGET>:")
        for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode([idx.item()])
            print(f"  {j+1}. '{token}' (id={idx.item()}, prob={prob.item():.4f})")

        # Check what the expected first token should be
        expected_target = example['target']
        expected_tokens = tokenizer.encode(expected_target, add_special_tokens=False)
        if expected_tokens:
            expected_first = expected_tokens[0]
            expected_first_token = tokenizer.decode([expected_first])
            print(f"\nExpected first token: '{expected_first_token}' (id={expected_first})")

            # Find rank of expected token
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            rank = (sorted_indices == expected_first).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                rank = rank[0].item() + 1
                prob = torch.softmax(next_token_logits, dim=-1)[expected_first].item()
                print(f"Expected token rank: {rank} (prob={prob:.6f})")
            else:
                print(f"Expected token not found in vocabulary!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="/home/sca63/ttuned-pysr/training/checkpoints/tiny_183867/checkpoint-60000",
                       help="Checkpoint to load")
    parser.add_argument("--val_file", type=str,
                       default="/home/sca63/ttuned-pysr/datasets/training/gen1k_arith_100_c05_20251003_121116_basic.jsonl",
                       help="Validation data file")
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of examples to test")

    args = parser.parse_args()
    main(args)
