"""
Diagnostic script to inspect training data and model tokenization.
Helps identify issues with training setup.
"""
import json
import argparse
from transformers import AutoTokenizer

def main(args):
    # Load tokenizer
    if args.checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        # Add special tokens like in training
        special_tokens = {
            "additional_special_tokens": [
                "<CONTEXT>", "<POPULATION>", "<FITNESS>", "<TARGET>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # Check special tokens
    print("\nSpecial tokens:")
    for tok in ["<CONTEXT>", "<POPULATION>", "<FITNESS>", "<TARGET>"]:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)
        print(f"  {tok}: {tok_id}")

    # Load and inspect training data
    print(f"\n{'='*80}\nInspecting training file: {args.train_file}\n{'='*80}")

    with open(args.train_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.num_examples:
                break

            example = json.loads(line)
            print(f"\n--- Example {i+1} ---")
            print(f"Context: {example['context'][:100]}...")
            print(f"Population (first 200 chars): {example['population'][:200]}...")
            print(f"Target: {example['target']}")

            # Simulate training tokenization
            bos = tokenizer.bos_token or ""
            eos = tokenizer.eos_token or ""

            input_part = bos + "<CONTEXT>" + example['context'] + "<POPULATION>" + example['population']
            target_part = "<TARGET>" + example['target'] + eos

            # Tokenize
            input_tokens = tokenizer(input_part, add_special_tokens=False)
            target_tokens = tokenizer(target_part, add_special_tokens=False)

            full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]

            print(f"\nTokenization stats:")
            print(f"  Input part tokens: {len(input_tokens['input_ids'])}")
            print(f"  Target part tokens: {len(target_tokens['input_ids'])}")
            print(f"  Total tokens: {len(full_input_ids)}")
            print(f"  Context length limit: {args.context_length}")

            if len(full_input_ids) > args.context_length:
                print(f"  ⚠️  TRUNCATED! Losing {len(full_input_ids) - args.context_length} tokens")

            # Decode the target to see what it looks like
            print(f"\nTarget tokens: {target_tokens['input_ids'][:20]}")
            print(f"Target decoded: {tokenizer.decode(target_tokens['input_ids'])}")

            # Check label masking
            labels = [-100] * len(input_tokens["input_ids"])
            labels.extend(target_tokens["input_ids"])
            print(f"\nLabel stats:")
            print(f"  Total labels: {len(labels)}")
            print(f"  Masked labels (-100): {labels.count(-100)}")
            print(f"  Training labels: {len(labels) - labels.count(-100)}")
            print(f"  Training ratio: {(len(labels) - labels.count(-100)) / len(labels):.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str,
                       default="datasets/training/gen1k_arith_1k_c05_20251003_135753_basic_train.jsonl",
                       help="Training data file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint to load tokenizer from")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of examples to inspect")
    parser.add_argument("--context_length", type=int, default=1024,
                       help="Context length from config")

    args = parser.parse_args()
    main(args)
