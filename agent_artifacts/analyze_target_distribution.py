"""
Analyze the distribution of first tokens in targets to understand training data bias.
"""
import json
import argparse
from collections import Counter

def main(args):
    print(f"Analyzing {args.train_file}...")

    first_chars = Counter()
    first_tokens = Counter()
    total = 0

    with open(args.train_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            target = example['target']

            if target:
                first_char = target[0]
                first_chars[first_char] += 1

                # Identify first "semantic" token
                if target.startswith('(('):
                    first_tokens['(('] += 1
                elif target.startswith('('):
                    first_tokens['('] += 1
                elif target.startswith('x'):
                    # Find full variable name
                    i = 1
                    while i < len(target) and target[i].isdigit():
                        i += 1
                    first_tokens[target[:i]] += 1
                elif target[0].isdigit() or target[0] == '.':
                    # Number
                    first_tokens['NUMBER'] += 1
                else:
                    first_tokens[first_char] += 1

            total += 1

            if total >= args.max_examples:
                break

    print(f"\nAnalyzed {total} examples")
    print(f"\n{'='*60}")
    print("First CHARACTER distribution:")
    print(f"{'='*60}")
    for char, count in first_chars.most_common(20):
        pct = 100 * count / total
        print(f"  '{char}': {count:8d} ({pct:5.2f}%)")

    print(f"\n{'='*60}")
    print("First TOKEN distribution:")
    print(f"{'='*60}")
    for token, count in first_tokens.most_common(20):
        pct = 100 * count / total
        print(f"  '{token}': {count:8d} ({pct:5.2f}%)")

    # Compute simple vs complex expressions
    simple_count = sum(count for token, count in first_tokens.items() if token.startswith('x') or token == 'NUMBER')
    complex_count = sum(count for token, count in first_tokens.items() if token in ['(', '(('])

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"  Simple (starts with variable/number): {simple_count:8d} ({100*simple_count/total:5.2f}%)")
    print(f"  Complex (starts with parenthesis):    {complex_count:8d} ({100*complex_count/total:5.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str,
                       default="/home/sca63/ttuned-pysr/datasets/training/gen1k_arith_1k_c05_20251003_135753_basic_train.jsonl",
                       help="Training data file")
    parser.add_argument("--max_examples", type=int, default=1000000,
                       help="Maximum examples to analyze")

    args = parser.parse_args()
    main(args)
