"""
Quick audit of one-step SR datasets.

Usage:
  python audit_onestep_dataset.py --files <jsonl> [<jsonl> ...] [--context_length 1024] [--tokenizer <path_or_model>]

Prints:
- Basic counts and consistency of the context header (especially constants field)
- Target first-token distribution and fraction of simple-variable targets
- Optional tokenized length stats (requires tokenizer)
"""
import argparse
import json
import re
from collections import Counter
from typing import List, Optional


def audit_jsonl(path: str, sample: Optional[int] = None):
    n = 0
    constants_counter = Counter()
    first_tok = Counter()
    simple_var = 0

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if sample is not None and i >= sample:
                break
            if not line.strip():
                continue
            ex = json.loads(line)

            # Extract constants part from context header
            m = re.search(r"constants:\s*([^|]+)", ex.get('context', ''))
            if m:
                constants_counter[m.group(1).strip()] += 1

            # Inspect target shape
            t = ex.get('target', '').strip()
            if re.fullmatch(r"x\d+", t):
                simple_var += 1
            if t.startswith('('):
                first_tok['('] += 1
            else:
                tok = t.split()[0] if t else ''
                if tok:
                    first_tok[tok] += 1
            n += 1

    return {
        'n': n,
        'constants_top': constants_counter.most_common(8),
        'first_tok_top': first_tok.most_common(12),
        'simple_var_frac': (simple_var / n) if n else 0.0,
    }


def tokenize_length_stats(paths: List[str], context_length: int, tokenizer_name_or_path: str, sample: Optional[int] = None):
    try:
        from transformers import AutoTokenizer
    except Exception:
        print("[warn] transformers not available; skipping tokenization stats.")
        return None

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    # Ensure special tokens present (best if you pass your trained checkpoint tokenizer)
    special = {"additional_special_tokens": ["<CONTEXT>", "<POPULATION>", "<FITNESS>", "<TARGET>"]}
    tok.add_special_tokens(special)

    lengths = Counter()
    truncations = 0
    total = 0

    for path in paths:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if sample is not None and i >= sample:
                    break
                if not line.strip():
                    continue
                ex = json.loads(line)
                input_text = f"<CONTEXT>{ex['context']}<POPULATION>{ex['population']}<TARGET>"
                target_text = ex['target']

                inp_ids = tok(input_text, add_special_tokens=False)['input_ids']
                tgt_ids = tok(target_text + (tok.eos_token or ''), add_special_tokens=False)['input_ids']
                full = inp_ids + tgt_ids
                lengths[min(len(full), context_length)] += 1
                if len(full) > context_length:
                    truncations += 1
                total += 1

    return {
        'total_samples': total,
        'truncated_frac': (truncations / total) if total else 0.0,
        'length_histogram_top': lengths.most_common(10),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True, help='One or more JSONL files')
    ap.add_argument('--sample', type=int, default=50000, help='Limit examples per file (default: 50k)')
    ap.add_argument('--context_length', type=int, default=1024, help='Context length to evaluate truncation')
    ap.add_argument('--tokenizer', type=str, default=None, help='Tokenizer path or model name (use your checkpoint tokenizer for accuracy)')
    args = ap.parse_args()

    print('=== One-step Dataset Audit ===')
    for p in args.files:
        stats = audit_jsonl(p, sample=args.sample)
        print(f"\nFILE: {p}")
        print(f"  n: {stats['n']}")
        print(f"  constants_top: {stats['constants_top']}")
        print(f"  first_tok_top: {stats['first_tok_top']}")
        print(f"  simple_var_frac: {stats['simple_var_frac']:.3f}")

    if args.tokenizer:
        tok_stats = tokenize_length_stats(args.files, args.context_length, args.tokenizer, sample=args.sample)
        if tok_stats:
            print('\nTokenization/length stats:')
            print(f"  total_samples: {tok_stats['total_samples']}")
            print(f"  truncated_frac: {tok_stats['truncated_frac']:.3f}")
            print(f"  length_histogram_top: {tok_stats['length_histogram_top']}")


if __name__ == '__main__':
    main()

