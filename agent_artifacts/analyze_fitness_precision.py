#!/usr/bin/env python3
"""
Analyze token and character savings from formatting <FITNESS> values
to 3 significant figures in population strings.

Usage:
  python analyze_fitness_precision.py datasets/training/gen50_arith_10_c05_20251016_220146_basic.jsonl

Outputs aggregate stats and a few before/after examples.
"""
import json
import math
import re
import sys
from statistics import mean

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None


FITNESS_RE = re.compile(r"(<FITNESS>)([-+]?\d*\.\d+|[-+]?\d+(?:[eE][-+]?\d+)?)")


def format_sigfigs(x: float, n: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return str(v)
    return f"{v:.{n}g}"


def round_fitness_in_population(pop: str, n: int = 3) -> str:
    def _repl(m: re.Match) -> str:
        return m.group(1) + format_sigfigs(m.group(2), n)
    return FITNESS_RE.sub(_repl, pop)


def count_tokens(text: str) -> int:
    if ENC is None:
        # Fallback: rough approximation (4 chars/token)
        return math.ceil(len(text) / 4)
    return len(ENC.encode(text))


def main(path: str):
    total_lines = 0
    orig_tokens = []
    new_tokens = []
    orig_chars = []
    new_chars = []
    examples = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            total_lines += 1
            obj = json.loads(line)
            pop = obj.get("population", "")
            pop_new = round_fitness_in_population(pop, 3)

            orig_chars.append(len(pop))
            new_chars.append(len(pop_new))

            orig_tokens.append(count_tokens(pop))
            new_tokens.append(count_tokens(pop_new))

            if len(examples) < 5 and pop != pop_new:
                examples.append((pop, pop_new))

    saved_tokens = [o - n for o, n in zip(orig_tokens, new_tokens)]
    saved_chars = [o - n for o, n in zip(orig_chars, new_chars)]

    print("Lines:", total_lines)
    print("Average tokens (orig -> new):", f"{mean(orig_tokens):.1f}", "->", f"{mean(new_tokens):.1f}")
    print("Average token save:", f"{mean(saved_tokens):.1f}", f"({mean(saved_tokens)/mean(orig_tokens)*100:.1f}% )")
    print("Median token save:", sorted(saved_tokens)[len(saved_tokens)//2])
    print("Min/Max token save:", min(saved_tokens), "/", max(saved_tokens))
    print()
    print("Average chars (orig -> new):", f"{mean(orig_chars):.1f}", "->", f"{mean(new_chars):.1f}")
    print("Average char save:", f"{mean(saved_chars):.1f}", f"({mean(saved_chars)/mean(orig_chars)*100:.1f}% )")
    print("Min/Max char save:", min(saved_chars), "/", max(saved_chars))
    print()
    for i, (a, b) in enumerate(examples, 1):
        print(f"Example {i}:")
        print("  BEFORE:", a[:300] + ("..." if len(a) > 300 else ""))
        print("  AFTER :", b[:300] + ("..." if len(b) > 300 else ""))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_fitness_precision.py <path-to-jsonl>")
        sys.exit(1)
    main(sys.argv[1])

