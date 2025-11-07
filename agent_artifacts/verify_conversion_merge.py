#!/usr/bin/env python3
"""
Verify that merged convert_data.py reproduces outputs of the original converters.

It compares three formats on a small dataset:
- Autoregressive (original: autoreg_conversion.py)
- One-step (original: convert_data.py default)
- Direct (original: direct_conversion.py)

Usage:
  python scripts/verify_conversion_merge.py \
    --trace datasets/traces/gen10_arith_10_c05_20251016_220146.pkl.gz

Assumes the corresponding expressions file exists and is referenced by the
trace's metadata['source_expressions_file'].
The script will:
  - Copy baseline JSONL files to .baseline copies
  - Re-run the new convert_data.py with flags to regenerate outputs
  - Compare regenerated outputs to baselines for exact equality
"""
import argparse
import gzip
import hashlib
import pickle
import shutil
import subprocess
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Verify merged convert_data produces identical outputs")
    parser.add_argument("--trace", required=True, help="Input trace file (.pkl.gz) for one-step/autoreg tests")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    # Load expressions file from trace metadata
    with gzip.open(trace_path, 'rb') as f:
        trace_obj = pickle.load(f)
    expressions_file = Path(trace_obj['metadata']['source_expressions_file'])
    if not expressions_file.exists():
        # Try resolving relative to repo if metadata held a relative path
        candidate = Path("datasets/expressions") / expressions_file.name
        if candidate.exists():
            expressions_file = candidate
        else:
            print(f"Expressions file not found: {expressions_file}", file=sys.stderr)
            sys.exit(1)

    # Derive baseline output filenames
    def stem_no_pkl_gz(p: Path) -> str:
        name = p.stem
        return name[:-4] if name.endswith('.pkl') else name

    trace_stem = stem_no_pkl_gz(trace_path)
    expr_stem = stem_no_pkl_gz(expressions_file)

    out_dir = Path("datasets/training")
    autoreg_file = out_dir / f"{trace_stem}_autoreg.jsonl"
    one_step_file = out_dir / f"{trace_stem}.jsonl"
    direct_file = out_dir / f"{expr_stem}_direct.jsonl"

    required = [autoreg_file, one_step_file, direct_file]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Baseline files missing:")
        for p in missing:
            print(f"  - {p}")
        print("Please run the original converters to generate baselines before running this check.")
        sys.exit(2)

    # Backup baselines
    backups = []
    for p in required:
        backup = p.with_suffix(p.suffix + ".baseline")
        shutil.copy2(p, backup)
        backups.append(backup)

    # Re-generate outputs using merged convert_data.py
    cmds = [
        [sys.executable, "convert_data.py", "--input", str(trace_path), "--autoreg"],
        [sys.executable, "convert_data.py", "--input", str(trace_path), "--expressions_file", str(expressions_file)],
        [sys.executable, "convert_data.py", "--input", str(expressions_file), "--direct"],
    ]

    for cmd in cmds:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # Compare files
    pairs = [
        (autoreg_file, autoreg_file.with_suffix(autoreg_file.suffix + ".baseline"), "autoreg"),
        (one_step_file, one_step_file.with_suffix(one_step_file.suffix + ".baseline"), "one-step"),
        (direct_file, direct_file.with_suffix(direct_file.suffix + ".baseline"), "direct"),
    ]

    all_ok = True
    for new_path, baseline_path, label in pairs:
        new_hash = sha256_file(new_path)
        base_hash = sha256_file(baseline_path)
        if new_hash == base_hash:
            print(f"OK [{label}]: {new_path.name} matches baseline ({new_hash})")
        else:
            print(f"MISMATCH [{label}]: {new_path.name} != {baseline_path.name}")
            print(f"  new:  {new_hash}")
            print(f"  base: {base_hash}")
            all_ok = False

    if all_ok:
        print("All formats match baselines.")
        sys.exit(0)
    else:
        print("Some formats did not match baselines.")
        sys.exit(3)


if __name__ == "__main__":
    main()

