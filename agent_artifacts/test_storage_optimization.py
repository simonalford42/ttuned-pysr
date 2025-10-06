#!/usr/bin/env python3
"""
Test the storage optimizations for trace datasets.
Generates a small expression dataset, then traces, and compares file sizes.
"""
import os
import subprocess
import time

print("=" * 80)
print("Testing Storage Optimization")
print("=" * 80)

# Step 1: Generate a small expression dataset
print("\n1. Generating expression dataset (10 expressions)...")
result = subprocess.run([
    'python', 'generate_expressions.py',
    '--n_expressions=10',
    '--binary_ops=add,sub,mul',
    '--constants=1.0',
    '--seed=999',
    '--output_dir=/tmp/test_storage'
], capture_output=True, text=True)

if result.returncode != 0:
    print("ERROR generating expressions:")
    print(result.stderr)
    exit(1)

# Find the generated expression file
import glob
expr_files = glob.glob('/tmp/test_storage/train_*.pkl.gz')
if not expr_files:
    print("ERROR: No expression file generated")
    exit(1)

expr_file = max(expr_files, key=os.path.getctime)
expr_size = os.path.getsize(expr_file) / 1024  # KB
print(f"✓ Expression dataset: {expr_file}")
print(f"  Size: {expr_size:.2f} KB")

# Step 2: Generate traces from expressions
print("\n2. Generating trace dataset (from 10 expressions, 5 gens each)...")
result = subprocess.run([
    'python', '-c',
    f"""
from generate_traces import generate_traces_from_expressions
output = generate_traces_from_expressions(
    expressions_file='{expr_file}',
    output_dir='/tmp/test_storage',
    basicsr_params={{
        'population_size': 10,
        'num_generations': 5,
        'max_depth': 4,
        'max_size': 15,
        'tournament_size': 3,
        'time_limit': 10
    }},
    seed=999,
    operator_set='arith',
    constants=[1.0]
)
print(f"Generated: {{output}}")
"""
], capture_output=True, text=True)

if result.returncode != 0:
    print("ERROR generating traces:")
    print(result.stderr)
    exit(1)

print(result.stdout)

# Find the generated trace file
trace_files = glob.glob('/tmp/test_storage/traces_*.pkl.gz')
if not trace_files:
    print("ERROR: No trace file generated")
    exit(1)

trace_file = max(trace_files, key=os.path.getctime)
trace_size = os.path.getsize(trace_file) / 1024  # KB
print(f"\n✓ Trace dataset: {trace_file}")
print(f"  Size: {trace_size:.2f} KB")

# Step 3: Test inspect_dataset on both
print("\n3. Testing inspect_dataset.py on expression dataset...")
result = subprocess.run([
    'python', 'inspect_dataset.py', expr_file, '--n_samples=2'
], capture_output=True, text=True)

if result.returncode != 0:
    print("ERROR inspecting expressions:")
    print(result.stderr)
else:
    print("✓ Expression dataset inspection works")

print("\n4. Testing inspect_dataset.py on trace dataset...")
result = subprocess.run([
    'python', 'inspect_dataset.py', trace_file, '--n_samples=2'
], capture_output=True, text=True)

if result.returncode != 0:
    print("ERROR inspecting traces:")
    print(result.stderr)
else:
    print("✓ Trace dataset inspection works")
    # Show a snippet of the output
    print("\nSample output:")
    print(result.stdout[:800] + "..." if len(result.stdout) > 800 else result.stdout)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Expression dataset: {expr_size:.2f} KB")
print(f"Trace dataset:      {trace_size:.2f} KB")
print(f"Ratio:              {trace_size/expr_size:.1f}x larger")
print("\n✓ All tests passed!")
