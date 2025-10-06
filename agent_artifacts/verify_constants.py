#!/usr/bin/env python3
"""
Verify that constants generation works correctly.
"""
import pickle
import gzip
import os
from generate_expressions import generate_e2e_expressions

# Generate expressions with specific constants
print('Generating 15 expressions with constants [1.0, 2.5, 3.14]...')
print('=' * 70)

expressions = generate_e2e_expressions(
    n_expressions=300,
    binary_ops="add,sub,mul",
    unary_ops="",
    complexity=0.5,
    n_input_points=64,
    seed=999,
    constants=[1.0, 2.5, 3.14]
)

print('\nGenerated expressions:')
print('=' * 70)

has_1_0 = []
has_2_5 = []
has_3_14 = []

for i, expr in enumerate(expressions, 1):
    expr_str = expr['expression']
    markers = []
    if '1.0' in expr_str:
        markers.append('1.0✓')
        has_1_0.append(i)
    if '2.5' in expr_str:
        markers.append('2.5✓')
        has_2_5.append(i)
    if '3.14' in expr_str:
        markers.append('3.14✓')
        has_3_14.append(i)

    marker_str = f'  [{" ".join(markers)}]' if markers else ''
    print(f'{i:2d}. {expr_str}{marker_str}')

print('\n' + '=' * 70)
print('VERIFICATION SUMMARY:')
print(f'  Expressions with 1.0:   {len(has_1_0)} {has_1_0 if has_1_0 else "(none)"}')
print(f'  Expressions with 2.5:   {len(has_2_5)} {has_2_5 if has_2_5 else "(none)"}')
print(f'  Expressions with 3.14:  {len(has_3_14)} {has_3_14 if has_3_14 else "(none)"}')
print(f'  Total with our constants: {len(set(has_1_0 + has_2_5 + has_3_14))}')
print('\n✓ Constants are working!' if (has_1_0 or has_2_5 or has_3_14) else '\n✗ No custom constants found - bug still present')
