# Polynomial BasicSR Results on Updated Simple Problems

## Overview

This document presents the results of running our simplified BasicSR algorithm on 10 polynomial and rational function problems. The algorithm was optimized specifically for polynomial problems by:

- Removing trigonometric and exponential operators (sin, cos, exp)
- Limiting constants to only 1.0 and 2.0
- Enhancing polynomial-specific simplification rules
- Focusing on arithmetic operations: +, -, *, /

## Test Problems

The problems were designed to use only arithmetic operations with constants limited to 1 and 2:

1. **quadratic**: `x^2 + x + 1`
2. **cubic**: `x^3 - x^2 - x^2 + x` (equivalent to `x^3 - 2x^2 + x`)
3. **simple_rational**: `1 / (1 + x^2)`
4. **bivariate_product**: `x1 * x2^2`
5. **rational_division**: `x / (x + 1)`
6. **quartic**: `x^4 - x^3 + x^2`
7. **bivariate_sum**: `x1^2 + x2^2`
8. **trivariate_product**: `x1 * x2 * x3`
9. **mixed_polynomial**: `x1^2 - x1*x2 + x2^2`
10. **complex_rational**: `(x1 + x2) / (x1 - x2 + 1)`

## Results Summary

| Problem | Ground Truth | BasicSR Expression | MSE | Size | Status |
|---------|--------------|-------------------|-----|------|--------|
| quadratic | x^2 + x + 1 | ((((((x1 + x0) + x1) / x0) + x0) / (x0 - x0)) * x0) | 1.08e+24 | 15 | ❌ Failed |
| cubic | x^3 - 2x^2 + x | ((((x0 / ((2.0 * 2.0) + x0)) + x0) + 2.0) / ((2.0 * 2.0) + x0)) | 7.09 | 17 | ⚠️ Poor |
| simple_rational | 1 / (1 + x^2) | (x1 / (x0 + (((x1 + x0) + (x0 + (x1 / 2.0))) + (((x0 / x1) / x1) * x0)))) | 0.031 | 21 | ✓ Good |
| bivariate_product | x1 * x2^2 | ((((x1 - ((2.0 / x1) + x0)) + x1) * (x0 - 2.0)) / (x1 - x1)) | 1.74e+27 | 17 | ❌ Failed |
| rational_division | x / (x + 1) | (((x0 / (x1 + 2.0)) / 2.0) / (x0 / (x0 - (((x1 + 2.0) / x0) + ((2.0 - x0) + 2.0))))) | 0.006 | 23 | ✓ Excellent |
| quartic | x^4 - x^3 + x^2 | (((x0 * (x0 - 1.0)) - 2.0) / 2.0) | 7.92 | 9 | ⚠️ Poor |
| bivariate_sum | x1^2 + x2^2 | (((((x0 - x0) - 2.0) - 2.0) + (x0 * x0)) / (x0 - x0)) | 1.40e+22 | 15 | ❌ Failed |
| trivariate_product | x1 * x2 * x3 | (((1.0 - x2) / (1.0 + x0)) * (((1.0 + x0) - (x1 * x0)) - ((x0 - 1.0) * x0))) | 2.95 | 21 | ⚠️ Moderate |
| mixed_polynomial | x1^2 - x1*x2 + x2^2 | ((((((2.0 - x0) - x0) / 2.0) / 2.0) / 2.0) * (((((2.0 - x0) - (x0 * x1)) / 2.0) / 2.0) * x1)) | 2.14 | 25 | ⚠️ Moderate |
| complex_rational | (x1 + x2) / (x1 - x2 + 1) | ((x0 / (x1 + (x1 * (x0 - x1)))) / (x2 + 2.0)) | 97.37 | 13 | ❌ Poor |

## Key Observations

### Successful Cases
- **simple_rational** (MSE: 0.031): Found a complex but functional approximation to `1/(1+x^2)`
- **rational_division** (MSE: 0.006): Excellent performance on `x/(x+1)`, though with a complex expression

### Major Issues Identified

#### 1. Division by Zero Problems
Several expressions contain `(x0 - x0)` or similar terms that evaluate to zero, causing division by zero:
- **quadratic**: `/ (x0 - x0)`
- **bivariate_product**: `/ (x1 - x1)` 
- **bivariate_sum**: `/ (x0 - x0)`

This suggests the algorithm is generating unstable expressions that lead to numerical overflow.

#### 2. Expression Complexity vs. Ground Truth Simplicity
The algorithm tends to generate overly complex expressions even for simple problems:
- Ground truth `x^2 + x + 1` → 15-node complex expression
- Ground truth `x1 * x2^2` → 17-node expression with division by zero

#### 3. Limited Polynomial Pattern Recognition
The algorithm struggles to discover basic polynomial patterns:
- Cannot find simple `x^2` terms for quadratic problems
- Fails to recognize multiplication patterns for `x1 * x2^2`
- Does not discover addition patterns for `x1^2 + x2^2`

#### 4. Variable Index Confusion
Some expressions use incorrect variable indices (e.g., using `x1` when `x0` is expected), suggesting issues with the variable handling in the simplified problems.

## Algorithm Performance Analysis

### Strengths
- **Rational Functions**: Shows competency with some rational functions, particularly simpler ones
- **Expression Diversity**: Generates varied expressions using available operators
- **Complexity Control**: Parsimony pressure helps limit expression size to some extent

### Weaknesses
- **Numerical Stability**: Major issues with division by zero leading to infinite MSE values
- **Pattern Recognition**: Poor at recognizing basic polynomial structures
- **Variable Handling**: Confusion with variable indices in multivariate problems
- **Simplification**: Despite simplification rules, generates unnecessarily complex expressions

## Recommendations for Improvement

1. **Fix Division by Zero**: Implement better checks to prevent `(x - x)` type expressions
2. **Enhanced Initialization**: Add more polynomial-specific initialization patterns (x^2, x^3, etc.)
3. **Variable Index Validation**: Ensure proper variable indexing for multivariate problems
4. **Polynomial-Specific Operators**: Consider adding dedicated polynomial operators or terms
5. **Improved Mutation**: Bias mutations toward polynomial-like structures
6. **Better Fitness Function**: Consider alternative fitness measures that reward structural similarity

## Conclusion

The simplified BasicSR algorithm shows promise for rational function discovery but struggles significantly with basic polynomial problems. The major issues with division by zero and overly complex expressions suggest that further algorithm refinements are needed before proceeding with transformer training on these traces.

The next steps should focus on addressing the numerical stability issues and improving polynomial pattern recognition capabilities.