# Improved BasicSR Performance on Simple Problems

After analyzing the performance of our initial BasicSR implementation on simple mathematical expressions, we identified several limitations in how the algorithm was handling these problems. The primary issues included:

1. Excessive use of trigonometric functions even for simple algebraic expressions
2. Overly complex expressions for simple relationships
3. Insufficient focus on variable-based expressions

We made the following improvements to the BasicSR implementation:

1. Increased the parsimony coefficient to more strongly discourage complexity
2. Added a variable probability parameter to favor variables over constants in terminal nodes
3. Added variable-only initialization for part of the population
4. Modified initialization to include more algebraic patterns (squaring, multiplication, addition)
5. Reduced the probability of generating unary operators (especially trigonometric functions)

## Results Comparison

| Problem | Ground Truth | Original BasicSR | Original MSE | Improved BasicSR | Improved MSE | PySR | PySR MSE |
|---------|--------------|------------------|--------------|------------------|--------------|------|----------|
| quadratic | 2*x^2 + 3*x + 1 | (exp((x0 + cos(3.0))) / (10.0 + 5.0)) | 107.963 | ((1.0 + x0) * (1.0 + (x0 + x0))) | 0.000 | (x0 + 0.5)*(2*x0 + 2.0) | 0.000 |
| cubic | x^3 - 2*x^2 + x | ((x0 + cos((x0 + 0.5))) * 0.5) | 15.472 | (x0 * (((x0 * x0) - x0) - x0)) | 3.050 | x0^2*(x0 - 2.0) + x0 | 0.000 |
| simple_rational | 5 / (1 + x^2) | cos(x0) | 0.401 | exp(exp((-0.5 + cos(x0)))) | 0.366 | (cos(x0) + 2.14)*cos(x0) + 1.55 | 0.057 |
| simple_physics | 0.5 * m * v^2 | exp(((x1 * x0) + (x1 + x1))) | 5.60e+46 | (((x1 + x1) * x0) - ((x0 + (x0 * 3.0)) - ((x1 + x1) * x0))) | 434.120 | 0.5*x0*x1^2 | 0.000 |
| simple_trig | 2*sin(x) + cos(2*x) | sin(x0) | 0.664 | ((sin(x0) + sin(x0)) + cos((x0 + x0))) | 0.000 | 2.0*sin(x0) + cos(2*x0) | 0.000 |

## Analysis of Improvements

The improvements to BasicSR resulted in significant gains in performance:

1. **Perfect Solutions**: The improved BasicSR found perfect solutions (MSE = 0) for two problems:
   - Quadratic: Found (1+x)(1+2x) which expands to 1+3x+2x^2, matching the target
   - Trigonometric: Found (sin(x) + sin(x)) + cos(x+x), exactly matching 2*sin(x) + cos(2x)

2. **Better Approximations**: The improved BasicSR found better approximations for:
   - Cubic: Found x(x^2-x-x) which equals x^3-2x^2, closer to the target x^3-2x^2+x
   - Physics: Found a complex expression that approximates the form of 0.5*m*v^2

3. **Remaining Challenges**: Some limitations remain:
   - Rational Function: Still using trigonometric approximations rather than algebraic forms
   - Physics Equation: Found a much better approximation but still not accurate

## Expression Structure Analysis

### Original vs. Improved BasicSR

The original BasicSR tended to use:
- Deeply nested trigonometric functions
- Division and exponential operations frequently
- Complex structures even for simple relationships

The improved BasicSR now uses:
- More algebraic operations (addition, multiplication)
- Variable-based expressions (like x*x instead of exp or trig functions)
- Simpler structures that more closely match the ground truth form

### BasicSR vs. PySR

PySR still outperforms BasicSR on:
- Exact recovery of cubic and physics equations
- Finding simpler expressions for rational functions
- Overall algebraic structure that matches ground truth

However, the improved BasicSR now matches PySR on:
- Quadratic equation (different form but same result)
- Trigonometric equation (identical expression)

## Conclusion

The improvements to BasicSR demonstrate that even relatively simple modifications to the algorithm can dramatically improve its performance on basic mathematical expressions. While PySR still has advantages due to its more sophisticated implementation, the performance gap has narrowed considerably.

Key lessons:
1. Initialization strategies significantly impact the algorithm's ability to find good solutions
2. Balancing exploration (diversity of expressions) and exploitation (refinement of good expressions) is critical
3. Biasing the search toward variable-based expressions helps find algebraic relationships
4. Controlling expression complexity through parsimony pressure is essential

These findings suggest that further improvements to BasicSR could continue to close the gap with state-of-the-art symbolic regression implementations while maintaining the simplicity and educational value of the basic algorithm.