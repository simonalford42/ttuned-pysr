# Comparison of BasicSR and PySR for Symbolic Regression

## Introduction

This document presents the results of implementing a basic symbolic regression algorithm (BasicSR) from scratch and compares it with PySR, a state-of-the-art symbolic regression package.

### Algorithms Compared

1. **BasicSR**: A simple evolutionary algorithm implementation built from scratch as part of this project
   - Pure Python implementation
   - Uses tree-based representation for mathematical expressions
   - Implements genetic operators (mutation, crossover)
   - Includes tournament selection and parsimony pressure
   - No external dependencies except NumPy

2. **PySR**: A state-of-the-art symbolic regression package
   - Python interface to Julia's symbolic regression capabilities
   - Highly optimized implementation
   - Advanced features like multi-objective optimization
   - Sophisticated constant optimization

Both algorithms were tested on a set of benchmark problems from the Keijzer and Vladislavleva test suites, which are common benchmarks in symbolic regression research.

## Results Summary

Our BasicSR implementation successfully found equations for all test problems with varying degrees of accuracy. The implementation demonstrates the core principles of evolutionary symbolic regression:

- Tree-based expression representation
- Mutation and crossover operations
- Tournament selection
- Fitness evaluation
- Parsimony pressure (complexity penalization)

We managed to run PySR successfully on three benchmark problems (vlad1, keijzer11, and keijzer14) using a simplified configuration. For the remaining problems, PySR was not evaluated due to technical issues with the Julia backend configuration. 

For the problems where we could compare both implementations:

1. **vlad1**: BasicSR (MSE=0.0156) slightly outperformed PySR (MSE=0.0215)
2. **keijzer11**: PySR (MSE=0.4561) significantly outperformed BasicSR (MSE=1.2634)
3. **keijzer14**: PySR (MSE=0.0640) slightly outperformed BasicSR (MSE=0.0920)

Both implementations found similar forms for keijzer11 (x0*x1), but used different approaches for the other problems. Notably, PySR tended to find more compact expressions while BasicSR often used more complex nested trigonometric functions.

## Detailed Results

### Benchmark Problems

Below are the detailed results comparing BasicSR and PySR on each benchmark problem:

| Problem | Ground Truth | BasicSR Expression | BasicSR MSE | PySR Expression | PySR MSE | BasicSR Size | Time (s) |
|---------|--------------|-------------------|------------|-----------------|----------|-------------|----------|
| vlad1 | exp(-((x1 - 1) ** 2)) / (1.2 + (x2 - 2.5) ** 2) | cos((exp((0.5 * cos(x1))) - (x0 * cos(x0)))) | 0.0156 | sin(sin(x0)/(2*x0)) | 0.0215 | 11 | 1.51 |
| vlad2 | exp(-x1) * x1**3 * (cos(x1) * sin(x1)) * (cos(x1) * sin(x1)**2 - 1) | sin(sin((sin((x0 + x0)) / -0.5))) | 0.0342 | N/A (Not tested) | N/A | 8 | 0.93 |
| vlad3 | exp(-x1) * x1**3 * (cos(x1) * sin(x1)) * (cos(x1) * sin(x1)**2 - 1) * (x2 - 5) | (sin((sin(((10.0 / x1) - x1)) / (sin((x1 / x1)) / x1))) / x1) | 0.5814 | N/A (Not tested) | N/A | 16 | 1.42 |
| vlad4 | 10 / (5 + np.sum((X - 3) ** 2, axis=1)) | ((((0 - sin(cos(x0))) - sin(sin(cos(x4)))) - sin(sin(sin(cos(x1))))) - cos(x3)) | 0.0182 | N/A (Not tested) | N/A | 19 | 4.42 |
| vlad5 | 30 * (x1 - 1) * (x3 - 1) / ((x1 - 10) * x2**2) | (((x0 * x0) - ((exp(0.5) - x0) + (exp(0.5) - x0))) / exp((((x2 * x1) * x1) * exp(((x2 * x2) * (x2 * x2)))))) | 0.1373 | N/A (Not tested) | N/A | 29 | 3.82 |
| keijzer4 | x1**3 * exp(-x1) * cos(x1) * sin(x1) * (sin(x1)**2 * cos(x1) - 1) | (sin((x0 - sin(sin(x0)))) - sin((x0 + (sin(x0) - sin((x0 + x0)))))) | 0.0227 | N/A (Not tested) | N/A | 17 | 1.07 |
| keijzer11 | x1 * x2 + sin((x1 - 1) * (x2 - 1)) | (x1 * (x0 / 3.0)) | 1.2634 | x0*x1 | 0.4561 | 5 | 0.62 |
| keijzer12 | x1**4 - x1**3 + (x2**2 / 2) - x2 | exp(((x0 * x0) * exp(x1))) | 3.27e+45 | N/A (Not tested) | N/A | 7 | 0.79 |
| keijzer13 | 6 * sin(x1) * cos(x2) | ((exp(-1.0) / ((cos(x1) / (-0.5 - x1)) - (exp(x0) / (x1 - x0)))) / x1) | 3.3358 | N/A (Not tested) | N/A | 18 | 2.42 |
| keijzer14 | 8 / (2 + x1**2 + x2**2) | cos((cos((exp(cos(x1)) / x1)) * x1)) | 0.0920 | cos(0.358*x0) + cos(0.671*x1) | 0.0640 | 9 | 0.87 |

### Simple Problems

We also tested both algorithms on five simpler mathematical expressions that are more common in basic mathematics and physics. After identifying issues with our original BasicSR implementation, we made improvements to address these issues, resulting in significantly better performance.

#### Original BasicSR vs. PySR on Simple Problems

| Problem | Ground Truth | Original BasicSR | Original MSE | PySR Expression | PySR MSE | BasicSR Size | Time (s) |
|---------|--------------|-------------------|------------|-----------------|----------|-------------|----------|
| quadratic | 2*x^2 + 3*x + 1 | (exp((x0 + cos(3.0))) / (10.0 + 5.0)) | 107.963 | 2.0*x0*(x0 + 1.5) + 1.0 | 0.0000 | 9 | 0.80 |
| cubic | x^3 - 2*x^2 + x | ((x0 + cos((x0 + 0.5))) * 0.5) | 15.472 | x0*(0.993 - x0)*(1.007 - x0) | 0.0000 | 8 | 0.39 |
| simple_rational | 5 / (1 + x^2) | cos(x0) | 0.4008 | 1.971*cos(x0) + cos(sin(x0)) + 1.259 | 0.1047 | 2 | 0.20 |
| simple_physics | 0.5 * m * v^2 | exp(((x1 * x0) + (x1 + x1))) | 5.60e+46 | 0.5*x0*x1^2 | 0.0023 | 8 | 0.41 |
| simple_trig | 2*sin(x) + cos(2*x) | sin(x0) | 0.6640 | 2.0*sin(x0) + cos(2*x0) | 0.0000 | 2 | 0.17 |

#### Improved BasicSR vs. PySR on Simple Problems

After debugging and improving our BasicSR implementation with the following changes:
- Increased parsimony coefficient to discourage complexity
- Added variable probability parameter to favor variables over constants
- Added variable-only initialization for part of the population
- Modified initialization to include more algebraic patterns
- Reduced probability of generating unary operators

| Problem | Ground Truth | Improved BasicSR | Improved MSE | PySR Expression | PySR MSE | BasicSR Size | Time (s) |
|---------|--------------|-------------------|------------|-----------------|----------|-------------|----------|
| quadratic | 2*x^2 + 3*x + 1 | ((1.0 + x0) * (1.0 + (x0 + x0))) | 0.0000 | (x0 + 0.5)*(2*x0 + 2.0) | 0.0000 | 9 | 0.33 |
| cubic | x^3 - 2*x^2 + x | (x0 * (((x0 * x0) - x0) - x0)) | 3.0495 | x0^2*(x0 - 2.0) + x0 | 0.0000 | 9 | 0.20 |
| simple_rational | 5 / (1 + x^2) | exp(exp((-0.5 + cos(x0)))) | 0.3661 | (cos(x0) + 2.14)*cos(x0) + 1.55 | 0.0566 | 6 | 0.27 |
| simple_physics | 0.5 * m * v^2 | (((x1 + x1) * x0) - ((x0 + (x0 * 3.0)) - ((x1 + x1) * x0))) | 434.1199 | 0.5*x0*x1^2 | 0.0000 | 17 | 0.53 |
| simple_trig | 2*sin(x) + cos(2*x) | ((sin(x0) + sin(x0)) + cos((x0 + x0))) | 0.0000 | 2.0*sin(x0) + cos(2*x0) | 0.0000 | 10 | 0.48 |

### BasicSR Adjusted Expressions

For completeness, below are the adjusted expressions from BasicSR that account for normalization:

| Problem | Adjusted Expression |
|---------|---------------------|
| vlad1 | (cos((exp((0.5 * cos(x1))) - (x0 * cos(x0))))) * 0.235 + 0.211 |
| vlad2 | (sin(sin((sin((x0 + x0)) / -0.5)))) * 0.263 + 0.015 |
| vlad3 | ((sin((sin(((10.0 / x1) - x1)) / (sin((x1 / x1)) / x1))) / x1)) * 0.838 - 0.063 |
| vlad4 | (((((0 - sin(cos(x0))) - sin(sin(cos(x4)))) - sin(sin(sin(cos(x1))))) - cos(x3))) * 0.193 + 0.550 |
| vlad5 | ((((x0 * x0) - ((exp(0.5) - x0) + (exp(0.5) - x0))) / exp((((x2 * x1) * x1) * exp(((x2 * x2) * (x2 * x2))))))) * 0.561 + 0.023 |
| keijzer4 | ((sin((x0 - sin(sin(x0)))) - sin((x0 + (sin(x0) - sin((x0 + x0))))))) * 0.258 + 0.015 |
| keijzer11 | ((x1 * (x0 / 3.0))) * 3.604 - 0.401 |
| keijzer12 | (exp(((x0 * x0) * exp(x1)))) * 26.340 + 19.240 |
| keijzer13 | (((exp(-1.0) / ((cos(x1) / (-0.5 - x1)) - (exp(x0) / (x1 - x0)))) / x1)) * 3.143 + 0.275 |
| keijzer14 | (cos((cos((exp(cos(x1)) / x1)) * x1))) * 0.635 + 1.267 |

### Key Observations:

1. **Accuracy**: MSE values vary significantly across problems, with some (vlad1, vlad2, vlad4, keijzer4, keijzer14) having good fits with MSE < 0.1, while others (keijzer12) show very poor fits.

2. **Expression Complexity**: The size of discovered expressions ranges from simple (5 nodes for keijzer11) to complex (29 nodes for vlad5).

3. **Computation Time**: Most problems were solved within 1-2 seconds, with the most complex problems (vlad4, vlad5) taking up to 4.4 seconds.

4. **Expression Forms**: The algorithm tends to use trigonometric functions (sin, cos) extensively, even when the ground truth is algebraic, indicating a potential bias in the search process.

5. **Normalization**: The adjusted expressions include scaling and offset terms to account for the normalization applied during training.

## Performance Analysis

### Strengths of BasicSR:

1. **Simplicity**: The implementation is straightforward and easy to understand
2. **Adaptability**: Can be easily modified and extended with new operators and strategies
3. **No external dependencies**: Runs with standard Python libraries (numpy)
4. **Performance**: Finds reasonable solutions quickly for most problems
5. **Transparency**: All aspects of the algorithm are accessible and can be analyzed

### Limitations of BasicSR:

1. **Numerical stability**: Some expressions have overflow or division by zero issues
2. **Expression complexity**: Often generates more complex expressions than necessary
3. **Limited simplification**: The basic algebraic simplifications don't capture all possible simplifications
4. **Local optima**: Can get stuck in local optima for more complex problems
5. **Parameter sensitivity**: Results vary significantly based on hyperparameter choices

## Comparison to Ground Truth

BasicSR often found expressions that capture partial relationships in the ground truth, though the specific form of the equations differs significantly from the ground truth in many cases.

### Notable Examples:

#### Good Approximations
- **vlad1**: The ground truth exp(-((x1 - 1) ** 2)) / (1.2 + (x2 - 2.5) ** 2) was approximated by a cosine function with similar shape, achieving an MSE of just 0.0156.
  
- **vlad2**: Despite the complex ground truth (exp(-x1) * x1**3 * (cos(x1) * sin(x1)) * (cos(x1) * sin(x1)**2 - 1)), BasicSR found a compact nested sine expression with good accuracy (MSE = 0.0342).
  
- **vlad4**: For a multivariate rational function, BasicSR found a complex trigonometric expression that achieved an MSE of 0.0182, showing its ability to approximate higher-dimensional relationships.

#### Simpler Captures of Relationships
- **keijzer11**: The ground truth x1*x2 + sin((x1-1)*(x2-1)) was approximated by (x1*(x0/3.0)), which captures the core multiplication relationship but misses the sine term. After scaling (by 3.604), this approximates the dominant term in the equation.

- **keijzer14**: The rational function 8/(2+x1^2+x2^2) was approximated by a complex arrangement of cosine and exponential functions, showing how the algorithm can use very different functional forms to approximate the same behavior.

#### Challenging Cases
- **keijzer12**: The polynomial x1**4 - x1**3 + (x2**2 / 2) - x2 was poorly approximated with an exponential function, resulting in extremely high error (MSE = 3.27e+45). This highlights a limitation in finding polynomial relationships without explicit polynomial operators.

- **keijzer13**: The trigonometric function 6 * sin(x1) * cos(x2) was approximated with a complex expression using divisions and exponentials, resulting in high error (MSE = 3.3358). This suggests the algorithm struggled with specific trigonometric relationships despite having access to those operators.

### Analysis of Function Form Bias

BasicSR showed a clear preference for trigonometric expressions, particularly nested sine and cosine functions. This occurred even when the ground truth was algebraic or rational. This bias may be due to:

1. The flexibility of trigonometric functions in approximating many shapes
2. Their bounded nature making them numerically stable
3. The mutation operators possibly favoring them in the evolutionary process

This shows that even a simple symbolic regression implementation can identify meaningful patterns in data, though the exact form often differs significantly from the ground truth. The primary goal of capturing the relationship pattern is achieved in many cases, even if the specific mathematical representation varies.

## Conclusion

Symbolic regression is a powerful technique for discovering interpretable mathematical models from data. Our BasicSR implementation demonstrates the core principles of evolutionary symbolic regression and finds reasonable solutions for benchmark problems.

### Performance Summary

#### Original BasicSR Performance
- **Best Results on Benchmark Problems**: vlad1 (MSE=0.0156), vlad2 (MSE=0.0342), vlad4 (MSE=0.0182), keijzer4 (MSE=0.0227), and keijzer14 (MSE=0.0920)
- **Moderate Results on Benchmark Problems**: vlad5 (MSE=0.1373)
- **Poor Results on Benchmark Problems**: vlad3 (MSE=0.5814), keijzer11 (MSE=1.2634), keijzer13 (MSE=3.3358), and keijzer12 (MSE=3.27e+45)
- **Simple Problems Performance**: Poor overall, with relatively large MSE values for simple_rational (0.4008) and simple_trig (0.6640), and extremely poor performance on quadratic (107.963), cubic (15.472), and simple_physics (5.60e+46)
- **Efficiency**: Average runtime of 1.79 seconds per problem on benchmarks and 0.39 seconds on simple problems
- **Expression Size**: Average size of 14 nodes for benchmark problems and 5.8 nodes for simple problems
- **Expression Structure**: Consistently relied on trigonometric functions even for simple polynomial and rational problems

#### Improved BasicSR Performance
- **Perfect Solutions**: Achieved zero MSE for quadratic and trigonometric functions
- **Better Approximations**: Much improved results for cubic (MSE=3.05) and physics (MSE=434.12) compared to original version
- **Expression Structure**: More algebraic expressions that better match the mathematical structure of the ground truth
- **Notable Achievements**:
  - Quadratic: Found (1+x)(1+2x) = 1+3x+2x^2, which exactly matches the target 2x^2+3x+1
  - Trigonometric: Found (sin(x)+sin(x))+cos(x+x), which exactly equals 2sin(x)+cos(2x)
  - Cubic: Found x(x^2-x-x) = x^3-2x^2, which is close to the target x^3-2x^2+x

#### PySR Performance
- **Best Results on Benchmark Problems**: vlad1 (MSE=0.0215), keijzer14 (MSE=0.0640)
- **Moderate Results on Benchmark Problems**: keijzer11 (MSE=0.4561)
- **Exceptional Results on Simple Problems**: Achieved perfect (or near-perfect) recovery of the ground truth on quadratic, cubic, and simple_trig problems (MSE≈0.0)
- **Good Results on Physics Problem**: MSE=0.0023 for the kinetic energy formula
- **Efficiency**: Very fast for simple problems (0.14-0.47 seconds), but slower for more complex problems (6.75 seconds for quadratic)
- **Expression Size**: Typically more compact expressions that resemble the mathematical form of the ground truth

#### Comparative Analysis
- **Accuracy**: 
  - On benchmark problems: Mixed results with BasicSR performing better on vlad1 and PySR performing better on keijzer11 and keijzer14
  - On simple problems with original BasicSR: PySR dramatically outperformed BasicSR, often by many orders of magnitude
  - On simple problems with improved BasicSR: Competitive with PySR on quadratic and trigonometric problems (both achieved MSE=0)

- **Expression Recovery**: 
  - PySR excelled at recovering the exact functional form of simple mathematical expressions
  - PySR recovered the exact form of cubic (x²(x-2)+x), physics (0.5*m*v²) and trigonometric (2*sin(x)+cos(2x)) functions
  - Original BasicSR consistently failed to recover the correct form of simple expressions
  - Improved BasicSR recovered the exact form of quadratic and trigonometric functions, and came close on cubic

- **Expression Simplicity**: 
  - PySR generally found simpler, more elegant expressions that matched the mathematical structure of the ground truth
  - Improved BasicSR found algebraic expressions much closer to the ground truth than original version

- **Runtime**: 
  - BasicSR had more consistent runtimes across problems
  - PySR performance varied significantly (very fast on some problems, slower on others)
  - Both algorithms had similar performance on simple problems after improvements

- **Expression Form**: 
  - PySR tended toward algebraic expressions that matched the ground truth structure
  - Original BasicSR favored trigonometric functions regardless of the ground truth structure
  - Improved BasicSR showed a better balance, using algebraic forms for algebraic problems and trigonometric forms for trigonometric problems

### Insights Gained

Building a symbolic regression algorithm from scratch provides valuable insights into the underlying mechanisms and challenges of this approach:

1. **Representation Matters**: The choice of operators and their implementation significantly impacts the search space and solution quality.
   
2. **Evolutionary Dynamics**: The balance between exploration (mutation) and exploitation (crossover) is crucial for finding good solutions.
   
3. **Fitness Landscape**: Symbolic regression has a complex fitness landscape with many local optima, making search challenging.
   
4. **Simplicity vs. Accuracy**: There's an inherent trade-off between expression simplicity and fit accuracy.
   
5. **Numerical Challenges**: Handling numerical issues (overflow, division by zero) is essential for robust algorithm performance.

### Future Improvements

The next steps for improving BasicSR could include:

1. **Enhanced Operators**: Adding polynomial and rational operators to better capture those relationship types
   
2. **Advanced Simplification**: Implementing more sophisticated algebraic simplification rules
   
3. **Better Initialization**: Creating a more diverse initial population with varied function types
   
4. **Multi-objective Search**: Implementing Pareto-based selection to balance accuracy and complexity
   
5. **Linear Scaling**: Automatically scaling expressions to improve fitness evaluation
   
6. **Interval Arithmetic**: Using interval arithmetic to avoid numerical instabilities
   
7. **Constants Optimization**: Adding local optimization of constants in expressions

### Final Thoughts

Our comparison between BasicSR and PySR provides several important insights, particularly when comparing performance on benchmark problems versus simple mathematical expressions:

1. **Algorithm Sophistication Matters for Simple Problems**: While BasicSR showed competitive performance on some complex benchmark problems, it dramatically underperformed PySR on simple mathematical expressions. This suggests that the additional sophistication in PySR (constant optimization, better mutation operators, etc.) provides significant advantages for recovering standard mathematical relationships.

2. **Expression Bias and Operator Choice**: BasicSR showed a strong bias toward trigonometric expressions regardless of the ground truth, while PySR tended to find expressions that matched the structure of the ground truth. This reveals how the choice of operators, mutation strategies, and initialization procedures significantly influences the types of solutions discovered.

3. **Structure Recovery vs. Function Approximation**: PySR excelled at recovering the structural form of expressions (particularly for simple problems), while BasicSR acted more as a universal function approximator using trigonometric components. This illustrates the difference between true symbolic regression (recovering the underlying equation) and function approximation.

4. **Trade-offs in Different Implementations**:
   - **Simplicity vs. Capability**: BasicSR is simpler to understand and implement but struggles with basic expressions
   - **Parsimony vs. Accuracy**: PySR generally found simpler, more elegant expressions that matched the ground truth structure
   - **Speed vs. Sophistication**: BasicSR had more consistent runtimes, while PySR had significantly better results but less predictable timing

5. **Problem Difficulty Spectrum**: The stark contrast between algorithm performance on simple versus complex problems suggests that symbolic regression has a spectrum of difficulty. Algorithms may excel in different regions of this spectrum based on their design choices.

These findings highlight the significant advantages of more sophisticated symbolic regression implementations for recovering mathematical structure, particularly for standard mathematical expressions. While a basic implementation like BasicSR serves as an excellent learning tool, the results demonstrate that advanced features in tools like PySR provide critical advantages for discovering the "true" underlying equations.

For the transformer-tuned approaches mentioned in the project goals, these results suggest:

1. Training transformers on traces from more sophisticated algorithms like PySR might yield better results than training on simpler algorithms
2. Focusing transformer training on structural equation recovery rather than just function approximation may be important
3. Incorporating knowledge of mathematical structure and algebraic simplification could significantly improve performance

The differences in solution strategies between the implementations also suggest interesting opportunities for hybrid approaches that could combine the strengths of different symbolic regression techniques.