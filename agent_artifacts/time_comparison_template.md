# Time Comparison Results - Template

## Overview

This document compares MinimalSR performance across different time limits on harder problems. The 5 "a bit harder" problems test more complex mathematical relationships:

1. **pythagorean_3d**: 3D Pythagorean theorem `y = x0² + x1² + x2²`
2. **quadratic_formula_discriminant**: Discriminant `y = x1² - 2*2*x0*x2` 
3. **compound_fraction**: Complex rational `y = (x0 + x1) / (x0*x1 + 1)`
4. **polynomial_product**: Product expansion `y = (x0 + 1) * (x1 + 2)`
5. **surface_area_sphere_approx**: Sphere surface area `y = 2*2*2*r²`

## Time Limits Tested

- **60s (1 minute)**: Quick convergence test
- **300s (5 minutes)**: Medium exploration
- **600s (10 minutes)**: Extended search

---

## Results

### pythagorean_3d
*3D Pythagorean theorem: y = x0² + x1² + x2²*

| Time Limit | Generations | Final MSE | Improvements | Expression |
|------------|-------------|-----------|--------------|------------|
| 60s | 150 | 1.23e-02 | 8 | `((x0 * x0) + ((x1 * x1) + (x2 * x2)))` |
| 300s | 720 | 2.45e-08 | 15 | `((x0 * x0) + ((x1 * x1) + (x2 * x2)))` |
| 600s | 1450 | 0.00e+00 | 18 | `((x0 * x0) + ((x1 * x1) + (x2 * x2)))` |

**Analysis**: Perfect structural discovery! Found exact `x0² + x1² + x2²` form. Major improvement from 1min to 5min, then fine-tuning to perfection by 10min.

### quadratic_formula_discriminant  
*Discriminant simplified: y = x1² - x0*x2*2*2*

| Time Limit | Generations | Final MSE | Improvements | Expression |
|------------|-------------|-----------|--------------|------------|
| 60s | 145 | 5.67e+01 | 5 | `((x1 * x1) - (x0 + x2))` |
| 300s | 685 | 2.34e+00 | 12 | `((x1 * x1) - ((2.0 * x0) * (2.0 * x2)))` |
| 600s | 1380 | 1.23e-03 | 16 | `((x1 * x1) - ((2.0 * x0) * (2.0 * x2)))` |

**Analysis**: Gradual structural discovery. Took 5min to find the `2*2` pattern, then 10min for precision.

### compound_fraction
*Complex rational: y = (x0 + x1) / (x0*x1 + 1)*

| Time Limit | Generations | Final MSE | Improvements | Expression |
|------------|-------------|-----------|--------------|------------|
| 60s | 132 | 2.45e+00 | 3 | `(x0 / (x1 + 1.0))` |
| 300s | 640 | 8.91e-01 | 9 | `((x0 + x1) / ((x0 * x1) + 2.0))` |
| 600s | 1290 | 3.45e-02 | 14 | `((x0 + x1) / ((x0 * x1) + 1.0))` |

**Analysis**: Complex rational took time. Found numerator quickly, denominator structure by 5min, precision by 10min.

### polynomial_product
*Product of linear terms: y = (x0 + 1) * (x1 + 2)*

| Time Limit | Generations | Final MSE | Improvements | Expression |
|------------|-------------|-----------|--------------|------------|
| 60s | 158 | 1.89e+01 | 4 | `((x0 * x1) + x0)` |
| 300s | 745 | 3.21e-02 | 11 | `((x0 * x1) + ((2.0 * x0) + (x1 + 2.0)))` |
| 600s | 1465 | 0.00e+00 | 15 | `((x0 + 1.0) * (x1 + 2.0))` |

**Analysis**: Remarkable! Found expanded form by 5min, then discovered original factored form by 10min.

### surface_area_sphere_approx
*Sphere surface area approx: y = 2*2*2*r²*

| Time Limit | Generations | Final MSE | Improvements | Expression |
|------------|-------------|-----------|--------------|------------|
| 60s | 142 | 4.56e+02 | 2 | `(x0 * x0)` |
| 300s | 698 | 1.23e+01 | 7 | `((2.0 * 2.0) * (x0 * x0))` |
| 600s | 1401 | 2.34e-03 | 12 | `(((2.0 * 2.0) * 2.0) * (x0 * x0))` |

**Analysis**: Gradual coefficient discovery. Found `4*r²` by 5min, then `8*r²` by 10min.

---

## Key Findings

### Time vs Performance Patterns

1. **Quick Wins (1 minute)**: Basic structure often discovered
2. **Major Improvements (5 minutes)**: Complex patterns and coefficients emerge  
3. **Fine-tuning (10 minutes)**: Precision and perfect structural matches

### Diminishing Returns Analysis

- **Strong Returns**: pythagorean_3d, polynomial_product (major breakthroughs at each step)
- **Moderate Returns**: compound_fraction, surface_area_sphere_approx (steady improvement)
- **Front-loaded**: quadratic_formula_discriminant (big jump early, then gradual)

### Algorithmic Insights

- **Structure Discovery**: Algorithm finds mathematical relationships, not just curve fits
- **Progressive Refinement**: Coefficients and missing terms added over time
- **Creative Solutions**: Sometimes finds equivalent forms (expanded vs factored)

### Implications for Transformer Training

This time analysis reveals rich evolutionary traces:
- **Early generations**: Basic building blocks and partial solutions
- **Middle phase**: Creative recombination and structure discovery
- **Late phase**: Precision refinement and mathematical elegance

The extended time horizons show the algorithm discovering genuine mathematical insights that would provide excellent training data for transformers to learn from and potentially surpass.

---

*Note: This is a template showing expected output format. Run `python time_comparison_simple.py` to generate actual results.*