# Improvement Trajectory Analysis

Each problem was run for 15 seconds to track how the algorithm discovers mathematical structure over time.

## pythagorean_3d

**Final Result**: 8.44e-31 MSE in 25302 generations  
**Expression**: `(x1 * ((x1 + ((x2 * x2) / x1)) + ((x0 * x0) / x1)))`

**Improvement Trajectory** (10 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 7.56e+01 | 1 |  |
| 0.0s | 1 | 2.00e+01 | 5 | 73.6% |
| 0.2s | 555 | 1.66e+01 | 9 | 16.8% |
| 0.2s | 579 | 1.39e+01 | 11 | 16.4% |
| 0.3s | 582 | 1.12e+01 | 15 | 19.5% |
| 0.4s | 732 | 8.13e+00 | 17 | 27.3% |
| 0.4s | 739 | 7.54e+00 | 17 | 7.2% |
| 1.3s | 2176 | 1.00e+00 | 19 | 86.7% |
| 1.4s | 2262 | 9.59e-01 | 19 | 4.1% |
| 1.4s | 2331 | 8.44e-31 | 15 | 100.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 1.4s (Gen 2331)
- Early phase (0-15s): 10 improvements
- Late phase (45-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## quadratic_formula_discriminant

**Final Result**: 3.51e-31 MSE in 25254 generations  
**Expression**: `((x0 + x0) * (((((x1 / 2.0) * x1) / x0) - x2) - x2))`

**Improvement Trajectory** (15 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 1.93e+01 | 1 |  |
| 0.0s | 106 | 1.86e+01 | 7 | 3.7% |
| 0.0s | 110 | 1.86e+01 | 5 | 0.0% |
| 0.0s | 135 | 1.65e+01 | 7 | 11.2% |
| 0.0s | 136 | 8.53e+00 | 7 | 48.3% |
| 0.0s | 138 | 5.97e+00 | 7 | 30.0% |
| 0.1s | 213 | 5.68e+00 | 13 | 4.8% |
| 0.1s | 238 | 5.68e+00 | 9 | 0.0% |
| 0.1s | 239 | 4.76e+00 | 17 | 16.3% |
| 0.1s | 245 | 2.56e+00 | 15 | 46.3% |
| 0.1s | 263 | 2.56e+00 | 11 | 0.0% |
| 0.3s | 553 | 1.43e+00 | 13 | 44.2% |
| 0.3s | 569 | 1.43e+00 | 11 | 0.0% |
| 0.4s | 821 | 3.51e-31 | 17 | 100.0% |
| 0.4s | 825 | 3.51e-31 | 15 | 0.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 0.4s (Gen 825)
- Early phase (0-15s): 15 improvements
- Late phase (45-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## compound_fraction

**Final Result**: 3.11e-02 MSE in 49799 generations  
**Expression**: `1.0`

**Improvement Trajectory** (1 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 3.11e-02 | 1 |  |

---

## polynomial_product

**Final Result**: 0.00e+00 MSE in 33544 generations  
**Expression**: `((x0 + 1.0) * (2.0 + x1))`

**Improvement Trajectory** (9 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 4.73e+00 | 3 |  |
| 0.0s | 2 | 2.35e+00 | 5 | 50.3% |
| 0.0s | 19 | 2.32e+00 | 7 | 1.5% |
| 0.0s | 52 | 1.27e+00 | 13 | 45.0% |
| 0.0s | 54 | 5.79e-01 | 15 | 54.5% |
| 0.0s | 59 | 1.54e-31 | 13 | 100.0% |
| 0.0s | 69 | 1.54e-31 | 11 | 0.0% |
| 8.0s | 17245 | 1.01e-31 | 9 | 34.0% |
| 11.9s | 25977 | 0.00e+00 | 7 | 100.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 11.9s (Gen 25977)
- Early phase (0-15s): 9 improvements
- Late phase (45-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## surface_area_sphere_approx

**Final Result**: 0.00e+00 MSE in 34051 generations  
**Expression**: `((x0 + x0) * ((x0 + x0) * 2.0))`

**Improvement Trajectory** (8 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 7.83e+02 | 3 |  |
| 0.0s | 1 | 7.76e+02 | 5 | 1.0% |
| 0.0s | 2 | 4.97e+02 | 7 | 35.9% |
| 0.0s | 3 | 2.02e+02 | 9 | 59.3% |
| 0.0s | 6 | 2.82e+01 | 11 | 86.1% |
| 0.0s | 8 | 1.02e+00 | 11 | 96.4% |
| 0.0s | 10 | 0.00e+00 | 11 | 100.0% |
| 0.5s | 1030 | 0.00e+00 | 9 |  |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 0.5s (Gen 1030)
- Early phase (0-15s): 8 improvements
- Late phase (45-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

