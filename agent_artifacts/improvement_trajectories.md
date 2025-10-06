# Improvement Trajectory Analysis

Each problem was run for 60 seconds using BasicSR with built-in time limits and MSE early stopping (≤ 3e-16).

## pythagorean_3d

**Final Result**: 7.45e-31 MSE in 0 generations  
**Actual Time**: 0.3s (limit: 60s)  
**Expression**: `((x0 * x0) + ((x1 * x1) + (x2 * x2)))`

**Improvement Trajectory** (14 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.3s | 0 | 4.33e+01 | 3 |  |
| 0.3s | 1 | 2.46e+01 | 5 | 43.2% |
| 0.3s | 4 | 1.78e+01 | 7 | 27.7% |
| 0.3s | 6 | 8.96e+00 | 9 | 49.6% |
| 0.3s | 15 | 8.13e+00 | 11 | 9.2% |
| 0.3s | 17 | 7.54e+00 | 11 | 7.2% |
| 0.3s | 19 | 7.23e+00 | 13 | 4.1% |
| 0.3s | 20 | 6.89e+00 | 13 | 4.7% |
| 0.3s | 168 | 4.00e+00 | 15 | 42.0% |
| 0.3s | 170 | 3.21e+00 | 17 | 19.8% |
| 0.3s | 172 | 3.21e+00 | 15 | 0.0% |
| 0.3s | 173 | 1.00e+00 | 17 | 68.8% |
| 0.3s | 177 | 1.00e+00 | 15 | 0.0% |
| 0.3s | 188 | 0.00e+00 | 11 | 100.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.3s (Gen 0)
- Last improvement at 0.3s (Gen 188)
- **Early stopping**: MSE reached near-zero (7.45e-31)
- Early phase (0-20s): 14 improvements
- Late phase (40-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## quadratic_formula_discriminant

**Final Result**: 3.21e-31 MSE in 0 generations  
**Actual Time**: 1.3s (limit: 60s)  
**Expression**: `(x0 * ((((((x1 / x0) * x1) - x2) - x2) - x2) - x2))`

**Improvement Trajectory** (19 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 1.3s | 0 | 1.93e+01 | 1 |  |
| 1.3s | 2 | 1.86e+01 | 5 | 3.7% |
| 1.3s | 8 | 1.86e+01 | 3 | 0.0% |
| 1.3s | 35 | 1.31e+01 | 7 | 29.3% |
| 1.3s | 36 | 1.12e+01 | 9 | 14.5% |
| 1.3s | 39 | 8.53e+00 | 15 | 24.0% |
| 1.3s | 40 | 6.97e+00 | 11 | 18.3% |
| 1.3s | 42 | 6.97e+00 | 9 | 0.0% |
| 1.3s | 43 | 5.79e+00 | 17 | 16.9% |
| 1.3s | 44 | 3.45e+00 | 11 | 40.5% |
| 1.3s | 49 | 3.45e+00 | 9 | 0.0% |
| 1.3s | 53 | 2.56e+00 | 13 | 25.9% |
| 1.3s | 87 | 2.56e+00 | 11 | 0.0% |
| 1.3s | 205 | 1.84e+00 | 13 | 28.1% |
| 1.3s | 223 | 1.43e+00 | 13 | 22.3% |
| 1.3s | 477 | 1.36e+00 | 15 | 4.8% |
| 1.3s | 569 | 1.36e+00 | 15 | 0.0% |
| 1.3s | 664 | 3.29e-01 | 17 | 75.8% |
| 1.3s | 668 | 0.00e+00 | 15 | 100.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 1.3s (Gen 0)
- Last improvement at 1.3s (Gen 668)
- **Early stopping**: MSE reached near-zero (3.21e-31)
- Early phase (0-20s): 19 improvements
- Late phase (40-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## mixed_polynomial

**Final Result**: 1.71e-31 MSE in 0 generations  
**Actual Time**: 0.2s (limit: 60s)  
**Expression**: `((x1 * (x1 - x0)) + ((1.0 * x0) * x0))`

**Improvement Trajectory** (12 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.2s | 0 | 6.72e+00 | 3 |  |
| 0.2s | 1 | 4.84e+00 | 5 | 27.9% |
| 0.2s | 3 | 4.12e+00 | 9 | 15.0% |
| 0.2s | 4 | 1.77e+00 | 9 | 57.0% |
| 0.2s | 9 | 1.77e+00 | 7 | 0.0% |
| 0.2s | 10 | 1.73e+00 | 7 | 2.3% |
| 0.2s | 83 | 1.50e+00 | 15 | 13.2% |
| 0.2s | 87 | 1.42e+00 | 17 | 5.7% |
| 0.2s | 91 | 1.35e+00 | 17 | 4.6% |
| 0.2s | 92 | 1.35e+00 | 15 | 0.0% |
| 0.2s | 107 | 8.97e-01 | 19 | 33.5% |
| 0.2s | 109 | 0.00e+00 | 11 | 100.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.2s (Gen 0)
- Last improvement at 0.2s (Gen 109)
- **Early stopping**: MSE reached near-zero (1.71e-31)
- Early phase (0-20s): 12 improvements
- Late phase (40-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## polynomial_product

**Final Result**: 1.49e-31 MSE in 0 generations  
**Actual Time**: 0.0s (limit: 60s)  
**Expression**: `(x0 + (((x0 + x1) + 2.0) + (((x0 + x1) - x0) * x0)))`

**Improvement Trajectory** (6 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 4.73e+00 | 3 |  |
| 0.0s | 1 | 2.35e+00 | 5 | 50.3% |
| 0.0s | 13 | 2.32e+00 | 7 | 1.5% |
| 0.0s | 15 | 1.35e+00 | 15 | 41.6% |
| 0.0s | 19 | 1.35e+00 | 13 | 0.0% |
| 0.0s | 21 | 0.00e+00 | 15 | 100.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 0.0s (Gen 21)
- **Early stopping**: MSE reached near-zero (1.49e-31)
- Early phase (0-20s): 6 improvements
- Late phase (40-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## surface_area_sphere_approx

**Final Result**: 4.65e-01 MSE in 0 generations  
**Actual Time**: 21.9s (limit: 60s)  
**Expression**: `(((x0 * ((2.0 + 2.0) + x0)) * x0) + (((x0 - 1.0) + (2.0 * x0)) + x0))`

**Improvement Trajectory** (11 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 21.9s | 0 | 1.41e+02 | 19 |  |
| 21.9s | 1 | 1.14e+02 | 9 | 19.1% |
| 21.9s | 3 | 7.65e+01 | 15 | 32.7% |
| 21.9s | 5 | 7.86e+00 | 17 | 89.7% |
| 21.9s | 8 | 2.55e+00 | 19 | 67.5% |
| 21.9s | 10 | 1.61e+00 | 19 | 36.9% |
| 21.9s | 11 | 1.57e+00 | 19 | 2.4% |
| 21.9s | 18 | 1.02e+00 | 17 | 35.1% |
| 21.9s | 19 | 8.98e-01 | 17 | 12.0% |
| 21.9s | 44 | 8.98e-01 | 17 | 0.0% |
| 21.9s | 673 | 4.65e-01 | 19 | 48.3% |

**Analysis**:
- Total improvement: 99.7% MSE reduction
- First improvement at 21.9s (Gen 0)
- Last improvement at 21.9s (Gen 673)
- Early phase (0-20s): 0 improvements
- Late phase (40-60s): 0 improvements
- **Pattern**: Steady progress throughout run

---

