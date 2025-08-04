# Improvement Trajectory Analysis

Each problem was run for 1 minute to track how the algorithm discovers mathematical structure over time.

## pythagorean_3d

**Final Result**: 7.38e-31 MSE in 10428 generations  
**Expression**: `(((x2 * x2) + (x0 * x0)) + (x1 * x1))`

**Improvement Trajectory** (22 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 4.33e+01 | 3 |  |
| 0.0s | 1 | 3.29e+01 | 5 | 23.9% |
| 0.0s | 2 | 3.03e+01 | 7 | 8.1% |
| 0.0s | 3 | 2.11e+01 | 9 | 30.3% |
| 0.0s | 4 | 9.11e+00 | 11 | 56.8% |
| 0.0s | 5 | 6.32e+00 | 9 | 30.6% |
| 0.1s | 20 | 6.20e+00 | 19 | 2.0% |
| 0.1s | 23 | 6.18e+00 | 19 | 0.2% |
| 0.1s | 24 | 5.93e+00 | 17 | 4.1% |
| 0.1s | 27 | 5.65e+00 | 19 | 4.7% |
| 0.2s | 34 | 5.64e+00 | 19 | 0.0% |
| 0.2s | 36 | 5.35e+00 | 17 | 5.2% |
| 0.2s | 39 | 5.06e+00 | 17 | 5.5% |
| 0.3s | 42 | 5.06e+00 | 15 | 0.0% |
| 0.8s | 130 | 3.45e+00 | 17 | 31.7% |
| 0.9s | 138 | 2.11e+00 | 17 | 38.9% |
| 0.9s | 141 | 7.63e-01 | 19 | 63.9% |
| 0.9s | 143 | 2.50e-01 | 17 | 67.2% |
| 0.9s | 144 | 7.38e-31 | 19 | 100.0% |
| 0.9s | 148 | 7.38e-31 | 17 | 0.0% |
| 1.0s | 150 | 7.38e-31 | 15 | 0.0% |
| 1.0s | 153 | 7.38e-31 | 11 | 0.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 1.0s (Gen 153)
- Early phase (0-15s): 22 improvements
- Late phase (45-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## quadratic_formula_discriminant

**Final Result**: 9.56e-32 MSE in 10260 generations  
**Expression**: `((x1 * x1) + (((x2 - ((x0 + (x2 + x0)) + x0)) - x0) * x2))`

**Improvement Trajectory** (13 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 1.93e+01 | 1 |  |
| 0.0s | 1 | 1.88e+01 | 9 | 2.8% |
| 0.0s | 2 | 1.86e+01 | 3 | 1.0% |
| 0.0s | 5 | 1.23e+01 | 5 | 33.5% |
| 0.0s | 11 | 9.60e+00 | 9 | 22.2% |
| 0.0s | 13 | 6.60e+00 | 7 | 31.2% |
| 0.1s | 15 | 3.49e+00 | 9 | 47.1% |
| 0.1s | 17 | 3.01e+00 | 17 | 13.9% |
| 0.1s | 18 | 2.70e+00 | 15 | 10.1% |
| 0.1s | 21 | 1.95e+00 | 15 | 27.9% |
| 0.1s | 23 | 1.95e+00 | 13 | 0.0% |
| 0.1s | 24 | 1.43e+00 | 15 | 26.7% |
| 0.3s | 53 | 9.56e-32 | 17 | 100.0% |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 0.3s (Gen 53)
- Early phase (0-15s): 13 improvements
- Late phase (45-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## compound_fraction

**Final Result**: 3.11e-02 MSE in 21761 generations  
**Expression**: `1.0`

**Improvement Trajectory** (1 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 3.11e-02 | 1 |  |

---

## polynomial_product

**Final Result**: 0.00e+00 MSE in 10444 generations  
**Expression**: `((x1 + 2.0) * (x0 + 1.0))`

**Improvement Trajectory** (7 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 4.73e+00 | 3 |  |
| 0.0s | 1 | 2.35e+00 | 5 | 50.3% |
| 0.0s | 2 | 9.51e-01 | 11 | 59.6% |
| 0.0s | 5 | 5.01e-01 | 11 | 47.3% |
| 0.0s | 7 | 0.00e+00 | 11 | 100.0% |
| 0.0s | 8 | 0.00e+00 | 9 |  |
| 0.1s | 15 | 0.00e+00 | 7 |  |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 0.1s (Gen 15)
- Early phase (0-15s): 7 improvements
- Late phase (45-60s): 0 improvements
- **Pattern**: Front-loaded discovery (most progress early)

---

## surface_area_sphere_approx

**Final Result**: 0.00e+00 MSE in 9387 generations  
**Expression**: `((x0 + x0) * (2.0 * (x0 + x0)))`

**Improvement Trajectory** (15 improvements):

| Time | Generation | MSE | Size | Improvement |
|------|------------|-----|------|-------------|
| 0.0s | 0 | 1.87e+01 | 13 |  |
| 0.0s | 1 | 1.10e+01 | 13 | 41.3% |
| 0.0s | 2 | 5.95e+00 | 15 | 45.7% |
| 0.0s | 5 | 8.98e-01 | 17 | 84.9% |
| 0.0s | 8 | 8.98e-01 | 17 | 0.0% |
| 0.3s | 54 | 8.98e-01 | 17 | 0.0% |
| 0.6s | 105 | 5.16e-01 | 19 | 42.5% |
| 1.3s | 203 | 4.65e-01 | 19 | 10.0% |
| 54.1s | 8397 | 4.65e-01 | 17 | -0.0% |
| 54.9s | 8536 | 4.65e-01 | 15 | -0.0% |
| 55.5s | 8627 | 3.81e-01 | 19 | 18.0% |
| 55.5s | 8630 | 1.40e-01 | 19 | 63.3% |
| 55.5s | 8631 | 0.00e+00 | 15 | 100.0% |
| 58.3s | 9081 | 0.00e+00 | 11 |  |
| 58.3s | 9084 | 0.00e+00 | 9 |  |

**Analysis**:
- Total improvement: 100.0% MSE reduction
- First improvement at 0.0s (Gen 0)
- Last improvement at 58.3s (Gen 9084)
- Early phase (0-15s): 8 improvements
- Late phase (45-60s): 7 improvements
- **Pattern**: Steady progress throughout run

---

