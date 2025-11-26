# Peak Shape Model Comparison Guide

## Overview

Your detector's resolution (FWHM) varies with energy. The question is: **which mathematical model best describes this relationship?**

This guide helps you compare different models and choose the best one for your data.

## Available Models

### 1. **Detector Model** (Standard Physics-Based)
```
FWHM(E) = √(FWHM₀² + 2.355² · ε · E)
```

**Parameters:**
- `FWHM₀`: Electronic noise (eV) - independent of energy
- `ε` (epsilon): Fano factor contribution (eV/keV) - scales with √E

**Physical Basis:**
- Based on detector physics (Fano statistics)
- FWHM₀ from electronic noise
- ε from statistical fluctuations in charge generation
- Standard for Si detectors

**When to use:**
- ✅ Default choice for Si detectors
- ✅ Physically interpretable parameters
- ✅ Expected for well-behaved detectors

### 2. **Linear Model** (Simplified)
```
FWHM(E) = a + b·E
```

**Parameters:**
- `a`: Intercept (eV)
- `b`: Slope (eV/keV)

**Physical Basis:**
- Simplified approximation of detector model
- Works well over limited energy ranges
- Less physically meaningful

**When to use:**
- ✅ Quick approximation
- ✅ Limited energy range
- ⚠️ Loses physical interpretation

### 3. **Quadratic Model** (Extended)
```
FWHM(E) = a + b·E + c·E²
```

**Parameters:**
- `a`: Intercept (eV)
- `b`: Linear coefficient (eV/keV)
- `c`: Quadratic coefficient (eV/keV²)

**Physical Basis:**
- Can capture non-standard behavior
- May indicate detector artifacts
- More flexible but less interpretable

**When to use:**
- ⚠️ If detector model fails
- ⚠️ May indicate detector issues
- ⚠️ Overfitting risk with few points

### 4. **Exponential Model**
```
FWHM(E) = a · exp(b·E)
```

**Parameters:**
- `a`: Amplitude (eV)
- `b`: Exponent (keV⁻¹)

**Physical Basis:**
- No clear physical basis for detectors
- Empirical fit only

**When to use:**
- ⚠️ Unusual for detectors
- ⚠️ May indicate systematic problems
- ❌ Not recommended unless detector model fails badly

### 5. **Power Law Model**
```
FWHM(E) = a · E^b
```

**Parameters:**
- `a`: Amplitude (eV)
- `b`: Power

**Physical Basis:**
- Empirical relationship
- Can approximate detector model if b ≈ 0.5

**When to use:**
- ⚠️ Empirical fit
- ⚠️ Less interpretable
- ✅ May work if b ≈ 0.5 (similar to detector model)

## Running Model Comparison

### Quick Start

```bash
python compare_calibration_models.py
```

This will:
1. Load all your calibration data
2. Fit all 5 models
3. Compare using statistical criteria
4. Generate comparison plots
5. Recommend the best model

### Expected Output

```
MODEL COMPARISON SUMMARY
======================================================================
Model           R²         RMSE (eV)    AIC        BIC        Rank
----------------------------------------------------------------------
⭐ detector     0.9712     4.23         -85.32     -82.15     #1
   linear       0.9645     5.12         -81.45     -78.28     #2
   quadratic    0.9723     4.89         -79.23     -74.89     #3
   power        0.9598     5.45         -78.12     -74.95     #4
   exponential  0.9512     6.01         -75.34     -72.17     #5

RECOMMENDATIONS
======================================================================
⭐ Best model: DETECTOR
   R² = 0.9712
   RMSE = 4.23 eV
```

## Model Selection Criteria

### AIC (Akaike Information Criterion)
**Lower is better**

- Balances fit quality with model complexity
- Penalizes extra parameters
- **ΔAIC interpretation:**
  - < 2: Models essentially equivalent
  - 2-10: Substantial support for lower AIC
  - > 10: Strong support for lower AIC

### BIC (Bayesian Information Criterion)
**Lower is better**

- Similar to AIC but penalizes complexity more strongly
- Prefers simpler models
- Good for avoiding overfitting

### R² (Coefficient of Determination)
**Higher is better (0-1)**

- Measures fraction of variance explained
- **Doesn't penalize complexity!**
- Can be misleading with complex models

### RMSE (Root Mean Square Error)
**Lower is better (in eV)**

- Average prediction error
- Direct measure of fit quality
- In same units as FWHM (eV)

## Interpretation Guide

### Case 1: Detector Model is Best ✅
```
⭐ detector     R² = 0.97    RMSE = 4.2 eV    AIC = -85.3
```

**Interpretation:**
- ✅ Your detector behaves as expected
- ✅ Physical parameters are meaningful
- ✅ Use FWHM₀ and ε for calibration

**Action:**
- Use detector model results
- Parameters have physical meaning
- All is well!

### Case 2: Linear Model is Best ⚠️
```
⭐ linear       R² = 0.96    RMSE = 5.1 eV    AIC = -81.5
   detector     R² = 0.95    RMSE = 5.8 eV    AIC = -78.2  (ΔAIC = 3.3)
```

**Interpretation:**
- ⚠️ Simplified model fits better
- May indicate limited energy range
- Detector model still has support (ΔAIC < 10)

**Action:**
- Check energy range (< 10 keV?)
- If ΔAIC < 2: Use detector model anyway (physically meaningful)
- If ΔAIC > 2: Linear is simpler and fits better

### Case 3: Quadratic/Higher Order is Best ⚠️
```
⭐ quadratic    R² = 0.98    RMSE = 3.5 eV    AIC = -88.1
   detector     R² = 0.95    RMSE = 5.2 eV    AIC = -79.3  (ΔAIC = 8.8)
```

**Interpretation:**
- ⚠️ Non-standard detector behavior
- May indicate:
  - Incomplete charge collection
  - Detector artifacts
  - Energy calibration issues
  - Sample effects (self-absorption)

**Action:**
- Check detector health
- Verify energy calibration
- Exclude problematic peaks
- Use quadratic if detector model fails badly (ΔAIC > 10)

### Case 4: All Models Similar
```
⭐ detector     R² = 0.96    RMSE = 4.8 eV    AIC = -82.1
   linear       R² = 0.96    RMSE = 4.9 eV    AIC = -81.8  (ΔAIC = 0.3)
   power        R² = 0.96    RMSE = 5.0 eV    AIC = -81.5  (ΔAIC = 0.6)
```

**Interpretation:**
- All models fit equally well (ΔAIC < 2)
- Limited energy range or few data points

**Action:**
- **Always choose detector model** when models are equivalent
- Physical interpretation is more valuable
- Simpler is better

## Practical Examples

### Example 1: Good Detector
```python
# Your data spans 1-16 keV with good statistics
# Expected: Detector model wins

Results:
  detector:  R² = 0.972, RMSE = 4.2 eV, AIC = -85.3  ⭐
  linear:    R² = 0.965, RMSE = 5.1 eV, AIC = -81.5
  
→ Use detector model
→ FWHM₀ = 115 eV, ε = 3.5 eV/keV
```

### Example 2: Limited Range
```python
# Your data only spans 4-9 keV
# Expected: Linear and detector similar

Results:
  linear:    R² = 0.968, RMSE = 4.5 eV, AIC = -83.2  ⭐
  detector:  R² = 0.965, RMSE = 4.7 eV, AIC = -82.9  (ΔAIC = 0.3)
  
→ Use detector model (ΔAIC < 2, physically meaningful)
→ Or use linear for simplicity
```

### Example 3: Detector Issues
```python
# Systematic deviations from detector model
# Expected: Quadratic or higher order wins

Results:
  quadratic: R² = 0.982, RMSE = 3.8 eV, AIC = -88.5  ⭐
  detector:  R² = 0.951, RMSE = 6.2 eV, AIC = -76.3  (ΔAIC = 12.2)
  
→ Investigate detector issues!
→ Check for:
   - Energy calibration drift
   - Detector artifacts
   - Sample effects
→ Use quadratic temporarily, but fix root cause
```

## Recommendations

### General Guidelines

1. **Start with detector model**
   - Physically motivated
   - Standard for Si detectors
   - Interpretable parameters

2. **Use AIC/BIC for comparison**
   - Don't just use R²
   - Penalizes overfitting
   - Balances complexity

3. **Check ΔAIC**
   - < 2: Models equivalent → choose simpler/physical
   - 2-10: Some support for better model
   - > 10: Strong support → use better model

4. **Physical interpretation matters**
   - If detector model is close (ΔAIC < 5), use it
   - Physical parameters are valuable
   - Helps diagnose detector issues

5. **Investigate anomalies**
   - If detector model fails badly, investigate why
   - May indicate real problems
   - Don't just accept complex model

### For Your Data

Based on your previous calibration (R² = 0.60), you should:

1. ✅ Run model comparison
2. ✅ Check if any model fits well (R² > 0.95)
3. ✅ If all fail: Investigate outliers (cubic zirconia L-lines)
4. ✅ After removing outliers: Detector model should win
5. ✅ Expected: FWHM₀ ≈ 110-120 eV, ε ≈ 3-4 eV/keV

## Next Steps

```bash
# 1. Run model comparison
python compare_calibration_models.py

# 2. Review results
#    - Check which model wins
#    - Look at ΔAIC values
#    - Examine residual plots

# 3. Use best model
#    - If detector model: Use FWHM₀ and ε
#    - If other model: Investigate why

# 4. Apply to analysis
#    - Update core/calibration.py
#    - Use calibrated parameters
```

## Troubleshooting

### All models fit poorly (R² < 0.90)
- ❌ Check for outliers
- ❌ Verify energy calibration
- ❌ Check peak identification
- ❌ Increase minimum counts threshold

### Exponential/Power law wins
- ⚠️ Unusual for detectors
- ⚠️ Check for systematic errors
- ⚠️ May indicate detector problems
- ⚠️ Verify data quality

### Large RMSE (> 10 eV)
- ❌ Poor fit quality
- ❌ Check background subtraction
- ❌ Verify peak fitting
- ❌ Exclude weak peaks

---

**Remember:** The detector model should win for well-behaved Si detectors. If it doesn't, investigate why before accepting a more complex model!
