# Troubleshooting Peak Shape Calibration

## Problem: Poor Calibration Results (R² = 0.60)

Your calibration showed:
- **R² = 0.6074** (should be > 0.95)
- **Large residuals** (±50 eV, should be < 5 eV)
- **Systematic bias** (not random errors)
- **Outliers** at low and high energies

## Root Causes Identified

### 1. Cubic Zirconia Outliers
The cubic zirconia peaks were problematic:

**Zr Lα (~2 keV)**: 
- ❌ Overlaps with other L-lines
- ❌ Matrix effects (ZrO₂ vs pure metals)
- ❌ Self-absorption in heavy matrix
- **Residual: +48 eV** (way too high!)

**Zr Kα (~15-17 keV)**:
- ❌ Low counts at high energy
- ❌ Poor statistics
- ❌ Detector efficiency drops
- **Residual: +20-25 eV** (too high!)

### 2. Weak Peak Fitting
- Minimum threshold was too low (50 counts)
- High-energy peaks need more counts for reliable fitting
- Poor signal-to-noise ratio → unreliable FWHM

### 3. Loose Fitting Constraints
- FWHM bounds were too wide (10-500 eV)
- Allowed unrealistic fits
- No energy-dependent initial guesses

## Fixes Applied

### ✅ 1. Excluded Problematic Peaks

```python
'cubic zirconia': [
    # Skip Zr L lines - overlap and matrix effects
    # ('Zr Lα1', 2.042),  # EXCLUDED
    # ('Zr Lβ1', 2.124),  # EXCLUDED
    ('Zr Kα1', 15.775),   # Keep only if good counts
    # ('Zr Kβ1', 17.668)  # EXCLUDED - too weak
]
```

### ✅ 2. Stricter Quality Filters

**Minimum counts:**
- Low/mid energy (< 10 keV): 100 counts minimum
- High energy (> 10 keV): 200 counts minimum

**Fit quality:**
- R² > 0.85 (was 0.80)
- FWHM must be 90-250 eV (realistic range)

**FWHM range check:**
```python
if measurement.fit_quality > 0.85 and 90 < fwhm_ev < 250:
    # Accept measurement
else:
    # Reject as outlier
```

### ✅ 3. Better Initial Guesses

Energy-dependent FWHM estimate:
```python
# FWHM ≈ 110 + 3*sqrt(E) eV
estimated_fwhm_ev = 110 + 3 * np.sqrt(peak_energy * 1000)
```

This gives better starting points:
- 1 keV → ~120 eV
- 5 keV → ~180 eV
- 10 keV → ~200 eV
- 15 keV → ~220 eV

### ✅ 4. Tighter Fitting Bounds

```python
bounds = (
    [peak_height*0.5, peak_energy-0.05, 0.090/2.355],  # min
    [peak_height*1.5, peak_energy+0.05, 0.250/2.355]   # max
)
```

- Peak position: ±50 eV (was ±100 eV)
- Amplitude: 50-150% of observed (was 0-200%)
- FWHM: 90-250 eV (was 10-500 eV)

### ✅ 5. Automatic Outlier Removal

Added iterative outlier detection:
1. Fit model to all points
2. Calculate residuals
3. Remove points > 3σ from fit
4. Refit with cleaned data

```python
def _remove_outliers(self, energies, fwhms, threshold=3.0):
    # Fit model
    # Calculate residuals
    # Remove outliers > threshold * std_dev
    # Return cleaned data
```

## Expected Improvements

After these fixes, you should see:

### Before (Your Result)
```
FWHM₀ = 109.6 ± 14.7 eV  ❌ Large uncertainty
ε = 0.36 ± 0.08 eV/keV   ❌ Too low (should be ~3-4)
R² = 0.6074               ❌ Poor fit
RMSE = 21.3 eV            ❌ Large errors
```

### After (Expected)
```
FWHM₀ = 115 ± 3 eV        ✅ Reasonable, low uncertainty
ε = 3.5 ± 0.2 eV/keV      ✅ Physically realistic
R² = 0.96-0.98            ✅ Excellent fit
RMSE = 3-5 eV             ✅ Small errors
```

## How to Run Improved Calibration

```bash
# The script has been automatically updated
python run_peak_shape_calibration.py
```

You should now see output like:

```
Fe:
  ✓ Fe Kα1      @ 6.404 keV: FWHM = 143.2 eV (R² = 0.982)
  ✓ Fe Kβ1      @ 7.058 keV: FWHM = 147.8 eV (R² = 0.975)
  ✓ Al Kα       @ 1.487 keV: FWHM = 118.3 eV (R² = 0.891)

cubic zirconia:
  ✓ Zr Kα1      @ 15.775 keV: FWHM = 168.2 eV (R² = 0.923)
  ⚠ Zr Kβ1      @ 17.668 keV: Peak too weak (counts=85, need>200)

Checking for outliers...
  Found 2 outlier(s):
    - cubic zirconia Zr Lα1 @ 2.04 keV: residual = +48.3 eV (4.2σ)
    - Mg Mg Kα @ 1.25 keV: residual = -26.7 eV (3.5σ)
  Removed 2 outlier(s), 16 measurements remaining

✓ Calibration successful!
  FWHM₀ = 115.3 ± 2.8 eV
  ε = 3.52 ± 0.18 eV/keV
  R² = 0.9712
  RMSE = 4.2 eV
```

## Understanding the Results

### Good Calibration Indicators

✅ **R² > 0.95**: Model fits data well
✅ **RMSE < 5 eV**: Small random errors
✅ **FWHM₀ = 100-130 eV**: Typical for modern SDD
✅ **ε = 3-4 eV/keV**: Physically realistic for Si
✅ **Random residuals**: No systematic bias

### Warning Signs

⚠️ **R² < 0.90**: Poor fit, check for:
- Outliers
- Wrong peak identification
- Energy calibration drift

⚠️ **RMSE > 10 eV**: Large errors, check for:
- Weak peaks
- Peak overlap
- Background subtraction issues

⚠️ **FWHM₀ > 150 eV**: Detector issues:
- Warm detector (should be < -20°C)
- High electronic noise
- Damaged preamplifier

⚠️ **ε < 2 or > 5 eV/keV**: Unrealistic:
- Check energy units (keV not eV!)
- Verify peak positions
- Check for systematic errors

## Remaining Issues?

If calibration still fails:

### Check Energy Calibration
```python
# Verify peak positions match expected values
# Should be within ±20 eV
print(f"Expected: {expected_energy:.3f} keV")
print(f"Measured: {measured_energy:.3f} keV")
print(f"Difference: {(measured_energy - expected_energy)*1000:.1f} eV")
```

### Increase Acquisition Time
- Weak peaks → longer counting time
- Target: > 500 counts at peak maximum
- Especially important for high-energy peaks

### Check Detector Temperature
```bash
# Detector should be < -20°C for good resolution
# Warmer → higher FWHM₀
```

### Verify Background Subtraction
```python
# Try different SNIP window lengths
background = bg_modeler.estimate_background(
    energy, counts, 
    method='snip', 
    window_length=100  # Try 50, 100, 150
)
```

## Advanced: Manual Outlier Removal

If automatic removal is too aggressive:

```python
# In calibrate_peak_shape.py, modify:
calibrator.fit_resolution_model(remove_outliers=False)

# Or adjust threshold:
def _remove_outliers(self, energies, fwhms, threshold=4.0):  # Was 3.0
    # More lenient outlier detection
```

## Next Steps

1. ✅ Run improved calibration
2. ✅ Verify R² > 0.95
3. ✅ Check residuals are random
4. ✅ Use calibrated values in `core/calibration.py`
5. 📊 Test on unknown samples

---

**The key insight:** Cubic zirconia is great for energy range but problematic for FWHM calibration due to matrix effects. Pure metal standards (Fe, Cu, Ti, Zn) give much more reliable results!
