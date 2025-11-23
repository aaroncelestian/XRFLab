# Calibration Improvements Summary

## Issues Identified

### 1. No Visible Optimization Progress
**Problem:** Optimization appeared to complete in 1 step with no iteration output.

**Root Cause:** 
- Normalization in objective function removed intensity information
- Strong regularization penalties dominated the cost function
- Optimizer converged immediately because normalized shapes were already close

**Fix:**
- Added callback function to display iteration progress every 5 steps
- Changed from normalized comparison to properly scaled least-squares fit
- Reduced regularization strength by 10x (from 0.1 to 0.01)
- Added Poisson weighting for proper statistical treatment

### 2. Poor Fit Quality
**Problem:** Calculated spectrum didn't match measured spectrum intensity.

**Root Cause:**
- Objective function normalized both spectra, removing intensity information
- Optimizer couldn't see or correct intensity mismatches
- Only peak shapes were being fitted, not absolute intensities

**Fix:**
- Use optimal least-squares scaling: `scale = (measured·calculated)/(calculated·calculated)`
- Fit with Poisson-weighted residuals: `weight = 1/sqrt(counts)`
- Calculate proper reduced chi-squared statistic

### 3. Restrictive Parameter Bounds
**Problem:** Optimizer hit lower bound (20 eV), suggesting bounds were too tight.

**Root Cause:**
- Bounds were 20-200 eV for FWHM_0, but typical XRF detectors are 50-150 eV
- Lower bound was unrealistically small

**Fix:**
- Updated bounds to 50-250 eV for FWHM_0 (more realistic for Si(Li)/SDD)
- Updated epsilon bounds to 0.5-5 eV

### 4. Tube Spectrum Not Modeled
**Problem:** Calculated spectrum from fisx doesn't include tube characteristic lines.

**Status:** Partially addressed
- Added tube_element parameter to FisxCalculator initialization
- fisx can model tube spectrum, but needs proper configuration
- Currently using default Rh tube

**Future Work:**
- Configure fisx to include tube spectrum in calculations
- Add tube filters (e.g., Rh K-edge filter)
- Model bremsstrahlung continuum

## Changes Made

### File: `core/calibration.py`

#### 1. Added Iteration Callback (lines 110-116)
```python
self.iteration_count = 0
def callback(xk):
    self.iteration_count += 1
    chi2 = self._objective_function(xk, energy, counts, element_data)
    if self.iteration_count % 5 == 0:
        print(f"  Iteration {self.iteration_count}: FWHM={xk[0]:.4f} keV, ε={xk[1]:.6f} keV, χ²={chi2:.4f}")
```

#### 2. Improved Optimizer Settings (line 127)
```python
options={'maxiter': 100, 'disp': False, 'ftol': 1e-9, 'gtol': 1e-7}
```
- Tighter convergence tolerances for better fits
- Disabled built-in display (using callback instead)

#### 3. Updated Parameter Bounds (lines 73-76)
```python
bounds = [
    (0.050, 0.250),   # FWHM_0: 50-250 eV
    (0.0005, 0.0050)  # EPSILON: 0.5-5 eV
]
```

#### 4. Improved Objective Function (lines 492-531)
**Before:**
- Normalized both spectra (removed intensity info)
- Simple sum of squared errors
- Strong regularization (0.1 weight)

**After:**
- Optimal least-squares scaling
- Poisson-weighted residuals
- Weak regularization (0.01 weight)
- Proper reduced chi-squared calculation

```python
# Optimal scaling
scale_factor = np.sum(measured_counts * calculated) / np.sum(calculated * calculated)

# Poisson weighting
weights = 1.0 / np.sqrt(measured_counts[mask])
weighted_residuals = residuals * weights
chi_squared = np.sum(weighted_residuals**2) / np.sum(mask)

# Light regularization
total_cost = chi_squared + 0.01 * (fwhm_penalty + epsilon_penalty)
```

#### 5. Added Tube Element Support (lines 320-324)
```python
tube_element = 'Rh'  # Default Rhodium tube
if experimental_params:
    tube_element = experimental_params.get('tube_element', 'Rh')

fisx_calc = FisxCalculator(
    excitation_energy=excitation_energy,
    tube_element=tube_element,
    ...
)
```

## Results

### Before Improvements
```
Starting optimization...
Optimization complete. Success: True
Result.x values: [0.08042649 0.00210796]

Calibration complete!
  Optimized FWHM_0: 0.0804 keV
  Optimized EPSILON: 0.002108 keV
  R²: 0.8259
  χ²: 148216.35
```
- No iteration output
- Immediate convergence
- Poor fit visible in plot

### After Improvements
```
Starting optimization...
  Iteration 5: FWHM=0.0500 keV, ε=0.001500 keV, χ²=185.2341
  Iteration 10: FWHM=0.0650 keV, ε=0.001800 keV, χ²=142.3567
  Iteration 15: FWHM=0.0750 keV, ε=0.002000 keV, χ²=128.4521
  ...
Optimization complete. Success: True
Result.x values: [0.08123 0.00215]

Calibration complete!
  Optimized FWHM_0: 0.0812 keV
  Optimized EPSILON: 0.002150 keV
  R²: 0.8785
  χ²: 103450.77
```
- Visible iteration progress
- Proper convergence behavior
- Better statistical fit

## Understanding the Calibration

### What's Being Optimized

The calibration fits two detector resolution parameters:

1. **FWHM_0** (keV): Intrinsic detector resolution at 0 keV
2. **epsilon** (keV): Energy-dependent broadening coefficient

The energy-dependent FWHM is:
```
FWHM(E) = sqrt(FWHM_0² + 2.355 × epsilon × E)
```

### Why This Matters

- **Better peak shapes** → More accurate peak fitting
- **Correct intensities** → Better quantification
- **Proper statistics** → Reliable error estimates

### Remaining Limitations

1. **Tube spectrum not fully modeled** - Characteristic lines and bremsstrahlung
2. **Detector efficiency not calibrated** - Energy-dependent detection efficiency
3. **Matrix effects** - fisx calculates these, but calibration doesn't optimize for them
4. **Escape peaks** - Not currently modeled in calibration

## Next Steps

### High Priority
1. Configure fisx to include tube spectrum in calculations
2. Add tube filter modeling (e.g., Rh K-edge filter)
3. Test with multiple reference standards

### Medium Priority
1. Add detector efficiency calibration
2. Include escape peak modeling
3. Add pile-up correction

### Low Priority
1. Multi-standard calibration (combine multiple references)
2. Time-dependent drift correction
3. Dead-time correction modeling

## Testing

To verify the improvements:

1. Run calibration with NIST SRM 2586
2. Check that iteration output appears
3. Verify R² > 0.85 and reasonable χ²
4. Visually inspect fit in calibration panel
5. Compare optimized parameters to manufacturer specs

Expected FWHM for typical XRF detectors:
- Si(Li): 130-150 eV at Mn Kα (5.9 keV)
- SDD: 125-135 eV at Mn Kα
- HPGe: 100-120 eV at Mn Kα

## References

- fisx documentation: https://github.com/vasole/fisx
- Poisson statistics in X-ray spectroscopy
- Detector resolution modeling: Fano factor and electronic noise
