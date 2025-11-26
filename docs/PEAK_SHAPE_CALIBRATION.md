# Peak Shape vs Energy Calibration

## Overview

This calibration determines how detector resolution (FWHM) varies with photon energy using pure element standards. The relationship follows:

```
FWHM(E) = √(FWHM₀² + 2.355² · ε · E)
```

Where:
- **FWHM₀**: Electronic noise contribution (eV) - independent of energy
- **ε** (epsilon): Fano factor contribution (eV/keV) - scales with energy
- **E**: Photon energy (keV)

## Standards Used

Your new XRF data includes excellent calibration standards:

| File | Element | Major Peaks | Energy Range |
|------|---------|-------------|--------------|
| `Fe.txt` | Iron | Fe Kα (6.40 keV), Fe Kβ (7.06 keV) | Mid-range |
| `Cu.txt` | Copper | Cu Kα (8.05 keV), Cu Kβ (8.91 keV) | High-range |
| `Ti.txt` | Titanium | Ti Kα (4.51 keV), Ti Kβ (4.93 keV) | Mid-range |
| `Zn.txt` | Zinc | Zn Kα (8.64 keV), Zn Kβ (9.57 keV) | High-range |
| `Mg.txt` | Magnesium | Mg Kα (1.25 keV) | Low-range |
| `cubic zirconia.txt` | Zirconium | Zr Lα (2.04 keV), Zr Kα (15.75 keV) | Wide range |

**Note**: All samples except cubic zirconia also contain **Al Kα (1.49 keV)** from the aluminum sample holder, providing additional low-energy calibration points.

## Running the Calibration

### Quick Start

```bash
python run_peak_shape_calibration.py
```

This will:
1. Load all standard spectra from `sample_data/data/`
2. Subtract background using SNIP algorithm
3. Fit Gaussian profiles to each characteristic peak
4. Extract FWHM values at different energies
5. Fit the detector resolution model
6. Save results to `sample_data/peak_shape_calibration.json`
7. Generate plot: `sample_data/peak_shape_calibration.png`

### Expected Output

```
Processing XRF standards for peak shape calibration...
======================================================================

Fe:
  ✓ Fe Kα1      @ 6.404 keV: FWHM = 145.2 eV (R² = 0.982)
  ✓ Fe Kβ1      @ 7.058 keV: FWHM = 148.7 eV (R² = 0.975)
  ✓ Al Kα       @ 1.487 keV: FWHM = 128.3 eV (R² = 0.891)

Cu:
  ✓ Cu Kα1      @ 8.048 keV: FWHM = 151.4 eV (R² = 0.988)
  ✓ Cu Kβ1      @ 8.905 keV: FWHM = 153.9 eV (R² = 0.981)
  ...

======================================================================
Total successful measurements: 18

Fitting detector resolution model...
======================================================================

✓ Calibration successful!
  FWHM₀ = 115.3 ± 2.1 eV
  ε = 3.45 ± 0.15 eV/keV
  R² = 0.9876
  RMSE = 2.3 eV

Example FWHM predictions:
   1.5 keV →  118.2 eV
   5.0 keV →  131.7 eV
  10.0 keV →  147.8 eV
  15.0 keV →  161.4 eV
```

## Understanding the Results

### FWHM₀ (Electronic Noise)
- Typical range: 80-150 eV for modern SDDs
- Lower is better (less electronic noise)
- Independent of photon energy
- Dominated by:
  - Preamplifier noise
  - Detector capacitance
  - Temperature (should be cooled to -30°C or lower)

### ε (Epsilon - Fano Factor)
- Typical range: 2-5 eV/keV for Si detectors
- Related to statistical fluctuations in charge carrier generation
- Theoretical Fano factor for Si: F ≈ 0.12
- ε ≈ 2.355 × √(F × w × E) where w = 3.65 eV (Si ionization energy)
- Expected: ε ≈ 3.5 eV/keV

### Fit Quality
- **R² > 0.95**: Excellent fit
- **R² = 0.90-0.95**: Good fit
- **R² < 0.90**: Check for:
  - Peak overlap issues
  - Poor background subtraction
  - Detector artifacts
  - Energy calibration drift

## Using Calibration Results

### In Python Code

```python
import json
import numpy as np

# Load calibration
with open('sample_data/peak_shape_calibration.json', 'r') as f:
    cal = json.load(f)

fwhm_0 = cal['fwhm_0_keV']
epsilon = cal['epsilon_keV']

# Predict FWHM at any energy
def predict_fwhm(energy_keV):
    """Predict FWHM in keV at given energy"""
    return np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * energy_keV)

# Example: FWHM at 6.4 keV (Fe Kα)
fwhm_fe = predict_fwhm(6.4)
print(f"Fe Kα FWHM: {fwhm_fe*1000:.1f} eV")
```

### Updating Calibration Module

The calibration results should be used in `core/calibration.py`:

```python
# In _calculate_spectrum method, line ~569:
fwhm = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * line_energy)
```

Replace the initial guesses with your calibrated values:

```python
# In calibrate method, line ~78:
p0 = [
    0.115,      # FWHM_0 from calibration (keV)
    0.00345,    # epsilon from calibration (keV)
    1000.0,     # Overall intensity scaling
    0.01        # Rh tube scatter scaling
]
```

## Troubleshooting

### Issue: Poor fits (R² < 0.8)

**Possible causes:**
1. Peak overlap (e.g., Kα1 and Kα2 not resolved)
2. Weak signal (counts < 50)
3. Background subtraction artifacts

**Solutions:**
- Increase `window_width` parameter for overlapping peaks
- Use longer acquisition times for weak peaks
- Adjust SNIP `window_length` parameter

### Issue: Outliers in FWHM vs Energy plot

**Possible causes:**
1. Energy calibration drift between samples
2. Detector artifacts (e.g., escape peaks, sum peaks)
3. Sample charging effects

**Solutions:**
- Check energy calibration consistency
- Verify peak identification (not fitting artifacts)
- Exclude outliers and refit

### Issue: Unrealistic parameters

**Expected ranges:**
- FWHM₀: 80-150 eV (modern SDD)
- ε: 2-5 eV/keV (Si detector)

If outside these ranges:
- Check energy calibration (should be in keV, not eV!)
- Verify peak positions match expected energies
- Check for systematic errors in fitting

## Advanced Usage

### Custom Peak Selection

Edit `calibrate_peak_shape.py` to add/remove peaks:

```python
self.expected_peaks = {
    'Fe': [
        ('Fe Kα1', 6.404),
        ('Fe Kβ1', 7.058),
        # Add more peaks...
    ],
    # Add more elements...
}
```

### Adjusting Fit Windows

Modify the `window_width` parameter in `measure_peak_width()`:

```python
measurement = self.measure_peak_width(
    energy, counts_bg_sub, peak_energy, 
    filename, line_name,
    window_width=0.5  # Wider window for overlapping peaks
)
```

### Background Subtraction

Adjust SNIP parameters in `load_and_process_file()`:

```python
background = self.bg_modeler.estimate_background(
    energy, counts, 
    method='snip', 
    window_length=100  # Larger for smoother background
)
```

## Theory

### Detector Resolution Components

The total detector resolution has two main contributions:

1. **Electronic Noise (FWHM₀)**
   - Independent of energy
   - From preamplifier, detector capacitance
   - Minimized by cooling and low-noise electronics

2. **Statistical Noise (Fano term)**
   - Proportional to √E
   - From statistical fluctuations in charge carrier generation
   - Fundamental limit (cannot be eliminated)

### Why 2.355?

The factor 2.355 converts between Gaussian sigma (σ) and FWHM:
```
FWHM = 2.355 × σ
```

This comes from the Gaussian distribution where FWHM is the width at half maximum.

### Fano Factor

The Fano factor (F) describes the variance in charge carrier generation:
```
σ²_N = F × N
```

For silicon: F ≈ 0.12 (much less than Poisson statistics, F=1)

This is why Si detectors have excellent energy resolution!

## References

1. Knoll, G.F. "Radiation Detection and Measurement" (4th ed.)
2. Jenkins, R. "X-ray Fluorescence Spectrometry" (2nd ed.)
3. Beckhoff, B. et al. "Handbook of Practical X-Ray Fluorescence Analysis"

## Next Steps

After calibration:

1. ✅ **Verify results** - Check FWHM predictions match measured peaks
2. ✅ **Update calibration.py** - Use calibrated FWHM₀ and ε values
3. ✅ **Test on unknowns** - Validate improved peak fitting
4. 📊 **Monitor stability** - Recalibrate periodically (monthly/quarterly)

---

**Questions?** Check the main README or open an issue on GitHub.
