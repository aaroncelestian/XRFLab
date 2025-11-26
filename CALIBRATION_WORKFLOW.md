# XRF Peak Shape Calibration Workflow

## Quick Start Guide

You have excellent calibration data! Here's how to use it:

### Step 1: Preview Your Data (Optional but Recommended)

```bash
python preview_calibration_data.py
```

This will:
- Load all 6 standard spectra
- Show background-subtracted spectra
- Annotate major peaks
- Print data quality metrics
- Save preview plot to `sample_data/calibration_data_preview.png`

**What to look for:**
- ✅ Clean, well-defined peaks
- ✅ Good signal-to-noise ratio (peak counts > 100)
- ✅ Proper background subtraction
- ❌ Peak overlap issues
- ❌ Detector artifacts

### Step 2: Run Calibration

```bash
python run_peak_shape_calibration.py
```

This will:
- Measure FWHM of ~15-20 peaks across 1-16 keV range
- Fit detector resolution model: `FWHM(E) = √(FWHM₀² + 2.355² · ε · E)`
- Save results to `sample_data/peak_shape_calibration.json`
- Generate calibration plot: `sample_data/peak_shape_calibration.png`

**Expected runtime:** 10-30 seconds

### Step 3: Review Results

Open `sample_data/peak_shape_calibration.png` to see:
- **Top panel**: FWHM vs Energy with fitted model
- **Bottom panel**: Fit residuals (should be random, < 5 eV)

Check `sample_data/peak_shape_calibration.json` for:
```json
{
  "fwhm_0_eV": 115.3,        // Electronic noise (80-150 eV typical)
  "epsilon_eV_per_keV": 3.45, // Fano factor (2-5 eV/keV typical)
  "r_squared": 0.9876,        // Fit quality (>0.95 excellent)
  "rmse_eV": 2.3              // Residual error (<5 eV good)
}
```

### Step 4: Apply to Your Analysis

Update `core/calibration.py` with your calibrated values:

```python
# Line ~78 in calibrate() method:
p0 = [
    0.115,      # ← Use your fwhm_0_keV value
    0.00345,    # ← Use your epsilon_keV value
    1000.0,     # Overall intensity scaling
    0.01        # Rh tube scatter scaling
]

# Line ~85 in bounds:
bounds = [
    (0.110, 0.120),     # ← Narrow range around your FWHM_0
    (0.003, 0.004),     # ← Narrow range around your epsilon
    (10.0, 100000.0),   
    (0.0, 0.5)
]
```

## Your Calibration Standards

| Standard | Key Peaks | Energy Range | Purpose |
|----------|-----------|--------------|---------|
| **Mg.txt** | Mg Kα (1.25 keV), Al Kα (1.49 keV) | Low | Tests low-energy resolution |
| **cubic zirconia.txt** | Zr Lα (2.04 keV), Zr Kα (15.75 keV) | Wide | Spans full range |
| **Ti.txt** | Ti Kα (4.51 keV), Ti Kβ (4.93 keV) | Mid | Mid-range resolution |
| **Fe.txt** | Fe Kα (6.40 keV), Fe Kβ (7.06 keV) | Mid-high | Common element |
| **Cu.txt** | Cu Kα (8.05 keV), Cu Kβ (8.91 keV) | High | High-energy resolution |
| **Zn.txt** | Zn Kα (8.64 keV), Zn Kβ (9.57 keV) | High | Additional high-E point |

**Bonus:** All except cubic zirconia have Al Kα (1.49 keV) from the sample holder!

## Understanding the Physics

### What is FWHM?

**Full Width at Half Maximum (FWHM)** is the width of a peak at 50% of its maximum height. It measures detector energy resolution:
- **Smaller FWHM** = Better resolution = Can distinguish closely-spaced peaks
- **Larger FWHM** = Worse resolution = Peaks blur together

### Why Does FWHM Increase with Energy?

Two contributions:

1. **Electronic Noise (FWHM₀)** - Constant
   - Preamplifier noise
   - Detector capacitance
   - Temperature effects
   - Typical: 80-150 eV

2. **Statistical Noise (ε term)** - Increases with √E
   - Fano statistics in charge generation
   - Fundamental quantum limit
   - Typical: 2-5 eV/keV

Combined: `FWHM(E) = √(FWHM₀² + 2.355² · ε · E)`

### Example Calculation

With calibrated values FWHM₀ = 115 eV, ε = 3.45 eV/keV:

```python
import numpy as np

def predict_fwhm(E_keV, fwhm_0_eV=115, epsilon_eV_per_keV=3.45):
    fwhm_0_keV = fwhm_0_eV / 1000
    epsilon_keV = epsilon_eV_per_keV / 1000
    fwhm_keV = np.sqrt(fwhm_0_keV**2 + 2.355**2 * epsilon_keV * E_keV)
    return fwhm_keV * 1000  # Return in eV

# Predictions
print(f"Mg Kα (1.25 keV): {predict_fwhm(1.25):.1f} eV")  # ~118 eV
print(f"Fe Kα (6.40 keV): {predict_fwhm(6.40):.1f} eV")  # ~143 eV
print(f"Cu Kα (8.05 keV): {predict_fwhm(8.05):.1f} eV")  # ~149 eV
print(f"Zr Kα (15.75 keV): {predict_fwhm(15.75):.1f} eV") # ~168 eV
```

## Troubleshooting

### Problem: Poor fit quality (R² < 0.90)

**Possible causes:**
1. Energy calibration drift between samples
2. Peak overlap (Kα1/Kα2 not resolved)
3. Weak peaks (counts < 50)
4. Background subtraction issues

**Solutions:**
- Check energy calibration consistency
- Increase `window_width` for overlapping peaks
- Exclude weak peaks (edit `expected_peaks` dict)
- Adjust SNIP `window_length` parameter

### Problem: Unrealistic parameters

**Expected ranges:**
- FWHM₀: 80-150 eV (modern SDD at -30°C)
- ε: 2-5 eV/keV (Si detector)

**If outside range:**
- ❌ Check units (should be keV, not eV!)
- ❌ Verify energy calibration
- ❌ Check for systematic fitting errors

### Problem: Large residuals (>5 eV)

**Possible causes:**
1. Non-Gaussian peak shapes (tailing)
2. Detector artifacts (escape peaks, sum peaks)
3. Sample effects (charging, self-absorption)

**Solutions:**
- Use Voigt profile instead of Gaussian
- Exclude problematic peaks
- Check for detector issues

## Advanced Topics

### Adding More Standards

Edit `calibrate_peak_shape.py`:

```python
self.expected_peaks = {
    'Fe': [...],
    'Cu': [...],
    'MyNewStandard': [
        ('Element Line', energy_keV),
        # Add more peaks...
    ]
}
```

### Voigt Profile Fitting

For better fits with tailing, modify `measure_peak_width()`:

```python
# Replace Gaussian with Voigt
def voigt_model(x, amp, mu, sigma, gamma):
    from scipy.special import wofz
    z = ((x - mu) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
```

### Energy-Dependent FWHM₀

For very high precision, allow FWHM₀ to vary:

```python
def resolution_model(E, fwhm_0, epsilon, fwhm_0_slope):
    fwhm_0_eff = fwhm_0 + fwhm_0_slope * E
    return np.sqrt(fwhm_0_eff**2 + 2.355**2 * epsilon * E)
```

## Maintenance

### When to Recalibrate

Recalibrate if:
- ✅ Monthly for critical work
- ✅ After detector service/repair
- ✅ After temperature changes
- ✅ If peak widths look wrong

### Monitoring Stability

Track FWHM₀ and ε over time:

```python
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load historical calibrations
calibrations = []
for file in Path("calibrations/").glob("*.json"):
    with open(file) as f:
        data = json.load(f)
        calibrations.append({
            'date': datetime.fromisoformat(data['calibration_date']),
            'fwhm_0': data['fwhm_0_eV'],
            'epsilon': data['epsilon_eV_per_keV']
        })

# Plot trends
dates = [c['date'] for c in calibrations]
fwhm_0s = [c['fwhm_0'] for c in calibrations]

plt.plot(dates, fwhm_0s, 'o-')
plt.xlabel('Date')
plt.ylabel('FWHM₀ (eV)')
plt.title('Detector Resolution Stability')
plt.show()
```

## Files Created

After running the calibration, you'll have:

```
XRFLab/
├── calibrate_peak_shape.py          # Main calibration code
├── run_peak_shape_calibration.py    # Quick run script
├── preview_calibration_data.py      # Data preview tool
├── PEAK_SHAPE_CALIBRATION.md        # Detailed documentation
├── CALIBRATION_WORKFLOW.md          # This file
└── sample_data/
    ├── data/
    │   ├── Fe.txt
    │   ├── Cu.txt
    │   ├── Ti.txt
    │   ├── Zn.txt
    │   ├── Mg.txt
    │   └── cubic zirconia.txt
    ├── peak_shape_calibration.json  # Calibration results
    ├── peak_shape_calibration.png   # Calibration plot
    └── calibration_data_preview.png # Data preview
```

## Next Steps

1. ✅ **Run preview** - Check data quality
2. ✅ **Run calibration** - Get FWHM₀ and ε
3. ✅ **Review results** - Verify fit quality
4. ✅ **Update code** - Use calibrated values in `calibration.py`
5. 📊 **Test** - Analyze unknown samples with improved resolution
6. 🔄 **Monitor** - Track stability over time

## Questions?

- 📖 See `PEAK_SHAPE_CALIBRATION.md` for detailed theory
- 🐛 Check GitHub issues for common problems
- 💬 Contact the XRFLab team

---

**Happy calibrating! 🎯**
