# FWHM Calibration Integration Guide

## Overview

The FWHM (peak shape) calibration is now **fully integrated** into your XRFLab application! This guide explains how it works and how to use it.

## Why This Matters

**FWHM calibration is fundamental to accurate XRF quantification** because:

1. **Peak Fitting**: Accurate FWHM → better peak deconvolution → correct intensities
2. **Overlap Resolution**: Knowing exact peak widths helps separate overlapping peaks
3. **Detection Limits**: Narrower peaks → better signal-to-noise → lower detection limits
4. **Quantification**: Accurate intensities → accurate concentrations

**Bottom line**: This should be the FIRST step in your calibration workflow!

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    XRFLab Application                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: FWHM Calibration (Pure Element Standards)          │
│  ├─ Load Fe, Cu, Ti, Zn, Mg, Zr spectra                    │
│  ├─ Measure peak widths at different energies               │
│  ├─ Fit FWHM(E) model (detector, linear, etc.)             │
│  └─ Save FWHMCalibration object                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Instrument Calibration (Reference Standards)       │
│  ├─ Load FWHMCalibration                                    │
│  ├─ Use calibrated FWHM_0 and epsilon (fixed or narrow)    │
│  ├─ Optimize intensity scale, efficiency, scatter          │
│  └─ Save CalibrationResult                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Sample Analysis (Unknowns)                         │
│  ├─ Load CalibrationResult (includes FWHM)                 │
│  ├─ Use calibrated FWHM for peak fitting                   │
│  ├─ Apply FP model for quantification                      │
│  └─ Report element concentrations                          │
└─────────────────────────────────────────────────────────────┘
```

## Workflow

### Option A: Two-Step Calibration (Recommended)

**Step 1: FWHM Calibration** (Do this once, or monthly)
```bash
# Run peak shape calibration on pure element standards
python compare_calibration_models.py

# This creates: sample_data/peak_shape_calibration_detector.json
```

**Step 2: Instrument Calibration** (Use FWHM from Step 1)
```python
from core.calibration import InstrumentCalibrator
from core.fwhm_calibration import load_fwhm_calibration

# Load FWHM calibration
fwhm_cal = load_fwhm_calibration("sample_data/peak_shape_calibration_detector.json")

# Create calibrator with FWHM
calibrator = InstrumentCalibrator(fwhm_calibration=fwhm_cal)

# Run instrument calibration (FWHM is now fixed/constrained)
result = calibrator.calibrate(energy, counts, concentrations, excitation_energy=30.0)

# Save complete calibration
calibrator.save_calibration(result, "calibrations/instrument_calibration.json")
```

**Step 3: Use for Analysis**
```python
from core.calibration import InstrumentCalibrator

# Load calibration (includes FWHM)
calibrator = InstrumentCalibrator()
result = calibrator.load_calibration("calibrations/instrument_calibration.json")

# FWHM is now available for peak fitting
fwhm_at_6keV = result.fwhm_0**2 + 2.355**2 * result.epsilon * 6.0
```

### Option B: Single-Step Calibration (Quick)

```python
from core.calibration import InstrumentCalibrator

# No pre-calibrated FWHM - optimize everything together
calibrator = InstrumentCalibrator()
result = calibrator.calibrate(energy, counts, concentrations, excitation_energy=30.0)

# FWHM_0 and epsilon are optimized along with other parameters
```

## Benefits of Two-Step Approach

### ✅ Advantages

1. **Better FWHM accuracy**
   - Pure element standards → clean, well-defined peaks
   - Multiple elements → wide energy range
   - No matrix effects or overlaps

2. **Faster instrument calibration**
   - Fewer parameters to optimize
   - FWHM is fixed → faster convergence
   - More robust optimization

3. **Reusable FWHM calibration**
   - Do once, use for all samples
   - Only recalibrate if detector changes
   - Saves time on routine analysis

4. **Better diagnostics**
   - Can monitor detector stability over time
   - Detect detector degradation early
   - Compare to manufacturer specs

### ⚠️ When to Use Single-Step

- Quick analysis needed
- No pure element standards available
- Detector is stable and well-characterized
- FWHM doesn't need high precision

## File Structure

```
XRFLab/
├── core/
│   ├── fwhm_calibration.py          # NEW: FWHM calibration module
│   ├── calibration.py                # UPDATED: Now supports FWHM input
│   └── ...
├── calibrations/
│   ├── fwhm_calibration.json         # FWHM model (from pure standards)
│   └── instrument_calibration.json   # Full calibration (includes FWHM)
├── sample_data/
│   ├── data/                         # Pure element standards
│   │   ├── Fe.txt
│   │   ├── Cu.txt
│   │   └── ...
│   └── peak_shape_calibration_detector.json  # FWHM calibration output
├── calibrate_peak_shape.py           # FWHM calibration script
├── compare_calibration_models.py     # Model comparison script
└── FWHM_INTEGRATION_GUIDE.md         # This file
```

## API Reference

### FWHMCalibration Class

```python
from core.fwhm_calibration import FWHMCalibration

# Create from calibration
fwhm_cal = FWHMCalibration(
    model_type='detector',
    parameters={'fwhm_0': 0.115, 'epsilon': 0.0035},
    parameter_errors={'fwhm_0': 0.003, 'epsilon': 0.0002},
    r_squared=0.972,
    rmse=0.0042,
    aic=-85.3,
    bic=-82.1,
    n_peaks=16,
    energy_range=(1.0, 17.0),
    calibration_date='2025-11-25T20:00:00'
)

# Predict FWHM at any energy
fwhm_6keV = fwhm_cal.predict_fwhm(6.0)  # Returns FWHM in keV

# Save/load
fwhm_cal.save("fwhm_calibration.json")
loaded = FWHMCalibration.load("fwhm_calibration.json")
```

### InstrumentCalibrator with FWHM

```python
from core.calibration import InstrumentCalibrator
from core.fwhm_calibration import load_fwhm_calibration

# Load FWHM calibration
fwhm_cal = load_fwhm_calibration("fwhm_calibration.json")

# Create calibrator with FWHM
calibrator = InstrumentCalibrator(fwhm_calibration=fwhm_cal)

# FWHM is now used as initial guess with narrow bounds
result = calibrator.calibrate(
    energy=energy,
    counts=counts,
    reference_concentrations=concentrations,
    excitation_energy=30.0
)

# Result includes FWHM information
print(f"FWHM₀ = {result.fwhm_0*1000:.1f} eV")
print(f"ε = {result.epsilon*1000:.2f} eV/keV")
print(f"Model type: {result.fwhm_model_type}")
```

## Recommended Workflow for Your Lab

### Initial Setup (One Time)

1. **Collect pure element standards**
   - ✅ You already have: Fe, Cu, Ti, Zn, Mg, cubic zirconia
   - Measure with same conditions as unknowns (30 kV, same geometry)

2. **Run FWHM calibration**
   ```bash
   python compare_calibration_models.py
   ```

3. **Verify results**
   - Check R² > 0.95
   - FWHM₀ ≈ 110-120 eV
   - ε ≈ 3-4 eV/keV
   - Residuals < 5 eV

4. **Save calibration**
   - File: `calibrations/fwhm_calibration.json`
   - Backup this file!

### Monthly Maintenance

1. **Check detector stability**
   - Re-run FWHM calibration
   - Compare to previous results
   - If FWHM₀ increases > 10%: detector may need service

2. **Update if needed**
   - If detector serviced: recalibrate
   - If new standards: add to calibration
   - Otherwise: use existing calibration

### Daily Analysis

1. **Load FWHM calibration**
   ```python
   fwhm_cal = load_fwhm_calibration("calibrations/fwhm_calibration.json")
   ```

2. **Run instrument calibration** (with reference standard)
   ```python
   calibrator = InstrumentCalibrator(fwhm_calibration=fwhm_cal)
   result = calibrator.calibrate(...)
   ```

3. **Analyze unknowns**
   - Use calibrated parameters
   - FWHM is accurate → better quantification

## Troubleshooting

### Issue: FWHM calibration fails (R² < 0.90)

**Solutions:**
1. Check for outliers (cubic zirconia L-lines)
2. Verify energy calibration
3. Increase minimum counts threshold
4. Check background subtraction

### Issue: Instrument calibration worse with FWHM

**Possible causes:**
- FWHM calibration was done at different conditions
- Sample matrix effects
- Energy calibration drift

**Solutions:**
- Use same excitation energy for both
- Widen FWHM bounds (edit calibration.py)
- Or don't use pre-calibrated FWHM

### Issue: Want to use different FWHM model

```python
# Load non-detector model
fwhm_cal = load_fwhm_calibration("peak_shape_calibration_linear.json")

# Still works! get_fwhm_initial_params() converts to detector parameters
calibrator = InstrumentCalibrator(fwhm_calibration=fwhm_cal)
```

## Advanced: Custom FWHM Models

If you want to use a custom FWHM model:

```python
from core.fwhm_calibration import FWHMCalibration
from datetime import datetime

# Create custom calibration
custom_fwhm = FWHMCalibration(
    model_type='custom',  # Your custom model
    parameters={'param1': 0.1, 'param2': 0.002},
    parameter_errors={'param1': 0.01, 'param2': 0.0001},
    r_squared=0.98,
    rmse=0.003,
    aic=-90.0,
    bic=-87.0,
    n_peaks=20,
    energy_range=(1.0, 20.0),
    calibration_date=datetime.now().isoformat()
)

# Implement predict_fwhm method
def custom_predict(energy):
    # Your custom formula
    return custom_fwhm.parameters['param1'] + custom_fwhm.parameters['param2'] * energy

custom_fwhm.predict_fwhm = custom_predict

# Use it
calibrator = InstrumentCalibrator(fwhm_calibration=custom_fwhm)
```

## Summary

### Key Points

1. ✅ **FWHM calibration is now integrated** into InstrumentCalibrator
2. ✅ **Two-step workflow is recommended** for best accuracy
3. ✅ **Backward compatible** - works without FWHM calibration too
4. ✅ **Multiple models supported** - detector, linear, quadratic, etc.
5. ✅ **Saves time** - FWHM calibration is reusable

### Next Steps

1. Run `python compare_calibration_models.py` to get your FWHM calibration
2. Use it in your instrument calibration workflow
3. Enjoy better quantification accuracy!

---

**Questions?** See `MODEL_COMPARISON_GUIDE.md` for model selection details.
