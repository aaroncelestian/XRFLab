# Advanced Peak Fitting Guide

## Problem Statement

**Real samples with mixed elements** (e.g., Mg + Ti + Zr) require different peak models:

- **Light elements (Mg, Al, Na)**: K-lines are simple → Gaussian peaks
- **Heavy elements (Zr, Ba, REE)**: L-lines are complex → Need asymmetric models

**Your observation**: Zr L-lines are ~30% wider than predicted by detector model due to:
1. Multiplet structure (multiple unresolved lines)
2. Matrix effects (self-absorption, scattering)
3. Natural line width (shorter lifetime)

**Solution**: Use sophisticated peak fitting with automatic model selection!

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              AdvancedPeakFitter                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Automatic Model Selection                              │ │
│  │  • K-lines (Z<30): Gaussian                            │ │
│  │  • K-lines (Z≥30): Voigt                               │ │
│  │  • L-lines (Z<50): Hypermet/EMG                        │ │
│  │  • L-lines (Z≥50): Hypermet/EMG with strong tails      │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│              ┌────────────┴────────────┐                    │
│              ▼                         ▼                     │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │   PyMca5         │      │   Fityk          │           │
│  │   (Hypermet)     │      │   (Custom EMG)   │           │
│  └──────────────────┘      └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                 Uses FWHM Calibration
                 (from your detector)
```

## Installation

### PyMca5 (Recommended for L-lines)

```bash
pip install PyMca5
```

**Pros:**
- ✅ Hypermet tails (asymmetric peaks)
- ✅ Automatic L-line multiplet handling
- ✅ Matrix corrections built-in
- ✅ Escape peaks, scatter peaks
- ✅ Well-tested for XRF

**Cons:**
- ⚠️ Complex configuration
- ⚠️ Less control over individual peaks

### Fityk (Alternative for custom fitting)

```bash
pip install fityk
```

**Pros:**
- ✅ Full control over peak shapes
- ✅ Custom functions (EMG, etc.)
- ✅ Interactive fitting
- ✅ Excellent diagnostics

**Cons:**
- ⚠️ Manual peak list required
- ⚠️ No automatic matrix corrections
- ⚠️ More setup needed

## Usage

### Basic Example: Mixed Sample (Mg + Ti + Zr)

```python
from core.advanced_peak_fitting import AdvancedPeakFitter
from core.fwhm_calibration import load_fwhm_calibration
from utils.spectrum_loader import load_spectrum

# Load your FWHM calibration
fwhm_cal = load_fwhm_calibration("calibrations/fwhm_calibration.json")

# Create fitter
fitter = AdvancedPeakFitter(fwhm_calibration=fwhm_cal)

# Load spectrum
energy, counts = load_spectrum("sample.txt")

# Define elements present
elements = ['Mg', 'Ti', 'Zr']

# Fit with automatic method selection
results = fitter.fit_spectrum(
    energy=energy,
    counts=counts,
    elements=elements,
    excitation_energy=30.0,  # kV
    method='auto'  # Automatically selects PyMca or Fityk
)

# Print results
for result in results:
    print(f"{result.element} {result.line}:")
    print(f"  Energy: {result.energy:.3f} keV")
    print(f"  Area: {result.area:.0f} ± {result.area_error:.0f}")
    print(f"  FWHM: {result.fwhm*1000:.1f} eV")
    print(f"  Model: {result.model_used}")
    print(f"  Type: {result.peak_type}")
    print()
```

### Example Output:

```
Mg Kα:
  Energy: 1.254 keV
  Area: 15234 ± 762
  FWHM: 130.2 eV
  Model: gaussian
  Type: K-line

Ti Kα:
  Energy: 4.511 keV
  Area: 45678 ± 1234
  FWHM: 145.3 eV
  Model: gaussian
  Type: K-line

Zr Lα:
  Energy: 2.042 keV
  Area: 8934 ± 567
  FWHM: 175.4 eV  ← Note: 30% wider than Mg!
  Model: hypermet
  Type: L-line

Zr Kα:
  Energy: 15.775 keV
  Area: 23456 ± 987
  FWHM: 182.1 eV
  Model: voigt
  Type: K-line
```

### Force Specific Method

```python
# Use PyMca (best for complex L-lines)
results_pymca = fitter.fit_spectrum(
    energy, counts, elements,
    excitation_energy=30.0,
    method='pymca'
)

# Use Fityk (best for custom control)
results_fityk = fitter.fit_spectrum(
    energy, counts, elements,
    excitation_energy=30.0,
    method='fityk'
)
```

## Peak Models

### 1. Gaussian (Simple K-lines)

**Used for**: Light elements (Z < 30), K-lines

```
f(x) = A · exp(-0.5 · ((x - μ) / σ)²)
```

**Parameters:**
- A: Peak height
- μ: Center (energy)
- σ: Width (from FWHM calibration)

**Example**: Mg Kα, Al Kα, Ti Kα

### 2. Voigt (Heavy K-lines)

**Used for**: Heavy elements (Z ≥ 30), K-lines

```
f(x) = Gaussian ⊗ Lorentzian
```

**Parameters:**
- Gaussian width: From detector
- Lorentzian width: Natural line width

**Example**: Zr Kα, Cu Kα

### 3. Hypermet (Complex L-lines)

**Used for**: All L-lines, especially Z ≥ 40

```
f(x) = Gaussian + Exponential_tail + Step_function
```

**Features:**
- Asymmetric tails on low-energy side
- Handles multiplet structure
- Accounts for incomplete charge collection

**Example**: Zr Lα, Zr Lβ, Ba Lα

### 4. EMG (Exponentially Modified Gaussian)

**Used for**: L-lines when PyMca not available

```
f(x) = A · exp(0.5·(σ/τ)² - (x-μ)/τ) · erfc((σ/τ - (x-μ)/σ) / √2)
```

**Parameters:**
- A: Amplitude
- μ: Center
- σ: Gaussian width
- τ: Exponential tail parameter

**Example**: Zr Lα (alternative to hypermet)

## Automatic Model Selection Logic

```python
def select_model(element, line, energy):
    z = atomic_number(element)
    
    if 'K' in line:
        if z < 30:
            return 'gaussian'  # Simple K-lines
        else:
            return 'voigt'  # Heavy K-lines
    
    elif 'L' in line:
        if z < 50:
            return 'hypermet'  # Medium-Z L-lines
        else:
            return 'hypermet'  # Heavy-Z L-lines (strong tails)
    
    elif 'M' in line:
        return 'hypermet'  # M-lines always complex
    
    else:
        return 'gaussian'  # Fallback
```

## L-Line Broadening Corrections

The fitter includes **empirical broadening factors** for L-lines:

```python
l_line_broadening = {
    'Zr': 1.3,  # 30% wider than detector model predicts
    'Ba': 1.4,  # 40% wider
    'La': 1.4,
    'Ce': 1.4,
    # Add more as you calibrate them
}
```

**How to calibrate your own:**

1. Measure pure element standard (e.g., pure Zr metal)
2. Fit Zr Lα peak, measure FWHM
3. Compare to predicted FWHM from detector model
4. Calculate broadening factor: `measured / predicted`
5. Add to dictionary

**Example for Zr:**
```
Predicted FWHM (2.04 keV) = 135 eV  (from detector model)
Measured FWHM (Zr Lα) = 175 eV      (from your data)
Broadening factor = 175 / 135 = 1.30
```

## Integration with Main Application

### In your analysis workflow:

```python
# main.py or analysis module

from core.advanced_peak_fitting import AdvancedPeakFitter
from core.fwhm_calibration import load_fwhm_calibration
from core.fundamental_parameters import FundamentalParameters

# 1. Load calibrations
fwhm_cal = load_fwhm_calibration("calibrations/fwhm_calibration.json")

# 2. Create fitter
fitter = AdvancedPeakFitter(fwhm_calibration=fwhm_cal)

# 3. Fit spectrum
results = fitter.fit_spectrum(energy, counts, elements=['Mg', 'Ti', 'Zr'], 
                              excitation_energy=30.0)

# 4. Extract peak areas
peak_areas = {f"{r.element} {r.line}": r.area for r in results}

# 5. Use for quantification
fp = FundamentalParameters()
concentrations = fp.calculate_concentrations(
    peak_areas=peak_areas,
    matrix_composition={'O': 50.0, 'Si': 25.0, ...},
    excitation_energy=30.0
)

print("Concentrations (ppm):")
for element, conc in concentrations.items():
    print(f"  {element}: {conc:.1f}")
```

### In UI (MainWindow):

```python
# ui/main_window.py

def analyze_spectrum(self):
    """Analyze loaded spectrum"""
    
    # Get spectrum data
    energy = self.spectrum.energy
    counts = self.spectrum.counts
    
    # Get selected elements
    elements = self.get_selected_elements()
    
    # Create fitter with calibration
    fitter = AdvancedPeakFitter(fwhm_calibration=self.fwhm_calibration)
    
    # Fit
    self.show_progress("Fitting peaks...")
    results = fitter.fit_spectrum(energy, counts, elements, 
                                  excitation_energy=self.excitation_energy)
    
    # Display results
    self.display_peak_results(results)
    
    # Calculate concentrations
    self.calculate_concentrations(results)
```

## Comparison: PyMca vs Fityk

| Feature | PyMca | Fityk |
|---------|-------|-------|
| **L-line handling** | ✅ Excellent (hypermet) | ⚠️ Good (EMG) |
| **Automatic peaks** | ✅ Yes | ❌ No (manual list) |
| **Matrix corrections** | ✅ Built-in | ❌ External |
| **Escape peaks** | ✅ Automatic | ⚠️ Manual |
| **Scatter peaks** | ✅ Automatic | ⚠️ Manual |
| **Custom shapes** | ⚠️ Limited | ✅ Full control |
| **Diagnostics** | ⚠️ Basic | ✅ Excellent |
| **Speed** | ✅ Fast | ⚠️ Slower |
| **Learning curve** | ⚠️ Steep | ⚠️ Steep |

**Recommendation**: Use **PyMca** for routine analysis, **Fityk** for troubleshooting.

## Troubleshooting

### PyMca fit fails

```python
# Check configuration
print(fitter.pymca_available)  # Should be True

# Try with simpler config
results = fitter.fit_with_pymca(
    energy, counts, elements=['Mg'],  # Start with one element
    excitation_energy=30.0
)
```

### Fityk fit fails

```python
# Check peak list
peak_list = fitter._build_peak_list(elements, excitation_energy=30.0)
print(f"Found {len(peak_list)} peaks")

# Fit with fewer peaks
results = fitter.fit_with_fityk(
    energy, counts,
    peak_list=[('Mg', 'Kα', 1.254)],  # Start simple
    background_degree=3  # Lower degree
)
```

### L-lines still too wide

```python
# Add custom broadening factor
fitter.l_line_broadening['Zr'] = 1.5  # Increase from 1.3

# Or measure it from your data
measured_fwhm = 180  # eV
predicted_fwhm = fitter.predict_fwhm(2.042, element='Zr', line_type='K')
broadening = measured_fwhm / (predicted_fwhm * 1000)
print(f"Empirical broadening factor: {broadening:.2f}")
```

## Best Practices

1. **Always use FWHM calibration** from pure element standards
2. **Start with PyMca** for complex samples
3. **Use Fityk** for diagnostic fitting and troubleshooting
4. **Calibrate L-line broadening** for your specific elements
5. **Check fit quality** (R², residuals) before trusting results
6. **Compare methods** (PyMca vs Fityk) for validation

## Summary

**For samples with Mg, Na, and Zr:**

1. ✅ Use `AdvancedPeakFitter` with your FWHM calibration
2. ✅ Automatic model selection: Gaussian for Mg/Na K-lines, Hypermet for Zr L-lines
3. ✅ Empirical broadening corrections for L-lines
4. ✅ PyMca handles complex multiplet structure automatically
5. ✅ Fityk available for custom refinement

**Result**: Accurate peak areas for all elements, even with complex L-lines!

---

**Next**: See `examples/advanced_fitting_example.py` for complete working example.
