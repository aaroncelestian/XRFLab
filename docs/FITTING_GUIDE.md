# Spectrum Fitting Guide

## Overview

XRFLab now includes a comprehensive spectrum fitting engine that performs:
- **Background modeling** (SNIP, polynomial, linear, adaptive)
- **Peak detection** (automatic and element-based)
- **Peak fitting** (Gaussian, Voigt, Pseudo-Voigt)
- **Quantification** (preliminary fundamental parameters)

## Features

### Background Modeling

#### SNIP Algorithm (Recommended)
- **Statistics-sensitive Non-linear Iterative Peak-clipping**
- Best for XRF spectra with complex backgrounds
- Iterations parameter controls smoothness (default: 20)
- Works in log-space for better results

#### Polynomial Background
- Fits polynomial of specified degree (1-5)
- Good for simple, smooth backgrounds
- Can exclude peak regions from fit

#### Linear Background
- Simple linear interpolation between endpoints
- Fast but less accurate
- Good for flat backgrounds

#### Adaptive Background
- Moving percentile filter
- Adapts to local background variations
- Good for variable backgrounds

### Peak Fitting

#### Peak Shapes

**Gaussian** (Default)
- Fast and simple
- Good approximation for most XRF peaks
- Formula: `A * exp(-(x-μ)²/(2σ²))`

**Voigt Profile**
- More accurate for X-ray peaks
- Convolution of Gaussian and Lorentzian
- Accounts for natural line width
- Slower to compute

**Pseudo-Voigt**
- Linear combination of Gaussian and Lorentzian
- Faster approximation of Voigt
- Good balance of speed and accuracy

#### Peak Detection

**Element-Based**
- Uses emission lines from selected elements
- Looks for K, L, M lines within energy range
- Most accurate when elements are known

**Automatic**
- Finds peaks using scipy peak detection
- Prominence-based filtering
- Good for unknown samples

**Combined** (Recommended)
- Uses both element lines and auto-detection
- Finds expected peaks plus unknowns
- Most comprehensive approach

### Quantification

**Current Implementation** (Preliminary)
- Uses peak areas as concentration proxy
- 10% error estimate
- Placeholder for full fundamental parameters

**Future Implementation**
- Full fundamental parameters using xraylib
- Matrix corrections
- Secondary fluorescence
- Absorption corrections

## Usage

### Basic Workflow

1. **Load Spectrum**
   ```
   File → Open Spectrum
   ```

2. **Select Elements**
   - Click elements in periodic table
   - Or use "Common XRF" button

3. **Configure Fitting**
   - Background: SNIP (recommended)
   - Peak Shape: Gaussian (fast) or Voigt (accurate)
   - Include Escape Peaks: Yes (for Si detectors)
   - Pile-up Correction: Optional

4. **Fit Spectrum**
   - Click "Fit Spectrum" button
   - Watch status bar for progress
   - Results appear in right panel

### Interpreting Results

#### Fit Statistics

**χ² (Chi-squared)**
- Measure of fit quality
- Lower is better
- Depends on counting statistics

**χ²ᵣ (Reduced Chi-squared)**
- Normalized by degrees of freedom
- Should be close to 1.0 for good fit
- < 1: Over-fitting
- \> 2: Poor fit

**R² (R-squared)**
- Coefficient of determination
- 0 to 1 scale
- > 0.99 is excellent
- > 0.95 is good

**Iterations**
- Number of fitting iterations
- Currently 1 (single-pass fitting)

#### Identified Peaks

Shows all fitted peaks with:
- Element and line designation
- Energy (keV)
- Integrated area (counts)
- FWHM (Full Width at Half Maximum, keV)

Example:
```
Fe-Kα1: 6.404 keV (Area=12500, FWHM=0.150 keV)
Fe-Kβ1: 7.058 keV (Area=2100, FWHM=0.155 keV)
Cu-Kα1: 8.048 keV (Area=8900, FWHM=0.160 keV)
```

#### Quantification Results

Shows element concentrations:
- Element symbol
- Concentration (%)
- Error estimate (%)
- Line used for quantification

**Total Concentration**
- Sum of all elements
- Color coded:
  - Green: 98-102% (good)
  - Orange: 95-105% (acceptable)
  - Red: Outside range (check fit)

## Advanced Options

### Background Parameters

**SNIP Iterations**
```python
# More iterations = smoother background
iterations=10  # Aggressive (less smooth)
iterations=20  # Default (balanced)
iterations=40  # Conservative (very smooth)
```

**Polynomial Degree**
```python
degree=1  # Linear
degree=2  # Quadratic
degree=3  # Cubic (default)
degree=4  # Quartic (for complex backgrounds)
```

### Peak Fitting Parameters

**Prominence** (auto-detection)
```python
prominence=None  # Auto (5% of max)
prominence=100   # Minimum 100 counts above background
```

**Distance** (minimum peak separation)
```python
distance=10  # Default (10 channels)
distance=20  # More separation (fewer peaks)
```

## Troubleshooting

### Poor Fit Quality (High χ²ᵣ)

**Possible causes:**
- Wrong background method
- Missing elements
- Incorrect peak shape
- Overlapping peaks not resolved

**Solutions:**
1. Try different background method
2. Add more elements to selection
3. Use Voigt instead of Gaussian
4. Check for peak overlap

### Total Concentration Not 100%

**Possible causes:**
- Missing elements
- Matrix effects not corrected
- Preliminary quantification algorithm

**Solutions:**
1. Add missing elements
2. Check element selection
3. Wait for full FP implementation

### Peaks Not Detected

**Possible causes:**
- Low signal-to-noise
- Background too high
- Elements not selected

**Solutions:**
1. Increase acquisition time
2. Adjust prominence parameter
3. Select elements manually

### Fitting Takes Too Long

**Solutions:**
1. Use Gaussian instead of Voigt
2. Reduce number of selected elements
3. Disable auto-detection
4. Use simpler background method

## Technical Details

### Files

**`core/background.py`**
- BackgroundModeler class
- SNIP, polynomial, linear, adaptive methods
- Background subtraction

**`core/peak_fitting.py`**
- PeakFitter class
- Peak shapes (Gaussian, Voigt, Pseudo-Voigt)
- Peak detection and fitting
- Fit statistics calculation

**`core/fitting.py`**
- SpectrumFitter class (main engine)
- Combines background and peak fitting
- Preliminary quantification
- FitResult dataclass

### Algorithms

**SNIP Background**
1. Log-transform spectrum
2. Iteratively clip peaks
3. Use decreasing window sizes
4. Transform back to linear scale

**Peak Fitting**
1. Define fitting window around peak
2. Initial parameter guess
3. Non-linear least squares (scipy.optimize.curve_fit)
4. Calculate FWHM and area

**Fit Statistics**
1. Calculate residuals
2. Chi-squared with Poisson statistics
3. Degrees of freedom = n_points - n_params
4. R-squared from residual sum of squares

## Future Enhancements

- [ ] Iterative fitting with peak refinement
- [ ] Escape peak modeling
- [ ] Pile-up correction
- [ ] Full fundamental parameters quantification
- [ ] Matrix correction factors
- [ ] Standards-based calibration
- [ ] Batch fitting
- [ ] Export fitted parameters

---

**The fitting engine provides professional-grade spectrum analysis with multiple algorithms and comprehensive results!**
