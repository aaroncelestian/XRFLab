# Spectrum Fitting Guide

## Overview

XRFLab now includes a comprehensive spectrum fitting engine that performs:
- **Background modeling** (SNIP, polynomial, linear, adaptive)
- **Peak detection** (automatic and element-based)
- **Peak fitting** (Tail-Gaussian, Gaussian, Hypermet)
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

**Tail-Gaussian** (Default)
- Main Gaussian plus a wider, slightly low-energy-shifted Gaussian
- Captures incomplete-charge-collection tails on lab EDXRF spectra
- Stable for routine fitting

**Gaussian**
- Detector core only; fewest free parameters
- Use for FWHM calibration, weak peaks, or a simple baseline

**Hypermet** (Phillips & Marlow)
- Gaussian ⊗ exponential tail (continuous erfc form) plus a low-energy step/shelf
- Physically complete ICC model; more free parameters than Tail-Gaussian
- Peak area is Gaussian + tail; the step is treated as a shelf, not peak counts

**FWHM calibration and peak shapes**
- With an FWHM calibration applied (Calibration → FWHM), the Gaussian *core* width of every shape is locked to FWHM(E)/2.355.
- Gaussian: the whole profile is fixed except amplitude and centre.
- Tail-Gaussian: tail fraction and tail width stay free; Hypermet: tail amplitude, β and step stay free; Voigt: γ stays free.
- Without a calibration all width parameters are free in the least-squares fit.

**Compton scatter geometry**
- The anode Compton humps are seeded at E' = E₀ / (1 + (E₀/511)(1 − cos θ)). θ is the tube→sample→detector angle; for Rh Kα it moves the hump from 19.45 keV (90°) to 18.8 keV (150°) — more than a full hump width.
- Default θ is 155° (near-backscatter, typical of benchtop / micro-XRF). Fit the true instrument value with **Fit θ** on the Fitting tab or by measuring a blank on the Tube Profiles tab; a measured profile overrides the Fitting-tab θ and Compton FWHM.

#### Grouped lines (default) vs. released ratios

**Grouped** — each element *sub-shell* (K; L1, L2, L3; M) is one free amplitude. Within a sub-shell the line pattern is fixed to the tabulated radiative rates (×detector efficiency, and ×matrix absorption when a composition is passed as `ratio_matrix`). Between sub-shells and between series nothing is tied — those populations depend on the excitation spectrum. Tube lines, Compton humps and unlabeled peaks are single free components with soft tube-profile ratio priors. Because centres come from the tables and core widths from the detector model, every component is linear in amplitude and the whole spectrum is solved in **one non-negative linear least squares** (AXIL / PyMca style). A small outer search refines spectrum-wide nuisance parameters: energy zero/gain (±40 eV, ±0.4 %) and the peak-shape extras shared by all peaks (tail fraction and width for Tail-Gaussian, etc.).

Consequences:
- Overlaps such as As Kα / Pb Lα or Ba L / Ti K are resolved by each element's *clean* lines (As Kβ, Pb Lβ …) instead of by whichever peak the sequential fitter reached first.
- Unresolved pairs (Kα1/Kα2, Kβ1/Kβ3, Lα1/Lα2) are split at the theoretical ratio rather than lumped into the first line.
- An element that is not in the list cannot be absorbed by a neighbour's free centre: the misfit stays in the residual where you can see it (e.g. an unfit Zr Kα at 15.7 keV shows up instead of being eaten by a shifted Sr Kβ). Add the element and re-fit.
- Fitted peaks carry a `group` tag (`Fe K`, `Pb L3`); the Results peak list shows each group's amplitude, fixed pattern, the energy refinement and the fitted global shape.

**Released ratios** (Fitting tab check-box, also on the Standards fit tab) — the legacy per-line sequential fit with free amplitudes, centres and tails. Use it for diagnostics: if a released Kβ/Kα ratio is far from theory, something is hiding under one of the lines. FP Composition reports the same check as obs/pred per line (see below). The Standards calibration stores which mode was used and warns if you quantify with the other.

#### Multi-line FP quantification

FP Composition uses *every* predictable K and L line of each element as an observation. The element's scale is the weighted least-squares slope of fitted area vs. FP-predicted intensity across its lines — weights from Poisson counting statistics plus a per-series model uncertainty (K 3 %, L 15 %, M 50 %) — iterated robustly so one contaminated line cannot drag the estimate. K and L series are separate observations (their ratio depends on the tube spectrum). Lines below 1 keV, lines predicted at < 3 % of the element's strongest, and M lines when K or L exist are excluded; unresolved sub-lines are pooled (`Kα` = Kα1+Kα2).

Standards calibration curves use the same idea: the default regressor is the **principal series** intensity (all K lines summed, else all L), shown as `K (Kα+Kβ)` in the Results table. Single-family (`Kα`) and all-lines modes remain available.

In the FP wt% table the **Line** column lists the lines used; hover for obs/pred per line. A ⚠ marks a line more than 25 % from the pooled prediction — with released ratios that points to an overlap or absorption edge on that line; with grouped lines it means the FP-corrected pattern disagrees with the pure tabulated pattern used in the fit (strong matrix absorption between Kα and Kβ), and the residual is worth a look.

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
- Peak shapes (Gaussian, Tail-Gaussian, Hypermet)
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
