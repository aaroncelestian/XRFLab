# Tube Spectrum Implementation

## Problem

The calculated spectrum from fisx was underestimating low-energy fluorescence and had incorrect relative intensities because **the incident X-ray tube spectrum was not properly modeled**.

### Symptoms
- Low energy region (0-5 keV) significantly underestimated
- Peaks appeared too wide
- Poor fit between measured and calculated spectra
- Large residuals in calibration plot

### Root Cause

fisx was configured with only a single excitation energy:
```python
self.fisx.setBeam([50.0])  # Single energy point
```

This doesn't represent the real tube output, which consists of:
1. **Bremsstrahlung continuum** - Broad spectrum from 0 to tube voltage
2. **Characteristic lines** - Sharp peaks from tube anode (e.g., Rh Kα, Rh Kβ)

## Solution

Implemented full tube spectrum modeling in `FisxCalculator._setup_tube_spectrum()`:

### 1. Energy Grid

Created a non-uniform energy grid with:
- **Fine resolution at low energies** (0-10 keV): 0.1 keV steps
- **Coarser resolution at high energies** (10-50 keV): 0.5 keV steps

This captures the rapid changes in fluorescence cross-sections at low energies while keeping computation efficient.

```python
# Fine grid at low energies
energy_grid.extend(np.arange(0.1, min(10.0, excitation_energy), 0.1))

# Coarser grid at high energies
if excitation_energy > 10.0:
    energy_grid.extend(np.arange(10.0, excitation_energy + 0.5, 0.5))
```

### 2. Bremsstrahlung Continuum

Implemented **Kramers' Law** for the bremsstrahlung spectrum:

```
I(E) ∝ Z × (E_max - E) / E
```

Where:
- `Z` = atomic number of tube anode
- `E_max` = tube voltage (keV)
- `E` = photon energy (keV)

This models the continuous X-ray emission from electron deceleration in the anode.

```python
z_tube = tube_z_map.get(tube_element, 45)  # Rh = 45
bremsstrahlung[mask] = z_tube * (excitation_energy - energy_grid[mask]) / energy_grid[mask]
```

### 3. Characteristic Lines

Added tube characteristic lines (K and L lines) on top of the continuum:

```python
# K lines: ~20% of continuum intensity at that energy
char_intensity = bremsstrahlung[idx] * 0.2 * line.get('relative_intensity', 1.0)

# L lines: ~15% of continuum intensity
char_intensity = bremsstrahlung[idx] * 0.15 * line.get('relative_intensity', 1.0)
```

For a **Rh tube at 50 keV**, this includes:
- **Rh Kα** (20.2 keV) - Major characteristic line
- **Rh Kβ** (22.7 keV) - Secondary characteristic line
- **Rh L lines** (2.7-3.1 keV) - Lower energy lines

### 4. Integration with fisx

The complete spectrum is passed to fisx:

```python
self.fisx.setBeam(energy_grid.tolist(), bremsstrahlung.tolist())
```

fisx then uses this incident spectrum to calculate:
- Energy-dependent excitation probabilities
- Proper weighting of fluorescence from different incident energies
- Realistic intensity ratios between elements

## Changes Made

### File: `core/fisx_integration.py`

#### 1. Added `_setup_tube_spectrum()` method (lines 72-150)
- Creates energy grid (180 points for 50 keV tube)
- Calculates Kramers' law bremsstrahlung
- Adds tube characteristic lines
- Configures fisx beam

#### 2. Updated `__init__()` (line 57)
```python
# Before
self.fisx.setBeam([excitation_energy])

# After
self._setup_tube_spectrum(excitation_energy, tube_element)
```

#### 3. Stored tube_element parameter (line 68)
```python
self.tube_element = tube_element
```

### File: `ui/calibration_panel.py`

#### 1. Added tube_element to experimental_params (lines 332-337)
```python
experimental_params = {
    'incident_angle': ...,
    'takeoff_angle': ...,
    'tube_current': ...,
    'tube_element': self.current_spectrum.metadata.get('tube_element', 'Rh'),
}
```

#### 2. Updated `_update_calibration_plot()` (lines 495-510)
- Extracts experimental parameters from spectrum metadata
- Passes them to `_prepare_element_data()`

### File: `core/calibration.py`

#### 1. Added tube_element support (lines 320-324)
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

## Supported Tube Elements

The implementation supports common XRF tube anodes:

| Element | Z  | K lines (keV) | L lines (keV) | Typical Use |
|---------|----|--------------:|---------------:|-------------|
| **Rh** (Rhodium) | 45 | 20.2, 22.7 | 2.7-3.1 | General purpose, EDXRF |
| **W** (Tungsten) | 74 | 59.3, 67.2 | 8.4-11.3 | High energy, thick samples |
| **Mo** (Molybdenum) | 42 | 17.5, 19.6 | 2.3-2.6 | Light element analysis |
| **Ag** (Silver) | 47 | 22.2, 25.0 | 3.0-3.4 | Alternative to Rh |
| **Cu** (Copper) | 29 | 8.0, 8.9 | 0.9-1.0 | Low energy applications |
| **Cr** (Chromium) | 24 | 5.4, 5.9 | 0.6 | Ultra-light elements |
| **Au** (Gold) | 79 | 68.8, 77.9 | 9.7-13.4 | High energy, specialized |

## Expected Improvements

### 1. Better Low Energy Fit
- Proper excitation of light elements (Na, Mg, Al, Si)
- Correct relative intensities for K vs L lines
- Accurate matrix absorption effects

### 2. Correct Peak Intensities
- Realistic fluorescence yields
- Proper energy-dependent excitation
- Better match between measured and calculated spectra

### 3. Improved Calibration
- Lower χ² values
- Better R² fit quality
- More accurate FWHM parameters

## Testing

Run the test script to verify tube spectrum configuration:

```bash
python test_tube_spectrum.py
```

Expected output:
```
Testing tube spectrum configuration...

  Configured Rh tube spectrum: 180 energy points

Tube spectrum configured successfully!

Calculating intensities for test composition...
Calculated intensities for 3 elements
  Fe: 41 lines
  Ca: 30 lines
  Si: 4 lines

Test complete!
```

## Physics Background

### Kramers' Law

The bremsstrahlung spectrum follows Kramers' law (1923):

```
dI/dE ∝ Z × i × (E_max - E) / E
```

Where:
- `dI/dE` = photon intensity per unit energy
- `Z` = atomic number of anode
- `i` = tube current
- `E_max` = maximum photon energy (= tube voltage)
- `E` = photon energy

This is a classical approximation that works well for XRF applications.

### Characteristic Lines

When incident electrons have sufficient energy (E > E_edge), they can eject inner-shell electrons from the anode atoms. The resulting vacancies are filled by outer-shell electrons, emitting characteristic X-rays:

- **K lines**: L→K transitions (Kα) and M→K transitions (Kβ)
- **L lines**: M→L transitions (Lα, Lβ, Lγ)

The intensity of characteristic lines depends on:
1. Fluorescence yield (probability of X-ray vs Auger electron)
2. Transition probability (quantum mechanical selection rules)
3. Tube current and voltage

### Energy-Dependent Excitation

Different incident energies excite different fluorescence lines:

- **Low energy photons** (< 10 keV): Excite K lines of light elements (Z < 30)
- **Medium energy photons** (10-30 keV): Excite K lines of medium elements, L lines of heavy elements
- **High energy photons** (> 30 keV): Excite K lines of heavy elements

By modeling the full tube spectrum, fisx can properly weight these contributions.

## Future Enhancements

### 1. Tube Filters
Add support for tube filters (e.g., Rh K-edge filter):
```python
def set_tube_filter(self, filter_element: str, thickness: float):
    # Attenuate tube spectrum below filter K-edge
    # Removes Rh L lines while preserving Rh K lines
```

### 2. Tube Aging
Model tube aging effects:
- Tungsten contamination on anode
- Reduced characteristic line intensity
- Changed spectrum shape

### 3. Tube Geometry
Include tube window and take-off angle effects:
- Self-absorption in anode
- Window transmission
- Angle-dependent intensity

### 4. Measured Tube Spectrum
Allow loading measured tube spectra:
```python
def load_tube_spectrum(self, energy: np.ndarray, intensity: np.ndarray):
    self.fisx.setBeam(energy.tolist(), intensity.tolist())
```

## References

1. Kramers, H.A. (1923). "On the theory of X-ray absorption and of the continuous X-ray spectrum". *Philosophical Magazine*. **46**: 836–871.

2. Ebel, H. (1999). "X-ray tube spectra". *X-Ray Spectrometry*. **28** (4): 255–266.

3. Pella, P.A., Feng, L., Small, J.A. (1985). "An analytical algorithm for calculation of spectral distributions of X-ray tubes for quantitative X-ray fluorescence analysis". *X-Ray Spectrometry*. **14** (3): 125–135.

4. fisx documentation: https://github.com/vasole/fisx

## Validation

To validate the tube spectrum implementation:

1. **Compare with manufacturer specs** - Check Rh Kα/Kβ ratio
2. **Measure pure element standards** - Verify calculated vs measured intensities
3. **Check energy dependence** - Plot fluorescence vs incident energy
4. **Test different tube voltages** - Verify Kramers' law scaling

Expected Rh tube characteristics at 50 kV:
- Rh Kα (20.2 keV): Dominant characteristic line
- Rh Kβ (22.7 keV): ~20% of Kα intensity
- Continuum: Smooth decrease from 50 keV to 0
- Kα/continuum ratio: ~0.2-0.3 at 20 keV
