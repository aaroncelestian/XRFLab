# fisx Integration Fix Summary

## Problems Fixed

### 1. setSample() API Error
```
fisx calculation failed: setSample() takes at most 2 positional arguments (3 given)
```

### 2. Material Registration Error  
```
fisx calculation failed: expected bytes, list found
```

### 3. Data Extraction Error
```
All line families case not implemented yet!!!
```
Results: 0 lines calculated

## Root Causes

### Issue 1: Incorrect setSample() Arguments
The `setSample()` method was being called incorrectly with **3 separate arguments** instead of **1 argument** (a list of layers).

### Incorrect Usage (Before)
```python
self.fisx.setSample([sample_composition], [1.0], [thickness])
```

### Correct Usage (After)
```python
self.fisx.setSample([[sample_composition, density, thickness]])
```

### Issue 2: Material Not Registered
fisx requires materials to be registered with the Elements database before use. Raw composition lists cannot be passed directly to `setSample()`.

### Issue 3: Incorrect Data Structure Parsing
The fisx results have a nested structure with layer indices that wasn't being parsed correctly:
```
{
  'Fe K': {
    0: {  # Layer index
      'KL2': {'energy': 6.4, 'rate': 1.2e-8, ...},
      'KL3': {'energy': 6.39, 'rate': 2.3e-8, ...}
    }
  }
}
```

## Changes Made

### 1. Fixed `setSample()` API Call
**File:** `core/fisx_integration.py`, lines 90-99

Created a Material object and registered it with Elements before passing to `setSample()`:
```python
# Create a Material object and register it with Elements
material_name = "Sample"
material = fisx.Material(material_name, density, thickness)
material.setComposition(composition)
self.elements.addMaterial(material)

# Set sample using the registered material name
self.fisx.setSample([[material_name, density, thickness]])
```

### 2. Added Elements Database Initialization
**File:** `core/fisx_integration.py`, lines 48-50

```python
# Initialize Elements database (required for fisx calculations)
self.elements = fisx.Elements()
self.elements.initializeAsPyMca()
```

The Elements database is required for fisx to look up atomic data and calculate fluorescence yields.

### 3. Updated `getMultilayerFluorescence()` Call
**File:** `core/fisx_integration.py`, line 115

Changed from:
```python
element_results = self.fisx.getMultilayerFluorescence(
    [element],
    None,  # Wrong: should pass Elements instance
    secondary=2,
    useMassFractions=True
)
```

To:
```python
element_results = self.fisx.getMultilayerFluorescence(
    [element],
    self.elements,  # Correct: pass Elements database
    secondary=2,
    useMassFractions=True
)
```

### 4. Fixed Line Family Specification
**File:** `core/fisx_integration.py`, lines 112-137

Changed from passing just element names to passing element + line family strings:
```python
# Determine which line families to request
element_lines = []
z = z_map.get(element, 0)
if z > 0:
    if z <= 30:  # Light to medium elements: K lines
        element_lines.append(f"{element} K")
    if z >= 20:  # Medium to heavy elements: L lines
        element_lines.append(f"{element} L")
    if z >= 56:  # Heavy elements: M lines
        element_lines.append(f"{element} M")

# Request specific line families
element_results = self.fisx.getMultilayerFluorescence(
    element_lines,  # e.g., ["Fe K", "Fe L"]
    self.elements,
    secondary=2,
    useMassFractions=True
)
```

### 5. Fixed Data Extraction Logic
**File:** `core/fisx_integration.py`, lines 152-172

Added proper nested iteration through layers and lines:
```python
for line_family_key in element_results.keys():
    line_family_data = element_results[line_family_key]
    
    # fisx returns a dict with layer indices
    for layer_idx, layer_lines in line_family_data.items():
        for line_name, line_data in layer_lines.items():
            if isinstance(line_data, dict) and 'rate' in line_data and 'energy' in line_data:
                total_rate = line_data['rate']
                line_energy = line_data['energy']
                
                if total_rate > 0 and line_energy < self.excitation_energy:
                    element_intensities[line_name] = {
                        'rate': total_rate,
                        'energy': line_energy
                    }
```

### 6. Updated convert_fisx_to_element_data()
**File:** `core/fisx_integration.py`, lines 237-268

Simplified to use energy directly from fisx instead of looking up by line name:
```python
for element, lines in fisx_results.items():
    for line_name, line_info in lines.items():
        if isinstance(line_info, dict):
            intensity = line_info.get('rate', 0)
            energy = line_info.get('energy', 0)
            
            if energy > 0 and energy < excitation_energy and intensity > 0:
                element_data.append({
                    'element': element,
                    'line': line_name,
                    'energy': energy,
                    'relative_intensity': intensity
                })
```

### 7. Added Density Parameter
**File:** `core/fisx_integration.py`, line 69

Added `density` parameter with a default value of 2.5 g/cm³ (appropriate for geological samples):
```python
def calculate_intensities(self,
                         composition: Dict[str, float],
                         thickness: float = 0.1,
                         density: float = 2.5) -> Dict[str, Dict[str, Dict]]:
```

## API Reference

Based on the [fisx GitHub repository](https://github.com/vasole/fisx), the correct API usage is:

### setSample()
```python
xrf.setSample([
    [composition_list, density, thickness],  # Layer 1
    # ... additional layers if needed
])
```

Where:
- `composition_list` is a list of `[element, fraction]` pairs
- `density` is in g/cm³
- `thickness` is in cm

### getMultilayerFluorescence()
```python
results = xrf.getMultilayerFluorescence(
    elements_list,      # List of element symbols
    elements_instance,  # Elements database instance
    secondary=2,        # Include secondary and tertiary fluorescence
    useMassFractions=True
)
```

## Testing

Run the test script to verify the fixes:
```bash
python test_fisx_fix.py
```

Note: This requires fisx to be installed:
```bash
pip install fisx
```

## Results

**Before fixes:**
```
fisx calculation failed: setSample() takes at most 2 positional arguments (3 given), falling back to simplified FP
Calculated 52 lines using simplified FP
```

**After fixes:**
```
Calibration complete!
  Optimized FWHM_0: 0.0804 keV
  Optimized EPSILON: 0.002108 keV
  Optimized gamma/sigma: 0.000
  R²: 0.8259
  χ²: 148216.35
  Calculated 6800 lines using fisx (PyMca FP)
```

## Impact

These fixes ensure that:
1. ✅ The fisx library is called with the correct API
2. ✅ Full fundamental parameters calculations work (including secondary/tertiary fluorescence)
3. ✅ The calibration routine uses accurate fisx calculations instead of falling back to simplified FP
4. ✅ Matrix effects and absorption are properly accounted for
5. ✅ All emission lines (K, L, M shells) are correctly extracted with their energies and intensities
6. ✅ 6800+ emission lines calculated vs. 52 with simplified FP (130x improvement)
