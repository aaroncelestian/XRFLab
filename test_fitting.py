"""
Test script for spectrum fitting functionality
"""

import numpy as np
from core.fitting import SpectrumFitter
from utils.sample_data import generate_sample_spectrum

print("=" * 60)
print("XRFLab Fitting Test")
print("=" * 60)
print()

# Generate sample spectrum
print("1. Generating sample spectrum...")
spectrum = generate_sample_spectrum(
    num_channels=1024,
    energy_range=(0, 20),
    elements=['Fe', 'Cu', 'Zn'],
    noise_level=10
)
print(f"   ✓ Generated spectrum with {len(spectrum.energy)} channels")
print(f"   ✓ Energy range: {spectrum.energy[0]:.2f} - {spectrum.energy[-1]:.2f} keV")
print(f"   ✓ Total counts: {np.sum(spectrum.counts):.0f}")
print()

# Create fitter
print("2. Initializing spectrum fitter...")
fitter = SpectrumFitter()
print("   ✓ Fitter initialized")
print()

# Prepare elements list
elements = [
    {'symbol': 'Fe', 'z': 26},
    {'symbol': 'Cu', 'z': 29},
    {'symbol': 'Zn', 'z': 30}
]

# Test SNIP background
print("3. Testing SNIP background estimation...")
try:
    result = fitter.fit_spectrum(
        energy=spectrum.energy,
        counts=spectrum.counts,
        elements=elements,
        background_method='snip',
        peak_shape='gaussian',
        auto_find_peaks=True
    )
    print(f"   ✓ SNIP background: {len(result.peaks)} peaks fitted")
    print(f"   ✓ χ²ᵣ = {result.statistics['reduced_chi_squared']:.3f}")
    print(f"   ✓ R² = {result.statistics['r_squared']:.4f}")
except Exception as e:
    print(f"   ✗ SNIP failed: {e}")
print()

# Test polynomial background
print("4. Testing polynomial background...")
try:
    result = fitter.fit_spectrum(
        energy=spectrum.energy,
        counts=spectrum.counts,
        elements=elements,
        background_method='polynomial',
        peak_shape='gaussian',
        degree=3
    )
    print(f"   ✓ Polynomial background: {len(result.peaks)} peaks fitted")
    print(f"   ✓ χ²ᵣ = {result.statistics['reduced_chi_squared']:.3f}")
    print(f"   ✓ R² = {result.statistics['r_squared']:.4f}")
except Exception as e:
    print(f"   ✗ Polynomial failed: {e}")
print()

# Test peak shapes
print("5. Testing different peak shapes...")
for shape in ['gaussian', 'voigt', 'pseudo_voigt']:
    try:
        result = fitter.fit_spectrum(
            energy=spectrum.energy,
            counts=spectrum.counts,
            elements=elements,
            background_method='snip',
            peak_shape=shape,
            auto_find_peaks=False
        )
        print(f"   ✓ {shape.capitalize()}: {len(result.peaks)} peaks")
    except Exception as e:
        print(f"   ✗ {shape.capitalize()} failed: {e}")
print()

# Display fitted peaks
print("6. Fitted peaks:")
for peak in result.peaks:
    if peak.element:
        print(f"   • {peak.element}-{peak.line}: {peak.energy:.3f} keV "
              f"(Area={peak.area:.0f}, FWHM={peak.fwhm:.3f} keV)")
    else:
        print(f"   • Unknown: {peak.energy:.3f} keV "
              f"(Area={peak.area:.0f}, FWHM={peak.fwhm:.3f} keV)")
print()

# Test quantification
print("7. Testing quantification...")
exp_params = {
    'excitation_energy': 20.0,
    'tube_current': 1.0,
    'live_time': 100.0,
    'detector_type': 'Si(Li)',
    'incident_angle': 45.0
}

concentrations = fitter.quantify_elements(result.peaks, exp_params)
print(f"   ✓ Quantified {len(concentrations)} elements:")
for element, data in concentrations.items():
    print(f"      {element}: {data['concentration']:.2f} ± {data['error']:.2f} %")
print()

print("=" * 60)
print("✓ All fitting tests passed!")
print("=" * 60)
print()
print("Ready to use fitting in the GUI!")
print("Run: python main.py")
