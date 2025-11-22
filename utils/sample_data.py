"""
Generate sample XRF spectrum data for testing
"""

import numpy as np
from core.spectrum import Spectrum


def generate_sample_spectrum(
    num_channels=2048,
    energy_range=(0, 20),
    elements=None,
    noise_level=10
):
    """
    Generate a synthetic XRF spectrum for testing
    
    Args:
        num_channels: Number of channels
        energy_range: (min, max) energy in keV
        elements: List of element symbols to include
        noise_level: Poisson noise level
        
    Returns:
        Spectrum object
    """
    if elements is None:
        elements = ['Fe', 'Cu', 'Zn']
    
    # Create energy axis
    energy = np.linspace(energy_range[0], energy_range[1], num_channels)
    
    # Initialize counts with background
    counts = _generate_background(energy, intensity=1000)
    
    # Add characteristic peaks for each element with realistic intensity ratios
    # Intensities based on relative fluorescence yields and transition probabilities
    peak_data = {
        'Fe': [
            # K-series (Kα doublet and Kβ)
            (6.404, 5000),   # Kα1 (strongest)
            (6.391, 4500),   # Kα2 (~90% of Kα1)
            (7.058, 1000),   # Kβ1
            (7.098, 100),    # Kβ3 (weak)
            # L-series
            (0.705, 800),    # Lα1
            (0.718, 400),    # Lβ1
            (0.792, 100),    # Lγ (weak)
        ],
        'Cu': [
            (8.048, 3000),   # Kα1
            (8.028, 2700),   # Kα2
            (8.905, 600),    # Kβ1
            (8.977, 60),     # Kβ3
        ],
        'Zn': [
            (8.639, 2500),   # Kα1
            (8.616, 2250),   # Kα2
            (9.572, 500),    # Kβ1
            (9.650, 50),     # Kβ3
        ],
        'Ca': [
            # K-series
            (3.692, 4000),   # Kα1
            (3.688, 3600),   # Kα2
            (4.013, 800),    # Kβ1
            # L-series (weak for Ca)
            (0.341, 200),    # Lα
        ],
        'Ti': [
            # K-series
            (4.511, 3500),   # Kα1
            (4.505, 3150),   # Kα2
            (4.932, 700),    # Kβ1
            # L-series
            (0.452, 300),    # Lα
            (0.458, 150),    # Lβ
        ],
        'Mn': [
            (5.899, 3000),   # Kα1
            (5.888, 2700),   # Kα2
            (6.490, 600),    # Kβ1
            (6.539, 60),     # Kβ3
        ],
        'Ni': [
            (7.478, 2800),   # Kα1
            (7.461, 2520),   # Kα2
            (8.265, 560),    # Kβ1
            (8.333, 56),     # Kβ3
        ],
        'Cr': [
            (5.415, 3200),   # Kα1
            (5.405, 2880),   # Kα2
            (5.947, 640),    # Kβ1
            (5.989, 64),     # Kβ3
        ],
    }
    
    # Add peaks for selected elements with energy-dependent FWHM
    for element in elements:
        if element in peak_data:
            for peak_energy, intensity in peak_data[element]:
                # Use energy-dependent FWHM matching the fitter
                fwhm = _calculate_fwhm(peak_energy)
                counts += _voigt_peak(energy, peak_energy, intensity, fwhm=fwhm)
    
    # Add Poisson noise
    counts = np.random.poisson(counts + noise_level)
    counts = np.maximum(counts, 1)  # Ensure positive counts
    
    # Create spectrum object
    spectrum = Spectrum(
        energy=energy,
        counts=counts.astype(float),
        live_time=100.0,
        real_time=105.0,
        metadata={
            'description': 'Synthetic XRF spectrum',
            'elements': elements
        }
    )
    
    return spectrum


def _calculate_fwhm(energy):
    """
    Calculate energy-dependent FWHM matching the peak fitter
    FWHM(E) = sqrt(FWHM_0^2 + 2.35 * epsilon * E)
    """
    FWHM_0 = 0.100  # keV - must match peak_fitting.py
    EPSILON = 0.0025  # Must match peak_fitting.py
    return np.sqrt(FWHM_0**2 + 2.35 * EPSILON * energy)


def _generate_background(energy, intensity=1000):
    """Generate realistic XRF background"""
    # Exponential decay with some structure
    background = intensity * np.exp(-energy / 5.0)
    
    # Add some Compton scatter structure
    compton_energy = 10.0
    compton_width = 2.0
    background += 200 * np.exp(-((energy - compton_energy) / compton_width) ** 2)
    
    return background


def _gaussian_peak(energy, center, intensity, fwhm=0.15):
    """Generate Gaussian peak"""
    sigma = fwhm / 2.355  # Convert FWHM to sigma
    return intensity * np.exp(-((energy - center) / sigma) ** 2 / 2)


def _voigt_peak(energy, center, intensity, fwhm=0.15):
    """
    Generate Voigt peak (convolution of Gaussian and Lorentzian)
    More realistic for X-ray peaks
    """
    from scipy.special import wofz
    
    sigma = fwhm / 2.355  # Convert FWHM to sigma
    gamma = sigma * 0.15  # Lorentzian width (15% of Gaussian width)
    
    # Voigt profile using Faddeeva function
    z = ((energy - center) + 1j * gamma) / (sigma * np.sqrt(2))
    voigt = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    
    # Normalize to match intensity
    peak_max = np.max(voigt)
    if peak_max > 0:
        voigt = voigt / peak_max * intensity
    
    return voigt


def save_sample_spectrum(file_path, **kwargs):
    """
    Generate and save a sample spectrum
    
    Args:
        file_path: Output file path
        **kwargs: Arguments passed to generate_sample_spectrum
    """
    from utils.io_handler import IOHandler
    
    spectrum = generate_sample_spectrum(**kwargs)
    io_handler = IOHandler()
    io_handler.save_spectrum(spectrum, file_path)
    
    return spectrum


if __name__ == "__main__":
    # Generate sample spectra for testing
    import os
    
    # Create sample_data directory
    os.makedirs("sample_data", exist_ok=True)
    
    # Generate different sample spectra
    print("Generating sample XRF spectra...")
    
    # Steel sample (Fe, Cr, Ni, Mn)
    save_sample_spectrum(
        "sample_data/steel_sample.txt",
        elements=['Fe', 'Cr', 'Ni', 'Mn']
    )
    print("✓ Generated steel_sample.txt")
    
    # Brass sample (Cu, Zn)
    save_sample_spectrum(
        "sample_data/brass_sample.txt",
        elements=['Cu', 'Zn']
    )
    print("✓ Generated brass_sample.txt")
    
    # Mineral sample (Ca, Ti, Fe)
    save_sample_spectrum(
        "sample_data/mineral_sample.txt",
        elements=['Ca', 'Ti', 'Fe']
    )
    print("✓ Generated mineral_sample.txt")
    
    print("\nSample spectra generated in sample_data/ directory")
