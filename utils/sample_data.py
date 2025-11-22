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
    
    # Add characteristic peaks for each element
    peak_data = {
        'Fe': [(6.404, 5000), (7.058, 1000)],  # Ka, Kb
        'Cu': [(8.048, 3000), (8.905, 600)],   # Ka, Kb
        'Zn': [(8.639, 2500), (9.572, 500)],   # Ka, Kb
        'Ca': [(3.692, 4000), (4.013, 800)],   # Ka, Kb
        'Ti': [(4.511, 3500), (4.932, 700)],   # Ka, Kb
        'Mn': [(5.899, 3000), (6.490, 600)],   # Ka, Kb
        'Ni': [(7.478, 2800), (8.265, 560)],   # Ka, Kb
        'Cr': [(5.415, 3200), (5.947, 640)],   # Ka, Kb
    }
    
    # Add peaks for selected elements
    for element in elements:
        if element in peak_data:
            for peak_energy, intensity in peak_data[element]:
                counts += _gaussian_peak(energy, peak_energy, intensity, fwhm=0.15)
    
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
