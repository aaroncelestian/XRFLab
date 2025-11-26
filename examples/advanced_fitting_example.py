#!/usr/bin/env python3
"""
Example: Advanced Peak Fitting for Mixed Element Sample

This example demonstrates fitting a spectrum containing:
- Light elements (Mg, Na) with simple K-lines
- Medium elements (Ti, Fe) with K-lines
- Heavy elements (Zr) with complex L-lines

Shows how AdvancedPeakFitter automatically selects appropriate models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.advanced_peak_fitting import AdvancedPeakFitter, PeakFitResult
from core.fwhm_calibration import load_fwhm_calibration, create_default_fwhm_calibration


def create_synthetic_spectrum():
    """
    Create a synthetic spectrum with Mg, Ti, and Zr
    
    This simulates a real sample with:
    - Mg Kα (1.254 keV) - simple Gaussian
    - Ti Kα (4.511 keV) - simple Gaussian
    - Zr Lα (2.042 keV) - complex, broad multiplet
    - Zr Kα (15.775 keV) - Voigt profile
    """
    # Energy axis
    energy = np.linspace(0, 20, 2000)  # 0-20 keV, 10 eV/channel
    
    # Background (exponential + polynomial)
    background = 1000 * np.exp(-energy/5) + 100
    
    # Detector resolution parameters
    fwhm_0 = 0.120  # 120 eV
    epsilon = 0.0035  # 3.5 eV/keV
    
    def gaussian_peak(E, center, area, fwhm):
        """Gaussian peak"""
        sigma = fwhm / 2.355
        return area / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((E - center) / sigma)**2)
    
    def emg_peak(E, center, area, fwhm, tail_factor=0.3):
        """Exponentially modified Gaussian (for L-lines)"""
        from scipy.special import erfc
        sigma = fwhm / 2.355
        tau = sigma * tail_factor
        
        z = (sigma/tau)**2 - (E - center)/tau
        erfcz = erfc(z / np.sqrt(2))
        
        return area / (2 * tau) * np.exp(0.5 * (sigma/tau)**2 - (E - center)/tau) * erfcz
    
    # Initialize spectrum
    spectrum = background.copy()
    
    # Mg Kα (1.254 keV) - Simple Gaussian
    mg_energy = 1.254
    mg_fwhm = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * mg_energy)
    mg_area = 50000
    spectrum += gaussian_peak(energy, mg_energy, mg_area, mg_fwhm)
    
    # Ti Kα (4.511 keV) - Simple Gaussian
    ti_energy = 4.511
    ti_fwhm = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * ti_energy)
    ti_area = 100000
    spectrum += gaussian_peak(energy, ti_energy, ti_area, ti_fwhm)
    
    # Zr Lα (2.042 keV) - Complex EMG (30% broader)
    zr_l_energy = 2.042
    zr_l_fwhm_base = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * zr_l_energy)
    zr_l_fwhm = zr_l_fwhm_base * 1.3  # 30% broader (empirical)
    zr_l_area = 30000
    spectrum += emg_peak(energy, zr_l_energy, zr_l_area, zr_l_fwhm, tail_factor=0.4)
    
    # Zr Kα (15.775 keV) - Voigt-like (Gaussian with small Lorentzian)
    zr_k_energy = 15.775
    zr_k_fwhm = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * zr_k_energy)
    zr_k_area = 80000
    spectrum += gaussian_peak(energy, zr_k_energy, zr_k_area, zr_k_fwhm)
    
    # Add Poisson noise
    spectrum = np.random.poisson(spectrum)
    
    return energy, spectrum


def main():
    """Run advanced fitting example"""
    
    print("=" * 70)
    print("Advanced Peak Fitting Example")
    print("=" * 70)
    print()
    print("Sample composition: Mg + Ti + Zr")
    print("Demonstrates automatic model selection for different line types")
    print()
    
    # Create synthetic spectrum
    print("Creating synthetic spectrum...")
    energy, counts = create_synthetic_spectrum()
    print(f"  Energy range: {energy[0]:.1f} - {energy[-1]:.1f} keV")
    print(f"  Total counts: {np.sum(counts):.0f}")
    print()
    
    # Create or load FWHM calibration
    print("Loading FWHM calibration...")
    try:
        fwhm_cal = load_fwhm_calibration("calibrations/fwhm_calibration.json")
        print(f"  ✓ Loaded from file")
    except:
        fwhm_cal = create_default_fwhm_calibration()
        print(f"  ⚠ Using default calibration")
    
    print(f"  FWHM₀ = {fwhm_cal.parameters['fwhm_0']*1000:.1f} eV")
    print(f"  ε = {fwhm_cal.parameters['epsilon']*1000:.2f} eV/keV")
    print()
    
    # Create fitter
    print("Initializing AdvancedPeakFitter...")
    fitter = AdvancedPeakFitter(fwhm_calibration=fwhm_cal)
    print(f"  PyMca available: {fitter.pymca_available}")
    print(f"  Fityk available: {fitter.fityk_available}")
    print()
    
    # Define elements
    elements = ['Mg', 'Ti', 'Zr']
    
    # Show automatic model selection
    print("Automatic model selection:")
    print("-" * 70)
    
    test_lines = [
        ('Mg', 'Kα', 1.254),
        ('Ti', 'Kα', 4.511),
        ('Zr', 'Lα', 2.042),
        ('Zr', 'Kα', 15.775)
    ]
    
    for element, line, line_energy in test_lines:
        model = fitter.select_peak_model(element, line, line_energy)
        predicted_fwhm = fitter.predict_fwhm(line_energy, element, 
                                             'L' if 'L' in line else 'K')
        print(f"  {element} {line:3s} ({line_energy:6.3f} keV): "
              f"{model:10s} (FWHM = {predicted_fwhm*1000:5.1f} eV)")
    print()
    
    # Fit spectrum
    print("Fitting spectrum...")
    print("-" * 70)
    
    try:
        results = fitter.fit_spectrum(
            energy=energy,
            counts=counts,
            elements=elements,
            excitation_energy=30.0,
            method='auto'
        )
        
        print(f"✓ Fit successful! Found {len(results)} peaks")
        print()
        
        # Display results
        print("Peak Fitting Results:")
        print("-" * 70)
        print(f"{'Element':<8} {'Line':<6} {'Energy':>8} {'Area':>12} {'FWHM':>8} {'Model':<10}")
        print(f"{'':8} {'':6} {'(keV)':>8} {'(counts)':>12} {'(eV)':>8} {'':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result.element:<8} {result.line:<6} "
                  f"{result.energy:8.3f} {result.area:12.0f} "
                  f"{result.fwhm*1000:8.1f} {result.model_used:<10}")
        
        print("-" * 70)
        print()
        
        # Plot results
        print("Creating diagnostic plot...")
        plot_results(energy, counts, results, fitter)
        
    except ImportError as e:
        print(f"❌ Fitting failed: {e}")
        print()
        print("To use advanced fitting, install:")
        print("  pip install PyMca5")
        print("  or")
        print("  pip install fityk")
        return
    
    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 70)
    print("Example complete!")
    print()
    print("Key observations:")
    print("  • Mg Kα: Simple Gaussian (light element K-line)")
    print("  • Ti Kα: Simple Gaussian (medium element K-line)")
    print("  • Zr Lα: Hypermet/EMG (heavy element L-line, ~30% broader)")
    print("  • Zr Kα: Voigt (heavy element K-line)")
    print()
    print("The fitter automatically selected appropriate models for each peak!")
    print("=" * 70)


def plot_results(energy, counts, results, fitter):
    """Create diagnostic plot"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Spectrum with fitted peaks
    ax1.plot(energy, counts, 'k-', linewidth=0.5, alpha=0.5, label='Data')
    
    # Reconstruct fitted spectrum
    fitted = np.zeros_like(energy)
    
    for result in results:
        # Simple Gaussian approximation for visualization
        sigma = result.fwhm / 2.355
        peak = result.area / (sigma * np.sqrt(2*np.pi)) * \
               np.exp(-0.5 * ((energy - result.energy) / sigma)**2)
        fitted += peak
        
        # Plot individual peaks
        ax1.plot(energy, peak, '--', linewidth=1, alpha=0.7,
                label=f"{result.element} {result.line}")
    
    ax1.plot(energy, fitted, 'r-', linewidth=1.5, label='Total fit')
    
    ax1.set_xlabel('Energy (keV)', fontsize=12)
    ax1.set_ylabel('Counts', fontsize=12)
    ax1.set_title('Advanced Peak Fitting: Mg + Ti + Zr', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 20)
    
    # Plot 2: Residuals
    residuals = counts - fitted
    ax2.plot(energy, residuals, 'k-', linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.fill_between(energy, -np.sqrt(counts), np.sqrt(counts), 
                     alpha=0.3, color='gray', label='±√N (Poisson)')
    
    ax2.set_xlabel('Energy (keV)', fontsize=12)
    ax2.set_ylabel('Residuals (counts)', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)
    
    plt.tight_layout()
    
    # Save
    output_path = Path("examples/advanced_fitting_example.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
