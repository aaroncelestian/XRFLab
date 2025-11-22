"""
Peak fitting for XRF spectra
"""

import numpy as np
from scipy import signal, optimize
from scipy.special import wofz
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Peak:
    """Represents a fitted peak"""
    energy: float  # keV
    amplitude: float
    fwhm: float  # keV
    area: float
    element: str = None
    line: str = None
    shape: str = 'gaussian'  # Peak shape used for fitting
    shape_params: dict = None  # Parameters specific to the shape (sigma, gamma, etc.)
    is_tube_line: bool = False  # True if this is from X-ray tube, not sample
    
    def __post_init__(self):
        if self.shape_params is None:
            self.shape_params = {}
    
    def __str__(self):
        tube_marker = " [TUBE]" if self.is_tube_line else ""
        if self.element and self.line:
            return f"{self.element}-{self.line}: {self.energy:.3f} keV (Area: {self.area:.1f}){tube_marker}"
        return f"Peak at {self.energy:.3f} keV (Area: {self.area:.1f}){tube_marker}"


class PeakFitter:
    """Peak fitting for XRF spectra"""
    
    # Detector parameters for energy-dependent FWHM
    # FWHM(E) = sqrt(FWHM_0^2 + 2.35 * epsilon * E)
    # These can be calibrated using InstrumentCalibrator
    FWHM_0 = 0.050  # keV at 0 keV (noise contribution)
    EPSILON = 0.0015  # Fano factor * w (eV per e-h pair)
    VOIGT_GAMMA_RATIO = 0.15  # gamma/sigma ratio for Voigt peaks
    USE_CALIBRATED_SHAPES = False  # If True, fix peak shapes during fitting
    
    @staticmethod
    def calculate_fwhm(energy):
        """Calculate energy-dependent FWHM for detector"""
        return np.sqrt(PeakFitter.FWHM_0**2 + 2.35 * PeakFitter.EPSILON * energy)
    
    @staticmethod
    def gaussian(x, amplitude, center, sigma):
        """Gaussian peak function"""
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    @staticmethod
    def lorentzian(x, amplitude, center, gamma):
        """Lorentzian peak function"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    @staticmethod
    def voigt(x, amplitude, center, sigma, gamma):
        """
        Voigt profile (convolution of Gaussian and Lorentzian)
        More accurate for X-ray peaks
        """
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def pseudo_voigt(x, amplitude, center, sigma, eta):
        """
        Pseudo-Voigt profile (linear combination of Gaussian and Lorentzian)
        Faster approximation of Voigt profile
        
        Args:
            eta: Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
        """
        gaussian = PeakFitter.gaussian(x, 1, center, sigma)
        lorentzian = PeakFitter.lorentzian(x, 1, center, sigma)
        return amplitude * (eta * lorentzian + (1 - eta) * gaussian)
    
    @staticmethod
    def hypermet(x, amplitude, center, sigma, tail_amplitude, tail_slope):
        """
        Hypermet function for XRF peaks with low-energy tail
        Combines Gaussian with exponential tail for incomplete charge collection
        
        Args:
            tail_amplitude: Relative amplitude of tail (0-1)
            tail_slope: Decay slope of tail (keV^-1)
        """
        # Main Gaussian peak
        gaussian = PeakFitter.gaussian(x, amplitude, center, sigma)
        
        # Low-energy exponential tail
        tail = np.zeros_like(x)
        mask = x < center
        if np.any(mask):
            tail[mask] = amplitude * tail_amplitude * np.exp(tail_slope * (x[mask] - center))
        
        return gaussian + tail
    
    @staticmethod
    def tail_gaussian(x, amplitude, center, sigma, tail_fraction, tail_sigma):
        """
        Gaussian with tail component (simplified hypermet)
        More stable for fitting than full hypermet
        
        Args:
            tail_fraction: Fraction of intensity in tail (0-1)
            tail_sigma: Width of tail relative to main peak (typically 2-5x sigma)
        """
        # Main Gaussian
        main_peak = (1 - tail_fraction) * PeakFitter.gaussian(x, amplitude, center, sigma)
        
        # Tail component (wider Gaussian on low-energy side)
        tail_peak = tail_fraction * PeakFitter.gaussian(x, amplitude, center - 0.5 * sigma, tail_sigma)
        
        return main_peak + tail_peak
    
    @staticmethod
    def find_peaks(energy, counts, prominence=None, distance=None, height=None):
        """
        Find peaks in spectrum using scipy peak detection
        
        Args:
            energy: Energy array
            counts: Counts array (background-subtracted recommended)
            prominence: Minimum peak prominence
            distance: Minimum distance between peaks (in indices)
            height: Minimum peak height
            
        Returns:
            List of (energy, height) tuples for detected peaks
        """
        if prominence is None:
            # Auto-calculate prominence as 5% of max
            prominence = np.max(counts) * 0.05
        
        if distance is None:
            # Default: at least 10 channels apart
            distance = 10
        
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            counts,
            prominence=prominence,
            distance=distance,
            height=height
        )
        
        # Extract peak information
        peaks = []
        for idx in peak_indices:
            peak_energy = energy[idx]
            peak_height = counts[idx]
            peaks.append((peak_energy, peak_height))
        
        return peaks
    
    @staticmethod
    def fit_single_peak(energy, counts, initial_center, shape='gaussian', 
                       bounds=None):
        """
        Fit a single peak
        
        Args:
            energy: Energy array
            counts: Counts array
            initial_center: Initial guess for peak center
            shape: 'gaussian', 'lorentzian', 'voigt', or 'pseudo_voigt'
            bounds: Parameter bounds
            
        Returns:
            Peak object with fitted parameters
        """
        # Define fitting window around peak
        # Use appropriate window for peak width (±3 FWHM is standard)
        fwhm_estimate = PeakFitter.calculate_fwhm(initial_center)
        
        # Use wider window for low-energy peaks due to more overlap
        if initial_center < 3.0:  # Low energy (Si, Al, Mg, Na)
            window_width = 5.0 * fwhm_estimate  # Wider window
        else:
            window_width = 3.0 * fwhm_estimate  # Standard window
        
        mask = np.abs(energy - initial_center) < window_width
        
        if np.sum(mask) < 5:
            # Not enough points
            return None
        
        x_fit = energy[mask]
        y_fit = counts[mask]
        
        # Initial parameter guesses
        amplitude_guess = np.max(y_fit)
        center_guess = initial_center
        # Use energy-dependent FWHM for better initial guess
        fwhm_guess = PeakFitter.calculate_fwhm(initial_center)
        sigma_guess = fwhm_guess / 2.355  # Convert FWHM to sigma
        
        try:
            shape_params = {}
            
            if shape == 'gaussian':
                p0 = [amplitude_guess, center_guess, sigma_guess]
                if bounds is None:
                    # Allow FWHM to refine freely within reasonable physical limits
                    # Use energy-dependent guess but don't constrain too tightly
                    bounds = ([0, center_guess - 0.2, sigma_guess * 0.3],
                             [np.inf, center_guess + 0.2, sigma_guess * 3.0])
                
                popt, _ = optimize.curve_fit(
                    PeakFitter.gaussian, x_fit, y_fit, p0=p0, bounds=bounds,
                    maxfev=5000
                )
                amplitude, center, sigma = popt
                fwhm = 2.355 * sigma  # FWHM = 2.355 * sigma for Gaussian
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma}
            
            elif shape == 'voigt':
                # Use calibrated gamma ratio if available
                gamma_guess = sigma_guess * PeakFitter.VOIGT_GAMMA_RATIO
                
                if PeakFitter.USE_CALIBRATED_SHAPES:
                    # Fix peak shape, only fit amplitude and center
                    sigma_fixed = sigma_guess
                    gamma_fixed = gamma_guess
                    
                    def voigt_fixed_shape(x, amplitude, center):
                        return PeakFitter.voigt(x, amplitude, center, sigma_fixed, gamma_fixed)
                    
                    p0 = [amplitude_guess, center_guess]
                    bounds = ([0, center_guess - 0.2], [np.inf, center_guess + 0.2])
                    
                    popt, _ = optimize.curve_fit(
                        voigt_fixed_shape, x_fit, y_fit, p0=p0, bounds=bounds,
                        maxfev=5000
                    )
                    amplitude, center = popt
                    sigma = sigma_fixed
                    gamma = gamma_fixed
                else:
                    # Fit all parameters including shape
                    p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess]
                    if bounds is None:
                        bounds = ([0, center_guess - 0.2, sigma_guess * 0.3, 0.001],
                                 [np.inf, center_guess + 0.2, sigma_guess * 3.0, sigma_guess * 2.0])
                    
                    popt, _ = optimize.curve_fit(
                        PeakFitter.voigt, x_fit, y_fit, p0=p0, bounds=bounds,
                        maxfev=5000
                    )
                    amplitude, center, sigma, gamma = popt
                
                fwhm = 2 * sigma  # Approximate
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma, 'gamma': gamma}
            
            elif shape == 'pseudo_voigt':
                # Start with more Gaussian character (eta=0.3)
                p0 = [amplitude_guess, center_guess, sigma_guess, 0.3]
                if bounds is None:
                    bounds = ([0, center_guess - 0.2, sigma_guess * 0.3, 0],
                             [np.inf, center_guess + 0.2, sigma_guess * 3.0, 1])
                
                popt, _ = optimize.curve_fit(
                    PeakFitter.pseudo_voigt, x_fit, y_fit, p0=p0, bounds=bounds,
                    maxfev=5000
                )
                amplitude, center, sigma, eta = popt
                fwhm = 2.355 * sigma
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma, 'eta': eta}
            
            elif shape == 'hypermet':
                p0 = [amplitude_guess, center_guess, sigma_guess, 0.1, 2.0]
                if bounds is None:
                    bounds = ([0, center_guess - 0.2, sigma_guess * 0.3, 0, 0.5],
                             [np.inf, center_guess + 0.2, sigma_guess * 3.0, 0.5, 10])
                
                popt, _ = optimize.curve_fit(
                    PeakFitter.hypermet, x_fit, y_fit, p0=p0, bounds=bounds,
                    maxfev=5000
                )
                amplitude, center, sigma, tail_amp, tail_slope = popt
                fwhm = 2.355 * sigma
                area = amplitude * sigma * np.sqrt(2 * np.pi) * (1 + tail_amp)
                shape_params = {'sigma': sigma, 'tail_amplitude': tail_amp, 'tail_slope': tail_slope}
            
            elif shape == 'tail_gaussian':
                p0 = [amplitude_guess, center_guess, sigma_guess, 0.15, sigma_guess * 3]
                if bounds is None:
                    bounds = ([0, center_guess - 0.2, sigma_guess * 0.3, 0, sigma_guess],
                             [np.inf, center_guess + 0.2, sigma_guess * 3.0, 0.5, sigma_guess * 10])
                
                popt, _ = optimize.curve_fit(
                    PeakFitter.tail_gaussian, x_fit, y_fit, p0=p0, bounds=bounds,
                    maxfev=5000
                )
                amplitude, center, sigma, tail_frac, tail_sigma = popt
                fwhm = 2.355 * sigma
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma, 'tail_fraction': tail_frac, 'tail_sigma': tail_sigma}
            
            else:
                raise ValueError(f"Unknown peak shape: {shape}")
            
            return Peak(
                energy=center,
                amplitude=amplitude,
                fwhm=fwhm,
                area=area,
                shape=shape,
                shape_params=shape_params
            )
        
        except Exception as e:
            print(f"Peak fitting failed at {initial_center:.3f} keV: {e}")
            return None
    
    @staticmethod
    def fit_multiple_peaks(energy, counts, peak_positions, shape='gaussian'):
        """
        Fit multiple peaks simultaneously
        
        Args:
            energy: Energy array
            counts: Counts array
            peak_positions: List of initial peak center energies
            shape: Peak shape to use
            
        Returns:
            List of fitted Peak objects
        """
        fitted_peaks = []
        
        for center in peak_positions:
            peak = PeakFitter.fit_single_peak(
                energy, counts, center, shape=shape
            )
            if peak is not None:
                fitted_peaks.append(peak)
        
        return fitted_peaks
    
    @staticmethod
    def calculate_residuals(energy, counts, fitted_peaks, background, shape='gaussian'):
        """
        Calculate residuals between data and fit
        
        Args:
            energy: Energy array
            counts: Original counts
            fitted_peaks: List of fitted Peak objects
            background: Background array
            shape: Peak shape used
            
        Returns:
            Residuals array
        """
        # Reconstruct fitted spectrum
        fitted_spectrum = np.copy(background)
        
        for peak in fitted_peaks:
            sigma = peak.fwhm / 2.355  # Convert FWHM to sigma
            
            if shape == 'gaussian':
                fitted_spectrum += PeakFitter.gaussian(
                    energy, peak.amplitude, peak.energy, sigma
                )
        
        residuals = counts - fitted_spectrum
        return residuals
    
    @staticmethod
    def calculate_fit_statistics(counts, fitted_counts, n_params):
        """
        Calculate goodness-of-fit statistics
        
        Args:
            counts: Original counts
            fitted_counts: Fitted counts
            n_params: Number of fit parameters
            
        Returns:
            Dictionary with chi-squared, reduced chi-squared, and R-squared
        """
        residuals = counts - fitted_counts
        
        # Chi-squared (assuming Poisson statistics)
        # Avoid division by zero
        variance = np.where(counts > 0, counts, 1)
        chi_squared = np.sum(residuals**2 / variance)
        
        # Reduced chi-squared
        n_points = len(counts)
        dof = n_points - n_params  # Degrees of freedom
        reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((counts - np.mean(counts))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'r_squared': r_squared,
            'dof': dof
        }
