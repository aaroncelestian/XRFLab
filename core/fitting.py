"""
Main fitting engine for XRF spectra
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.background import BackgroundModeler
from core.peak_fitting import PeakFitter, Peak
from core.xray_data import get_element_lines, get_tube_lines


@dataclass
class FitResult:
    """Results from spectrum fitting"""
    background: np.ndarray
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    peaks: List[Peak]
    statistics: Dict
    
    def __str__(self):
        return (f"Fit Result: {len(self.peaks)} peaks, "
                f"χ²ᵣ = {self.statistics['reduced_chi_squared']:.2f}, "
                f"R² = {self.statistics['r_squared']:.4f}")


class SpectrumFitter:
    """Main fitting engine for XRF spectra"""
    
    def __init__(self):
        self.background_modeler = BackgroundModeler()
        self.peak_fitter = PeakFitter()
    
    def fit_spectrum(self, energy, counts, elements=None, 
                    background_method='snip', peak_shape='gaussian',
                    auto_find_peaks=True, tube_element='Rh', 
                    excitation_kv=50.0, include_tube_lines=True, **kwargs):
        """
        Fit XRF spectrum with background and peaks
        
        Args:
            energy: Energy array (keV)
            counts: Counts array
            elements: List of element dicts with 'symbol' and 'z' keys
            background_method: 'snip', 'polynomial', 'linear', 'adaptive', 'none'
            peak_shape: 'gaussian', 'voigt', 'pseudo_voigt'
            auto_find_peaks: If True, automatically find peaks
            **kwargs: Additional parameters for background/peak fitting
            
        Returns:
            FitResult object
        """
        # Step 1: Estimate background
        print(f"Estimating background using {background_method} method...")
        background = self.background_modeler.estimate_background(
            energy, counts, method=background_method, **kwargs
        )
        
        # Step 2: Subtract background
        counts_bg_subtracted = self.background_modeler.subtract_background(
            counts, background
        )
        
        # Step 3: Identify peak positions
        peak_positions = []
        
        if elements and len(elements) > 0:
            # Use element emission lines as peak positions
            print(f"Using emission lines from {len(elements)} elements...")
            for elem in elements:
                symbol = elem.get('symbol', '')
                z = elem.get('z', 0)
                
                if symbol and z:
                    lines = get_element_lines(symbol, z)
                    
                    # Only add major lines to avoid fitting noise
                    # K-series: Kα1, Kα2, Kβ1 (skip weak Kβ2, Kβ3)
                    # L-series: Lα1, Lα2, Lβ1, Lβ2 (skip weak Lγ, Lβ3, Lβ4)
                    major_lines = {
                        'K': ['Kα1', 'Kα2', 'Kβ1'],
                        'L': ['Lα1', 'Lα2', 'Lβ1', 'Lβ2'],
                        'M': ['Mα1', 'Mα2']
                    }
                    
                    for series in ['K', 'L', 'M']:
                        for line in lines.get(series, []):
                            line_name = line['name']
                            line_energy = line['energy']
                            
                            # Only include major lines
                            if line_name in major_lines.get(series, []):
                                if energy[0] <= line_energy <= energy[-1]:
                                    peak_positions.append({
                                        'energy': line_energy,
                                        'element': symbol,
                                        'line': line_name,
                                        'is_tube_line': False
                                    })
        
        # Add X-ray tube lines if requested
        if include_tube_lines and tube_element:
            print(f"Including {tube_element} tube lines at {excitation_kv} keV...")
            tube_lines = get_tube_lines(tube_element, excitation_kv)
            
            for series in ['K', 'L']:
                for line in tube_lines.get(series, []):
                    line_name = line['name']
                    line_energy = line['energy']
                    
                    # Only major tube lines
                    if series == 'K' and line_name in ['Kα1', 'Kα2', 'Kβ1']:
                        if energy[0] <= line_energy <= energy[-1]:
                            peak_positions.append({
                                'energy': line_energy,
                                'element': tube_element,
                                'line': line_name,
                                'is_tube_line': True
                            })
                    elif series == 'L' and line_name in ['Lα1', 'Lβ1']:
                        if energy[0] <= line_energy <= energy[-1]:
                            peak_positions.append({
                                'energy': line_energy,
                                'element': tube_element,
                                'line': line_name,
                                'is_tube_line': True
                            })
        
        if auto_find_peaks:
            # Also find peaks automatically
            print("Auto-detecting peaks...")
            auto_peaks = self.peak_fitter.find_peaks(
                energy, counts_bg_subtracted,
                prominence=kwargs.get('prominence', None),
                distance=kwargs.get('distance', None)
            )
            
            # Add auto-detected peaks that aren't near element lines
            for peak_energy, peak_height in auto_peaks:
                # Check if this peak is near any element line
                near_element_line = False
                for pos in peak_positions:
                    if abs(peak_energy - pos['energy']) < 0.1:  # Within 0.1 keV
                        near_element_line = True
                        break
                
                if not near_element_line:
                    peak_positions.append({
                        'energy': peak_energy,
                        'element': None,
                        'line': None
                    })
        
        # Step 4: Fit peaks
        print(f"Fitting {len(peak_positions)} peaks using {peak_shape} shape...")
        fitted_peaks = []
        
        for pos in peak_positions:
            peak = self.peak_fitter.fit_single_peak(
                energy, counts_bg_subtracted,
                initial_center=pos['energy'],
                shape=peak_shape
            )
            
            if peak is not None:
                # Add element information
                peak.element = pos.get('element')
                peak.line = pos.get('line')
                peak.is_tube_line = pos.get('is_tube_line', False)
                fitted_peaks.append(peak)
        
        print(f"Successfully fitted {len(fitted_peaks)} peaks")
        
        # Step 5: Reconstruct fitted spectrum
        fitted_spectrum = np.copy(background)
        
        for peak in fitted_peaks:
            # Use the correct peak shape for reconstruction
            if peak.shape == 'gaussian':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                fitted_spectrum += self.peak_fitter.gaussian(
                    energy, peak.amplitude, peak.energy, sigma
                )
            elif peak.shape == 'voigt':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                gamma = peak.shape_params.get('gamma', 0.05)
                fitted_spectrum += self.peak_fitter.voigt(
                    energy, peak.amplitude, peak.energy, sigma, gamma
                )
            elif peak.shape == 'pseudo_voigt':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                eta = peak.shape_params.get('eta', 0.5)
                fitted_spectrum += self.peak_fitter.pseudo_voigt(
                    energy, peak.amplitude, peak.energy, sigma, eta
                )
            elif peak.shape == 'hypermet':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                tail_amp = peak.shape_params.get('tail_amplitude', 0.1)
                tail_slope = peak.shape_params.get('tail_slope', 2.0)
                fitted_spectrum += self.peak_fitter.hypermet(
                    energy, peak.amplitude, peak.energy, sigma, tail_amp, tail_slope
                )
            elif peak.shape == 'tail_gaussian':
                sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                tail_frac = peak.shape_params.get('tail_fraction', 0.15)
                tail_sigma = peak.shape_params.get('tail_sigma', sigma * 3)
                fitted_spectrum += self.peak_fitter.tail_gaussian(
                    energy, peak.amplitude, peak.energy, sigma, tail_frac, tail_sigma
                )
            else:
                # Default to Gaussian if shape not recognized
                sigma = peak.fwhm / 2.355
                fitted_spectrum += self.peak_fitter.gaussian(
                    energy, peak.amplitude, peak.energy, sigma
                )
        
        # Step 6: Calculate residuals
        residuals = counts - fitted_spectrum
        
        # Step 7: Calculate fit statistics
        n_params = len(fitted_peaks) * 3 + 1  # 3 params per peak + background
        statistics = self.peak_fitter.calculate_fit_statistics(
            counts, fitted_spectrum, n_params
        )
        
        # Add iteration count (placeholder for now)
        statistics['iterations'] = 1
        
        return FitResult(
            background=background,
            fitted_spectrum=fitted_spectrum,
            residuals=residuals,
            peaks=fitted_peaks,
            statistics=statistics
        )
    
    def identify_peaks(self, peaks, tolerance=0.05):
        """
        Identify unknown peaks by matching to element emission lines
        
        Args:
            peaks: List of Peak objects
            tolerance: Energy tolerance for matching (keV)
            
        Returns:
            List of peaks with updated element/line information
        """
        # This is a placeholder for future implementation
        # Would search through xraylib database to identify peaks
        return peaks
    
    def quantify_elements(self, peaks, experimental_params):
        """
        Quantify element concentrations from fitted peaks
        
        Args:
            peaks: List of fitted Peak objects
            experimental_params: Dict with excitation energy, current, etc.
            
        Returns:
            Dict with element concentrations (normalized to 100%)
        """
        # Sum all peak areas for each element (K, L, M lines combined)
        # EXCLUDE tube lines from quantification
        element_totals = {}
        element_lines = {}  # Track which lines contributed
        
        for peak in peaks:
            # Skip tube lines - they're not from the sample
            if peak.is_tube_line:
                continue
            
            if peak.element:
                if peak.element not in element_totals:
                    element_totals[peak.element] = 0.0
                    element_lines[peak.element] = []
                
                # Sum all lines for this element
                element_totals[peak.element] += peak.area
                element_lines[peak.element].append(peak.line)
        
        # Calculate total intensity
        total_intensity = sum(element_totals.values())
        
        if total_intensity == 0:
            return {}
        
        # Normalize to 100% (weight percent)
        concentrations = {}
        for element, total_area in element_totals.items():
            weight_percent = (total_area / total_intensity) * 100.0
            
            concentrations[element] = {
                'concentration': weight_percent,
                'error': weight_percent * 0.1,  # 10% relative error placeholder
                'lines': element_lines[element],  # All contributing lines
                'total_area': total_area
            }
        
        return concentrations
