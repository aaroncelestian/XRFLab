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
    
    def build_peak_positions(self, energy, counts_bg_subtracted=None, elements=None,
                             auto_find_peaks=True, tube_element='Rh',
                             excitation_kv=50.0, include_tube_lines=True, **kwargs):
        """
        Build the list of peak seed positions (element lines, tube lines, auto-find).

        Returns:
            List of dicts with keys: energy, element, line, is_tube_line
        """
        peak_positions = []

        if elements and len(elements) > 0:
            print(f"Using emission lines from {len(elements)} elements...")
            for elem in elements:
                symbol = elem.get('symbol', '')
                z = elem.get('z', 0)

                if symbol and z:
                    lines = get_element_lines(symbol, z)

                    major_lines = {
                        'K': ['Kα1', 'Kα2', 'Kβ1'],
                        'L': ['Lα1', 'Lα2', 'Lβ1', 'Lβ2'],
                        'M': ['Mα1', 'Mα2']
                    }

                    for series in ['K', 'L', 'M']:
                        for line in lines.get(series, []):
                            line_name = line['name']
                            line_energy = line['energy']

                            if line_name in major_lines.get(series, []):
                                if energy[0] <= line_energy <= energy[-1]:
                                    peak_positions.append({
                                        'energy': line_energy,
                                        'element': symbol,
                                        'line': line_name,
                                        'is_tube_line': False
                                    })

        if include_tube_lines and tube_element:
            print(f"Including {tube_element} tube lines at {excitation_kv} keV...")
            tube_lines = get_tube_lines(tube_element, excitation_kv)

            for series in ['K', 'L']:
                for line in tube_lines.get(series, []):
                    line_name = line['name']
                    line_energy = line['energy']

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

        if auto_find_peaks and counts_bg_subtracted is not None:
            print("Auto-detecting peaks...")
            auto_peaks = self.peak_fitter.find_peaks(
                energy, counts_bg_subtracted,
                prominence=kwargs.get('prominence', None),
                distance=kwargs.get('distance', None),
                height=kwargs.get('min_height', kwargs.get('height', None)),
                prominence_percent=kwargs.get('prominence_percent', None),
                min_separation_ev=kwargs.get('min_separation_ev', None),
            )

            match_tol = 0.1
            if kwargs.get('min_separation_ev') is not None:
                match_tol = max(0.05, float(kwargs['min_separation_ev']) / 1000.0)

            for peak_energy, peak_height in auto_peaks:
                near_element_line = False
                for pos in peak_positions:
                    if abs(peak_energy - pos['energy']) < match_tol:
                        near_element_line = True
                        break

                if not near_element_line:
                    peak_positions.append({
                        'energy': peak_energy,
                        'element': None,
                        'line': None,
                        'is_tube_line': False,
                    })

            print(f"Auto-detected {len(auto_peaks)} peaks "
                  f"({sum(1 for p in peak_positions if p.get('element') is None)} unknown added)")

        return peak_positions

    def fit_spectrum(self, energy, counts, elements=None, 
                    background_method='snip', peak_shape='gaussian',
                    auto_find_peaks=True, tube_element='Rh', 
                    excitation_kv=50.0, include_tube_lines=True,
                    peak_positions=None, **kwargs):
        """
        Fit XRF spectrum with background and peaks
        
        Args:
            energy: Energy array (keV)
            counts: Counts array
            elements: List of element dicts with 'symbol' and 'z' keys
            background_method: 'snip', 'polynomial', 'linear', 'adaptive', 'none'
            peak_shape: 'gaussian', 'voigt', 'pseudo_voigt'
            auto_find_peaks: If True, automatically find peaks
            peak_positions: Optional pre-built peak seeds (skips rebuild when provided)
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
        
        # Step 3: Identify peak positions (or use caller-provided list)
        if peak_positions is not None:
            peak_positions = [
                {
                    'energy': float(p['energy']),
                    'element': p.get('element'),
                    'line': p.get('line'),
                    'is_tube_line': p.get('is_tube_line', False),
                }
                for p in peak_positions
            ]
            print(f"Using {len(peak_positions)} caller-provided peak positions...")
        else:
            peak_positions = self.build_peak_positions(
                energy,
                counts_bg_subtracted=counts_bg_subtracted,
                elements=elements,
                auto_find_peaks=auto_find_peaks,
                tube_element=tube_element,
                excitation_kv=excitation_kv,
                include_tube_lines=include_tube_lines,
                **kwargs,
            )
        
        # Step 4: Fit peaks (strongest first; subtract each so overlaps don't stack)
        print(f"Fitting {len(peak_positions)} peaks using {peak_shape} shape...")
        fitted_peaks = []
        residual_counts = np.asarray(counts_bg_subtracted, dtype=float).copy()
        
        def _local_height(pos):
            e0 = pos['energy']
            idx = int(np.argmin(np.abs(energy - e0)))
            return float(residual_counts[idx])
        
        ordered_positions = sorted(peak_positions, key=_local_height, reverse=True)
        
        for pos in ordered_positions:
            known_line = bool(pos.get('element') and pos.get('line'))
            # Very weak residual peaks: lock center so LS cannot chase noise
            local_h = _local_height(pos)
            residual_max = float(np.nanmax(residual_counts)) if residual_counts.size else 0.0
            fix_center = (
                residual_max > 0
                and local_h < 0.05 * residual_max
            )
            peak = self.peak_fitter.fit_single_peak(
                energy, residual_counts,
                initial_center=pos['energy'],
                shape=peak_shape,
                known_line=known_line,
                fix_center=fix_center,
            )
            
            if peak is not None:
                # Add element information
                peak.element = pos.get('element')
                peak.line = pos.get('line')
                peak.is_tube_line = pos.get('is_tube_line', False)
                fitted_peaks.append(peak)
                
                # Remove this peak from the residual spectrum before fitting others
                if peak.shape == 'gaussian':
                    sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                    residual_counts -= self.peak_fitter.gaussian(
                        energy, peak.amplitude, peak.energy, sigma
                    )
                elif peak.shape == 'voigt':
                    sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                    gamma = peak.shape_params.get('gamma', 0.05)
                    residual_counts -= self.peak_fitter.voigt(
                        energy, peak.amplitude, peak.energy, sigma, gamma
                    )
                elif peak.shape == 'pseudo_voigt':
                    sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                    eta = peak.shape_params.get('eta', 0.5)
                    residual_counts -= self.peak_fitter.pseudo_voigt(
                        energy, peak.amplitude, peak.energy, sigma, eta
                    )
                elif peak.shape == 'hypermet':
                    sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                    tail_amp = peak.shape_params.get('tail_amplitude', 0.1)
                    tail_slope = peak.shape_params.get('tail_slope', 2.0)
                    residual_counts -= self.peak_fitter.hypermet(
                        energy, peak.amplitude, peak.energy, sigma, tail_amp, tail_slope
                    )
                elif peak.shape == 'tail_gaussian':
                    sigma = peak.shape_params.get('sigma', peak.fwhm / 2.355)
                    tail_frac = peak.shape_params.get('tail_fraction', 0.15)
                    tail_sigma = peak.shape_params.get('tail_sigma', sigma * 3)
                    residual_counts -= self.peak_fitter.tail_gaussian(
                        energy, peak.amplitude, peak.energy, sigma, tail_frac, tail_sigma
                    )
                else:
                    sigma = peak.fwhm / 2.355
                    residual_counts -= self.peak_fitter.gaussian(
                        energy, peak.amplitude, peak.energy, sigma
                    )
        
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
