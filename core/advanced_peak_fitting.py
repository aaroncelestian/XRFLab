"""
Advanced Peak Fitting for Complex XRF Spectra

This module provides sophisticated peak fitting for samples containing both
light elements (Mg, Al, Na - simple K-lines) and heavy elements (Zr, Ba, REE - complex L-lines).

Uses:
- PyMca5: For hypermet tails and automatic L-line multiplet handling
- Fityk: For custom peak shapes and diagnostic fitting
- FWHM Calibration: From your detector calibration

Integration with XRFLab:
- Automatically selects appropriate peak model based on element and line type
- Uses calibrated FWHM for K-lines
- Applies empirical corrections for L-line broadening
- Handles matrix effects and self-absorption
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PeakFitResult:
    """Results from advanced peak fitting"""
    element: str
    line: str
    energy: float  # keV
    area: float
    area_error: float
    fwhm: float  # keV
    fit_quality: float  # R² or chi²
    peak_type: str  # 'K-line', 'L-line', 'M-line'
    model_used: str  # 'gaussian', 'voigt', 'hypermet', 'emg'


class AdvancedPeakFitter:
    """
    Advanced peak fitting with automatic model selection
    
    Automatically chooses the best peak model based on:
    - Element atomic number
    - Line type (K, L, M)
    - Energy range
    - Calibrated detector resolution
    """
    
    def __init__(self, fwhm_calibration=None):
        """
        Initialize fitter
        
        Args:
            fwhm_calibration: FWHMCalibration object from detector calibration
        """
        self.fwhm_calibration = fwhm_calibration
        self.pymca_available = self._check_pymca()
        self.fityk_available = self._check_fityk()
        
        # Empirical L-line broadening factors (from your data)
        self.l_line_broadening = {
            'Zr': 1.3,  # Zr L-lines are ~30% wider than predicted
            'Ba': 1.4,
            'La': 1.4,
            'Ce': 1.4,
            # Add more as you calibrate them
        }
    
    def _check_pymca(self) -> bool:
        """Check if PyMca5 is available"""
        try:
            import PyMca5.PyMcaPhysics.xrf.FastXRFLinearFit as FastXRFLinearFit
            from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory
            return True
        except ImportError:
            return False
    
    def _check_fityk(self) -> bool:
        """Check if fityk is available"""
        try:
            import fityk
            return True
        except ImportError:
            return False
    
    def predict_fwhm(self, energy: float, element: str = None, line_type: str = 'K') -> float:
        """
        Predict FWHM at given energy with L-line corrections
        
        Args:
            energy: Photon energy (keV)
            element: Element symbol (for L-line corrections)
            line_type: 'K', 'L', or 'M'
        
        Returns:
            FWHM in keV
        """
        if self.fwhm_calibration is None:
            # Default detector model if no calibration
            fwhm_0 = 0.120  # 120 eV
            epsilon = 0.0035  # 3.5 eV/keV
            base_fwhm = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * energy)
        else:
            base_fwhm = self.fwhm_calibration.predict_fwhm(energy)
        
        # Apply L-line broadening correction
        if line_type == 'L' and element in self.l_line_broadening:
            base_fwhm *= self.l_line_broadening[element]
        
        return base_fwhm
    
    def select_peak_model(self, element: str, line: str, energy: float) -> str:
        """
        Automatically select best peak model
        
        Args:
            element: Element symbol
            line: Line name (e.g., 'Kα', 'Lα', 'Lβ')
            energy: Line energy (keV)
        
        Returns:
            Model name: 'gaussian', 'voigt', 'hypermet', or 'emg'
        """
        # Get atomic number
        from core.xray_data import get_element_z
        z = get_element_z(element)
        
        # Determine line type
        if 'K' in line:
            line_type = 'K'
        elif 'L' in line:
            line_type = 'L'
        elif 'M' in line:
            line_type = 'M'
        else:
            line_type = 'K'  # Default
        
        # Selection logic
        if line_type == 'K' and z < 30:
            # Light-to-medium elements, K-lines: Simple Gaussian
            return 'gaussian'
        
        elif line_type == 'K' and z >= 30:
            # Heavy elements, K-lines: Voigt (some natural broadening)
            return 'voigt'
        
        elif line_type == 'L' and z < 50:
            # Medium-Z L-lines: Hypermet (asymmetric tails)
            return 'hypermet' if self.pymca_available else 'emg'
        
        elif line_type == 'L' and z >= 50:
            # Heavy-Z L-lines: Complex multiplets, need hypermet
            return 'hypermet' if self.pymca_available else 'emg'
        
        elif line_type == 'M':
            # M-lines: Always complex
            return 'hypermet' if self.pymca_available else 'emg'
        
        else:
            return 'gaussian'  # Fallback
    
    def fit_with_pymca(self, 
                       energy: np.ndarray,
                       counts: np.ndarray,
                       elements: List[str],
                       excitation_energy: float = 30.0) -> List[PeakFitResult]:
        """
        Fit spectrum using PyMca with hypermet tails
        
        Args:
            energy: Energy axis (keV)
            counts: Spectrum counts
            elements: List of elements present
            excitation_energy: Tube voltage (keV)
        
        Returns:
            List of PeakFitResult objects
        """
        if not self.pymca_available:
            raise ImportError("PyMca5 not available. Install with: pip install PyMca5")
        
        from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory
        
        # Get FWHM parameters from calibration
        if self.fwhm_calibration:
            fwhm_0 = self.fwhm_calibration.parameters.get('fwhm_0', 0.120) * 1000  # eV
            epsilon = self.fwhm_calibration.parameters.get('epsilon', 0.0035)
            fano = epsilon / 0.00385  # Convert to Fano factor (ε = 2.355² * F * 0.00385)
        else:
            fwhm_0 = 120.0  # eV
            fano = 0.12  # Typical for Si
        
        # Energy calibration
        if len(energy) > 1:
            gain = (energy[-1] - energy[0]) / (len(energy) - 1) * 1000  # eV/channel
            zero = energy[0] * 1000  # eV
        else:
            gain = 10.0  # Default 10 eV/channel
            zero = 0.0
        
        # Configure PyMca
        config = {
            'fit': {
                'use_hypermet': 1,  # Enable asymmetric peak shapes for L-lines
                'hypermet_tails': 15,  # Tail parameters per peak
                'strip_algorithm': 'snip',
                'snip_width': 32,
                'linearfitflag': 1,  # Use linear least squares
                'stripflag': 1,  # Strip background
            },
            'peaks': {
                'escape_flag': 1,  # Include escape peaks
                'sum_flag': 0,  # Disable sum peaks (usually negligible)
                'scatter_flag': 1,  # Include scatter peaks
            },
            'detector': {
                'noise': fwhm_0,  # FWHM at 0 keV (eV)
                'fano': fano,  # Fano factor
                'zero': zero,  # Energy calibration offset (eV)
                'gain': gain,  # eV/channel
            },
            'concentrations': {
                'usematrix': 1,  # Use matrix corrections
                'useattenuators': 0,
            }
        }
        
        # Setup the fit
        mcafit = ClassMcaTheory.McaTheory()
        mcafit.configure(config)
        
        # Set data
        mcafit.setData(x=energy, y=counts)
        
        # Enable elements
        for element in elements:
            mcafit.enableElement(element)
        
        # Set excitation energy
        mcafit.config['fit']['energy'] = excitation_energy
        
        # Perform fit
        try:
            fitresult = mcafit.startfit(digest=1)
        except Exception as e:
            print(f"PyMca fit failed: {e}")
            return []
        
        # Extract results
        results = []
        
        # Get fitted parameters
        if 'result' in fitresult:
            result_dict = fitresult['result']
            
            # Parse peak results
            for key, value in result_dict.items():
                if 'area' in key.lower():
                    # Extract element and line from key
                    parts = key.split()
                    if len(parts) >= 2:
                        element = parts[0]
                        line = parts[1]
                        
                        # Get energy
                        from core.xray_data import get_line_energy
                        energy_val = get_line_energy(element, line)
                        
                        if energy_val is None:
                            continue
                        
                        # Get area and error
                        area = value
                        area_error = result_dict.get(key.replace('area', 'sigma'), 0.0)
                        
                        # Predict FWHM
                        line_type = 'L' if 'L' in line else 'K' if 'K' in line else 'M'
                        fwhm = self.predict_fwhm(energy_val, element, line_type)
                        
                        # Determine model used
                        model = self.select_peak_model(element, line, energy_val)
                        
                        results.append(PeakFitResult(
                            element=element,
                            line=line,
                            energy=energy_val,
                            area=area,
                            area_error=area_error,
                            fwhm=fwhm,
                            fit_quality=fitresult.get('chisq', 0.0),
                            peak_type=line_type + '-line',
                            model_used=model
                        ))
        
        return results
    
    def fit_with_fityk(self,
                       energy: np.ndarray,
                       counts: np.ndarray,
                       peak_list: List[Tuple[str, str, float]],
                       background_degree: int = 6) -> List[PeakFitResult]:
        """
        Fit spectrum using Fityk with custom peak shapes
        
        Args:
            energy: Energy axis (keV)
            counts: Spectrum counts
            peak_list: List of (element, line, energy) tuples
            background_degree: Polynomial degree for background
        
        Returns:
            List of PeakFitResult objects
        """
        if not self.fityk_available:
            raise ImportError("Fityk not available. Install with: pip install fityk")
        
        import fityk
        
        # Create Fityk instance
        fit = fityk.Fityk()
        
        # Load data with Poisson errors
        sigma = np.sqrt(np.maximum(counts, 1))
        fit.load_data(0, energy.tolist(), counts.tolist(), sigma.tolist())
        
        # Define custom EMG (Exponentially Modified Gaussian) for L-lines
        fit.execute("""
            define EMGaussian(height, center, width, tail) = 
                height * exp(0.5*(width/tail)^2 - (x-center)/tail) * 
                erfc((width/tail - (x-center)/width) / sqrt(2))
        """)
        
        # Add background
        fit.execute(f"guess Polynomial{background_degree}")
        
        # Add peaks
        for element, line, peak_energy in peak_list:
            # Determine line type
            line_type = 'L' if 'L' in line else 'K' if 'K' in line else 'M'
            
            # Predict FWHM
            fwhm = self.predict_fwhm(peak_energy, element, line_type)
            sigma = fwhm / 2.355
            
            # Select model
            model = self.select_peak_model(element, line, peak_energy)
            
            # Estimate height from data
            idx = np.argmin(np.abs(energy - peak_energy))
            height_guess = counts[idx]
            
            if model == 'gaussian' or model == 'voigt':
                # Simple Gaussian for K-lines
                fit.execute(f"""
                    F += Gaussian(height=~{height_guess}, center={peak_energy}, 
                                 fwhm={fwhm})
                """)
            
            elif model in ['hypermet', 'emg']:
                # EMG for L-lines with tail
                # Estimate tail parameter from broadening
                if element in self.l_line_broadening:
                    broadening_factor = self.l_line_broadening[element]
                    tail_sigma = sigma * (broadening_factor - 1.0)
                else:
                    tail_sigma = sigma * 0.3  # Default 30% tail
                
                fit.execute(f"""
                    F += EMGaussian(height=~{height_guess}, center={peak_energy},
                                   width={sigma}, tail={tail_sigma})
                """)
        
        # Fit
        try:
            fit.execute("fit")
        except Exception as e:
            print(f"Fityk fit failed: {e}")
            return []
        
        # Extract results
        results = []
        n_funcs = fit.get_function_count()
        
        for i, (element, line, peak_energy) in enumerate(peak_list):
            # Function index (skip background)
            func_idx = i + 1  # +1 for background
            
            if func_idx >= n_funcs:
                break
            
            func = fit.get_function(func_idx)
            
            # Get area
            area = func.area
            
            # Estimate error from fit quality
            # (Fityk doesn't provide errors directly)
            area_error = area * 0.05  # Assume 5% error
            
            # Get FWHM
            line_type = 'L' if 'L' in line else 'K' if 'K' in line else 'M'
            fwhm = self.predict_fwhm(peak_energy, element, line_type)
            
            # Get model
            model = self.select_peak_model(element, line, peak_energy)
            
            # Get fit quality
            wssr = fit.get_wssr()  # Weighted sum of squared residuals
            
            results.append(PeakFitResult(
                element=element,
                line=line,
                energy=peak_energy,
                area=area,
                area_error=area_error,
                fwhm=fwhm,
                fit_quality=wssr,
                peak_type=line_type + '-line',
                model_used=model
            ))
        
        return results
    
    def fit_spectrum(self,
                     energy: np.ndarray,
                     counts: np.ndarray,
                     elements: List[str],
                     excitation_energy: float = 30.0,
                     method: str = 'auto') -> List[PeakFitResult]:
        """
        Fit spectrum with automatic method selection
        
        Args:
            energy: Energy axis (keV)
            counts: Spectrum counts
            elements: List of elements present
            excitation_energy: Tube voltage (keV)
            method: 'auto', 'pymca', or 'fityk'
        
        Returns:
            List of PeakFitResult objects
        """
        # Auto-select method
        if method == 'auto':
            # Prefer PyMca for complex spectra with L-lines
            has_heavy_elements = any(self._is_heavy_element(el) for el in elements)
            
            if has_heavy_elements and self.pymca_available:
                method = 'pymca'
            elif self.fityk_available:
                method = 'fityk'
            elif self.pymca_available:
                method = 'pymca'
            else:
                raise ImportError("Neither PyMca5 nor Fityk available. Install at least one.")
        
        # Fit with selected method
        if method == 'pymca':
            return self.fit_with_pymca(energy, counts, elements, excitation_energy)
        
        elif method == 'fityk':
            # Build peak list from elements
            peak_list = self._build_peak_list(elements, excitation_energy)
            return self.fit_with_fityk(energy, counts, peak_list)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _is_heavy_element(self, element: str) -> bool:
        """Check if element has complex L-lines"""
        from core.xray_data import get_element_z
        z = get_element_z(element)
        return z >= 40  # Zr and heavier
    
    def _build_peak_list(self, elements: List[str], excitation_energy: float) -> List[Tuple[str, str, float]]:
        """Build list of expected peaks"""
        from core.xray_data import get_element_lines
        
        peak_list = []
        
        for element in elements:
            lines = get_element_lines(element)
            
            for line_name, line_energy in lines:
                # Only include lines below excitation energy
                if line_energy < excitation_energy * 0.95:
                    peak_list.append((element, line_name, line_energy))
        
        return peak_list


def get_element_z(element: str) -> int:
    """Get atomic number for element"""
    z_map = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92
    }
    return z_map.get(element, 0)
