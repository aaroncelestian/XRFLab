"""
Advanced calibration with peak shape refinement

Uses least-squares optimization to refine:
- FWHM parameters (FWHM_0, epsilon)
- Peak shape parameters (tail amplitude, tail slope)
- Individual peak intensity scale factors
- Detector efficiency parameters

Goal: R² > 0.98, χ² < 10
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AdvancedCalibrationParams:
    """Parameters for advanced calibration"""
    # FWHM parameters
    fwhm_0: float
    epsilon: float
    
    # Peak shape parameters (Hypermet/Tail-Gaussian)
    tail_amplitude: float = 0.05  # Fraction of peak in tail
    tail_slope: float = 2.0  # Tail decay rate
    
    # Detector efficiency parameters (polynomial coefficients)
    eff_a: float = 1.0
    eff_b: float = 0.0
    eff_c: float = 0.0
    
    # Intensity scale factors per element (multiplicative corrections to FP)
    intensity_scales: Dict[str, float] = None
    
    def __post_init__(self):
        if self.intensity_scales is None:
            self.intensity_scales = {}


class AdvancedCalibrator:
    """
    Advanced calibration with full peak shape refinement
    
    Optimizes all parameters simultaneously using least-squares
    """
    
    def __init__(self):
        from core.peak_fitting import PeakFitter
        self.peak_fitter = PeakFitter()
    
    def calibrate_with_shape_refinement(self,
                                       energy: np.ndarray,
                                       counts: np.ndarray,
                                       element_data: List[Dict],
                                       initial_params: AdvancedCalibrationParams = None) -> AdvancedCalibrationParams:
        """
        Calibrate with full peak shape refinement
        
        Args:
            energy: Energy array
            counts: Measured counts
            element_data: List of {element, line, energy, relative_intensity}
            initial_params: Initial parameter guesses
            
        Returns:
            Optimized parameters
        """
        if initial_params is None:
            initial_params = AdvancedCalibrationParams(
                fwhm_0=0.080,
                epsilon=0.002,
                tail_amplitude=0.05,
                tail_slope=2.0
            )
        
        # Build parameter vector
        # [fwhm_0, epsilon, tail_amp, tail_slope, eff_a, eff_b, eff_c, scale_1, scale_2, ...]
        
        # Get unique elements
        elements = list(set([d['element'] for d in element_data]))
        
        p0 = [
            initial_params.fwhm_0,
            initial_params.epsilon,
            initial_params.tail_amplitude,
            initial_params.tail_slope,
            initial_params.eff_a,
            initial_params.eff_b,
            initial_params.eff_c
        ]
        
        # Add intensity scale factors (one per element)
        for elem in elements:
            p0.append(1.0)  # Start with no scaling
        
        # Parameter bounds
        bounds_lower = [
            0.020,  # fwhm_0
            0.0005,  # epsilon
            0.0,  # tail_amp
            0.5,  # tail_slope
            0.5,  # eff_a
            -0.5,  # eff_b
            -0.1,  # eff_c
        ] + [0.5] * len(elements)  # intensity scales
        
        bounds_upper = [
            0.200,  # fwhm_0
            0.0100,  # epsilon
            0.3,  # tail_amp
            10.0,  # tail_slope
            1.5,  # eff_a
            0.5,  # eff_b
            0.1,  # eff_c
        ] + [2.0] * len(elements)  # intensity scales
        
        bounds = (bounds_lower, bounds_upper)
        
        print("Starting advanced calibration with peak shape refinement...")
        print(f"  Optimizing {len(p0)} parameters")
        print(f"  {len(element_data)} peaks")
        
        # Optimize
        try:
            result = optimize.least_squares(
                self._residual_function,
                p0,
                bounds=bounds,
                args=(energy, counts, element_data, elements),
                verbose=2,
                max_nfev=1000,
                ftol=1e-8,
                xtol=1e-8
            )
            
            # Extract optimized parameters
            params_opt = AdvancedCalibrationParams(
                fwhm_0=result.x[0],
                epsilon=result.x[1],
                tail_amplitude=result.x[2],
                tail_slope=result.x[3],
                eff_a=result.x[4],
                eff_b=result.x[5],
                eff_c=result.x[6]
            )
            
            # Extract intensity scales
            params_opt.intensity_scales = {}
            for i, elem in enumerate(elements):
                params_opt.intensity_scales[elem] = result.x[7 + i]
            
            # Calculate final fit quality
            residuals = self._residual_function(result.x, energy, counts, element_data, elements)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((counts - np.mean(counts))**2)
            r_squared = 1 - (ss_res / ss_tot)
            chi_squared = ss_res / len(counts)
            
            print(f"\nAdvanced calibration complete!")
            print(f"  FWHM_0: {params_opt.fwhm_0*1000:.1f} eV")
            print(f"  epsilon: {params_opt.epsilon*1000:.2f} eV")
            print(f"  Tail amplitude: {params_opt.tail_amplitude:.3f}")
            print(f"  Tail slope: {params_opt.tail_slope:.2f}")
            print(f"  R²: {r_squared:.4f}")
            print(f"  χ²: {chi_squared:.2f}")
            
            return params_opt, r_squared, chi_squared
            
        except Exception as e:
            print(f"Advanced calibration failed: {e}")
            return initial_params, 0.0, 1e10
    
    def _residual_function(self,
                          params: np.ndarray,
                          energy: np.ndarray,
                          measured_counts: np.ndarray,
                          element_data: List[Dict],
                          elements: List[str]) -> np.ndarray:
        """
        Calculate residuals for least-squares optimization
        
        Returns:
            Residuals array (measured - calculated)
        """
        # Unpack parameters
        fwhm_0 = params[0]
        epsilon = params[1]
        tail_amp = params[2]
        tail_slope = params[3]
        eff_a = params[4]
        eff_b = params[5]
        eff_c = params[6]
        
        intensity_scales = {}
        for i, elem in enumerate(elements):
            intensity_scales[elem] = params[7 + i]
        
        # Calculate spectrum with these parameters
        calculated = np.zeros_like(energy)
        
        for line_data in element_data:
            line_energy = line_data['energy']
            base_intensity = line_data['relative_intensity']
            element = line_data['element']
            
            # Apply intensity scale factor
            intensity = base_intensity * intensity_scales.get(element, 1.0)
            
            # Apply detector efficiency
            eff = eff_a + eff_b * line_energy + eff_c * line_energy**2
            eff = np.clip(eff, 0.1, 1.5)
            intensity *= eff
            
            # Calculate FWHM
            fwhm = np.sqrt(fwhm_0**2 + 2.35 * epsilon * line_energy)
            sigma = fwhm / 2.355
            
            # Add peak with tail (Hypermet-like)
            # Main Gaussian
            gaussian = intensity * np.exp(-(energy - line_energy)**2 / (2 * sigma**2))
            
            # Low-energy tail
            tail = np.zeros_like(energy)
            mask = energy < line_energy
            if np.any(mask):
                tail[mask] = intensity * tail_amp * np.exp(tail_slope * (energy[mask] - line_energy) / sigma)
            
            calculated += gaussian + tail
        
        # Normalize for comparison
        if np.max(calculated) > 0:
            scale = np.sum(measured_counts * calculated) / np.sum(calculated * calculated)
            calculated *= scale
        
        # Return residuals (weighted by sqrt of counts for Poisson statistics)
        weights = 1.0 / np.sqrt(np.maximum(measured_counts, 1.0))
        residuals = weights * (measured_counts - calculated)
        
        return residuals
