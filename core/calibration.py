"""
Instrument calibration using reference standards
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass
from typing import Dict, List, Tuple
from core.background import BackgroundModeler
from core.peak_fitting import PeakFitter, Peak
from core.xray_data import get_element_lines, get_tube_lines
from core.fundamental_parameters import FundamentalParameters
from core.fisx_integration import FisxCalculator, convert_fisx_to_element_data


@dataclass
class CalibrationResult:
    """Results from instrument calibration"""
    fwhm_0: float  # keV
    epsilon: float  # keV
    voigt_gamma_ratio: float  # gamma/sigma ratio
    efficiency_params: Dict  # Energy-dependent efficiency
    chi_squared: float
    r_squared: float
    success: bool
    message: str


class InstrumentCalibrator:
    """Calibrate instrument parameters using reference standards"""
    
    def __init__(self):
        self.peak_fitter = PeakFitter()
    
    def calibrate(self, 
                  energy: np.ndarray,
                  counts: np.ndarray,
                  reference_concentrations: Dict[str, float],
                  excitation_energy: float = 50.0,
                  initial_params: Dict = None,
                  use_measured_intensities: bool = True,
                  experimental_params: Dict = None) -> CalibrationResult:
        """
        Calibrate instrument parameters using a reference standard
        
        Args:
            energy: Energy array (keV)
            counts: Measured counts
            reference_concentrations: Dict of {element: concentration_ppm}
            excitation_energy: X-ray tube voltage (keV)
            initial_params: Initial guesses for parameters
            
        Returns:
            CalibrationResult with optimized parameters
        """
        # Set initial parameter guesses
        if initial_params is None:
            initial_params = {
                'fwhm_0': 0.050,  # keV
                'epsilon': 0.0015,  # keV
                'voigt_gamma_ratio': 0.15,  # gamma/sigma
            }
        
        # Convert to parameter array for optimization
        # Use Gaussian peaks (no Voigt gamma) for sharper, more realistic XRF peaks
        p0 = [
            initial_params['fwhm_0'],
            initial_params['epsilon']
        ]
        
        # Parameter bounds - must be a list of (min, max) tuples for each parameter
        bounds = [
            (0.020, 0.200),   # FWHM_0: 20-200 eV
            (0.0005, 0.0100)  # EPSILON: 0.5-10 eV  
        ]
        
        # Prepare element data
        try:
            if use_measured_intensities:
                # Extract intensities from measured spectrum
                element_data = self._prepare_element_data_from_spectrum(
                    energy, counts, reference_concentrations, excitation_energy, initial_params
                )
            else:
                # Use concentrations as proxy (less accurate)
                element_data = self._prepare_element_data(
                    reference_concentrations, 
                    excitation_energy,
                    experimental_params
                )
        except Exception as e:
            print(f"Error preparing element data: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("Starting instrument calibration...")
        print(f"  Total elements in CSV: {len(reference_concentrations)}")
        print(f"  Element lines for calibration: {len(element_data)}")
        print(f"  Initial FWHM_0: {p0[0]:.4f} keV")
        print(f"  Initial EPSILON: {p0[1]:.6f} keV")
        
        # Optimize parameters
        try:
            # Test objective function first
            print("Testing objective function...")
            test_chi2 = self._objective_function(p0, energy, counts, element_data)
            print(f"  Initial χ²: {test_chi2:.2f}")
            
            print("Starting optimization...")
            try:
                result = optimize.minimize(
                    self._objective_function,
                    p0,
                    args=(energy, counts, element_data),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'disp': True}
                )
            except Exception as opt_error:
                print(f"Optimization error: {opt_error}")
                print(f"Error type: {type(opt_error)}")
                import traceback
                traceback.print_exc()
                raise
            
            print(f"Optimization complete. Success: {result.success}")
            print(f"Result.x type: {type(result.x)}, length: {len(result.x)}")
            print(f"Result.x values: {result.x}")
            
            # Extract optimized parameters
            if len(result.x) != 2:
                raise ValueError(f"Expected 2 optimized parameters, got {len(result.x)}: {result.x}")
            
            fwhm_0_opt = result.x[0]
            epsilon_opt = result.x[1]
            gamma_ratio_opt = 0.0  # Not used for Gaussian peaks
            
            # Calculate final fit quality
            calculated_spectrum = self._calculate_spectrum(
                energy, element_data, result.x
            )
            
            # Scale to match measured
            scale_factor = np.sum(counts * calculated_spectrum) / np.sum(calculated_spectrum * calculated_spectrum)
            calculated_spectrum_scaled = calculated_spectrum * scale_factor
            
            residuals = counts - calculated_spectrum_scaled
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((counts - np.mean(counts))**2)
            r_squared = 1 - (ss_res / ss_tot)
            chi_squared = ss_res / len(counts)
            
            print(f"\nCalibration complete!")
            print(f"  Optimized FWHM_0: {fwhm_0_opt:.4f} keV")
            print(f"  Optimized EPSILON: {epsilon_opt:.6f} keV")
            print(f"  Optimized gamma/sigma: {gamma_ratio_opt:.3f}")
            print(f"  R²: {r_squared:.4f}")
            print(f"  χ²: {chi_squared:.2f}")
            
            return CalibrationResult(
                fwhm_0=fwhm_0_opt,
                epsilon=epsilon_opt,
                voigt_gamma_ratio=gamma_ratio_opt,
                efficiency_params={},  # TODO: implement efficiency calibration
                chi_squared=chi_squared,
                r_squared=r_squared,
                success=result.success,
                message=result.message
            )
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return CalibrationResult(
                fwhm_0=p0[0],
                epsilon=p0[1],
                voigt_gamma_ratio=p0[2],
                efficiency_params={},
                chi_squared=np.inf,
                r_squared=0.0,
                success=False,
                message=str(e)
            )
    
    def _prepare_element_data_from_spectrum(self,
                                            energy: np.ndarray,
                                            counts: np.ndarray,
                                            concentrations: Dict[str, float],
                                            excitation_energy: float,
                                            initial_params: Dict) -> List[Dict]:
        """
        Extract element line intensities from measured spectrum
        
        Args:
            energy: Energy array
            counts: Measured counts
            concentrations: Dict of {element: concentration_ppm}
            excitation_energy: Excitation energy (keV)
            initial_params: Initial FWHM parameters
            
        Returns:
            List of dicts with element, line, energy, relative_intensity (from measured data)
        """
        element_data = []
        
        # Get atomic numbers for all elements
        z_map = {
            'Li': 3, 'Be': 4, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'As': 33,
            'Se': 34, 'Y': 39, 'Nb': 41, 'Mo': 42, 'Sr': 38, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
            'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Hg': 80, 'Pb': 82,
            'Th': 90, 'Cd': 48
        }
        
        fwhm_0 = initial_params.get('fwhm_0', 0.050)
        epsilon = initial_params.get('epsilon', 0.0015)
        
        # Estimate and subtract background first
        from core.background import BackgroundModeler
        bg_modeler = BackgroundModeler()
        background = bg_modeler.estimate_background(energy, counts, method='snip', window_length=50)
        counts_bg_subtracted = counts - background
        
        for element, conc_ppm in concentrations.items():
            if element not in z_map:
                continue
            
            # Skip trace elements below 100 ppm
            if conc_ppm < 100:
                continue
            
            z = z_map[element]
            try:
                lines = get_element_lines(element, z)
            except Exception as e:
                continue
            
            # Get major K and L lines
            for series in ['K', 'L']:
                for line in lines.get(series, []):
                    line_energy = line['energy']
                    line_name = line['name']
                    
                    # Only include lines below excitation energy
                    if line_energy >= excitation_energy:
                        continue
                    
                    # Only major lines
                    if series == 'K' and line_name not in ['Kα1', 'Kα2', 'Kβ1']:
                        continue
                    if series == 'L' and line_name not in ['Lα1', 'Lα2', 'Lβ1']:
                        continue
                    
                    # Measure intensity from spectrum at this energy
                    # Use a window around the expected peak position
                    fwhm_est = np.sqrt(fwhm_0**2 + 2.35 * epsilon * line_energy)
                    window = 2.0 * fwhm_est  # ±2 FWHM
                    
                    mask = np.abs(energy - line_energy) < window
                    if np.sum(mask) > 0:
                        # Use max counts in window as intensity (background-subtracted)
                        measured_intensity = np.max(counts_bg_subtracted[mask])
                        
                        # Only include if there's significant signal
                        if measured_intensity > 20:
                            element_data.append({
                                'element': element,
                                'line': line_name,
                                'energy': line_energy,
                                'relative_intensity': measured_intensity
                            })
        
        print(f"  Extracted {len(element_data)} lines with measured intensities")
        return element_data
    
    def _prepare_element_data(self, 
                              concentrations: Dict[str, float],
                              excitation_energy: float,
                              experimental_params: Dict = None) -> List[Dict]:
        """
        Prepare element emission line data with FP-calculated intensities
        
        Args:
            concentrations: Dict of {element: concentration_ppm}
            excitation_energy: Excitation energy (keV)
            
        Returns:
            List of dicts with element, line, energy, relative_intensity (from FP)
        """
        element_data = []
        
        # Convert ppm to weight fractions
        composition = {}
        for elem, ppm in concentrations.items():
            if ppm > 0:
                composition[elem] = ppm / 1e6  # ppm to weight fraction
        
        # Normalize to sum = 1
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}
        
        # Use fisx (PyMca) for highly accurate intensity calculations
        # Includes secondary and tertiary fluorescence
        try:
            # Use experimental parameters if provided
            incident_angle = 45.0
            takeoff_angle = 45.0
            if experimental_params:
                incident_angle = experimental_params.get('incident_angle', 45.0)
                takeoff_angle = experimental_params.get('takeoff_angle', 45.0)
            
            fisx_calc = FisxCalculator(
                excitation_energy=excitation_energy,
                incident_angle=incident_angle,
                takeoff_angle=takeoff_angle
            )
            intensities = fisx_calc.calculate_intensities(composition, thickness=0.1)  # Infinite thickness
            
            # Convert fisx results to element_data format
            element_data = convert_fisx_to_element_data(intensities, excitation_energy)
            
            print(f"  Calculated {len(element_data)} lines using fisx (PyMca FP)")
            return element_data
            
        except Exception as e:
            print(f"  fisx calculation failed: {e}, falling back to simplified FP")
            # Fallback to our simplified FP if fisx fails
            fp = FundamentalParameters(excitation_energy=excitation_energy)
            intensities = fp.calculate_spectrum_intensities(composition)
            
            # Convert to element_data format
            for element, lines in intensities.items():
                for line_name, intensity in lines.items():
                    # Get line energy
                    z_map = {
                        'Al': 13, 'Si': 14, 'K': 19, 'Ca': 20, 'Ti': 22,
                        'Cr': 24, 'Mn': 25, 'Fe': 26, 'Zn': 30, 'As': 33, 'Pb': 82
                    }
                    if element in z_map:
                        from core.xray_data import get_element_lines
                        z = z_map[element]
                        lines_data = get_element_lines(element, z)
                        
                        # Find energy for this line
                        for series in ['K', 'L', 'M']:
                            for line_info in lines_data.get(series, []):
                                if line_info['name'] == line_name:
                                    element_data.append({
                                        'element': element,
                                        'line': line_name,
                                        'energy': line_info['energy'],
                                        'relative_intensity': intensity
                                    })
                                    break
            
            print(f"  Calculated {len(element_data)} lines using simplified FP")
            return element_data
        
        # Get atomic numbers for all elements
        z_map = {
            'Li': 3, 'Be': 4, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'As': 33,
            'Se': 34, 'Y': 39, 'Nb': 41, 'Mo': 42, 'Sr': 38, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
            'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Hg': 80, 'Pb': 82,
            'Th': 90, 'Cd': 48
        }
        
        for element, conc_ppm in concentrations.items():
            if element not in z_map:
                print(f"  Skipping {element} (not in z_map)")
                continue
            
            # Skip trace elements below 100 ppm for calibration
            if conc_ppm < 100:
                continue
            
            z = z_map[element]
            try:
                lines = get_element_lines(element, z)
            except Exception as e:
                print(f"  Error getting lines for {element}: {e}")
                continue
            
            # Get major K and L lines
            for series in ['K', 'L']:
                for line in lines.get(series, []):
                    line_energy = line['energy']
                    line_name = line['name']
                    
                    # Only include lines below excitation energy
                    if line_energy >= excitation_energy:
                        continue
                    
                    # Only major lines
                    if series == 'K' and line_name not in ['Kα1', 'Kα2', 'Kβ1']:
                        continue
                    if series == 'L' and line_name not in ['Lα1', 'Lα2', 'Lβ1']:
                        continue
                    
                    # Relative intensity based on concentration and line type
                    # Kα1 is strongest, Kα2 ~50%, Kβ1 ~20%
                    intensity_factors = {
                        'Kα1': 1.0, 'Kα2': 0.5, 'Kβ1': 0.2,
                        'Lα1': 1.0, 'Lα2': 0.1, 'Lβ1': 0.7
                    }
                    
                    relative_intensity = conc_ppm * intensity_factors.get(line_name, 0.1)
                    
                    element_data.append({
                        'element': element,
                        'line': line_name,
                        'energy': line_energy,
                        'relative_intensity': relative_intensity
                    })
        
        return element_data
    
    def _calculate_spectrum(self,
                           energy: np.ndarray,
                           element_data: List[Dict],
                           params: np.ndarray) -> np.ndarray:
        """
        Calculate synthetic spectrum with given parameters
        
        Args:
            energy: Energy array
            element_data: List of element line data
            params: [fwhm_0, epsilon, gamma_ratio]
            
        Returns:
            Calculated spectrum
        """
        try:
            fwhm_0, epsilon = params
        except ValueError as e:
            print(f"Error unpacking params: {params}")
            print(f"  Expected 2 values, got {len(params)}")
            raise
        
        spectrum = np.zeros_like(energy)
        
        for line_data in element_data:
            try:
                line_energy = line_data['energy']
                intensity = line_data['relative_intensity']
                
                # Calculate FWHM at this energy
                fwhm = np.sqrt(fwhm_0**2 + 2.35 * epsilon * line_energy)
                sigma = fwhm / 2.355
                
                # Add Gaussian peak (sharper than Voigt for XRF)
                spectrum += self.peak_fitter.gaussian(
                    energy, intensity, line_energy, sigma
                )
            except Exception as e:
                print(f"Error calculating peak for {line_data}: {e}")
                raise
        
        return spectrum
    
    def _objective_function(self,
                           params: np.ndarray,
                           energy: np.ndarray,
                           measured_counts: np.ndarray,
                           element_data: List[Dict]) -> float:
        """
        Objective function to minimize (chi-squared)
        
        Args:
            params: [fwhm_0, epsilon, gamma_ratio]
            energy: Energy array
            measured_counts: Measured spectrum
            element_data: Element line data
            
        Returns:
            Chi-squared value
        """
        try:
            calculated = self._calculate_spectrum(energy, element_data, params)
            
            # Normalize both to same total intensity for comparison
            # This removes absolute intensity differences and focuses on peak shapes
            measured_norm = measured_counts / np.sum(measured_counts)
            calculated_norm = calculated / np.sum(calculated)
            
            # For plotting, scale calculated to match measured peak height
            if np.max(calculated) > 0:
                scale_factor = np.max(measured_counts) / np.max(calculated)
                calculated_scaled = calculated * scale_factor
            else:
                calculated_scaled = calculated
            
            # Chi-squared (weighted by counts for Poisson statistics)
            # Only use regions where there's signal (avoid dividing by zero in background)
            mask = measured_counts > 10  # Only fit where we have signal
            
            if np.sum(mask) == 0:
                print("Warning: No signal points found!")
                return 1e10  # Return large penalty
            
            # Use normalized spectra for comparison (focuses on shapes, not absolute intensities)
            error = measured_norm[mask] - calculated_norm[mask]
            chi_squared = np.sum(error**2)
            
            # Add penalty for unrealistic parameters (regularization)
            # Prefer FWHM around 50-150 eV and epsilon around 1-3 eV
            fwhm_0, epsilon = params
            fwhm_penalty = ((fwhm_0 - 0.080)**2) / 0.01  # Prefer ~80 eV
            epsilon_penalty = ((epsilon - 0.002)**2) / 0.0001  # Prefer ~2 eV
            
            total_cost = chi_squared + 0.1 * (fwhm_penalty + epsilon_penalty)
            
            # Check for invalid values
            if not np.isfinite(total_cost):
                print(f"Warning: cost is not finite: {total_cost}")
                return 1e10
            
            return total_cost
        except Exception as e:
            print(f"Error in objective function: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_calibration(self, result: CalibrationResult, file_path: str):
        """Save calibration results to file"""
        import json
        
        data = {
            'fwhm_0': result.fwhm_0,
            'epsilon': result.epsilon,
            'voigt_gamma_ratio': result.voigt_gamma_ratio,
            'chi_squared': result.chi_squared,
            'r_squared': result.r_squared,
            'success': result.success,
            'message': result.message
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration saved to {file_path}")
    
    def load_calibration(self, file_path: str) -> CalibrationResult:
        """Load calibration results from file"""
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return CalibrationResult(
            fwhm_0=data['fwhm_0'],
            epsilon=data['epsilon'],
            voigt_gamma_ratio=data['voigt_gamma_ratio'],
            efficiency_params={},
            chi_squared=data['chi_squared'],
            r_squared=data['r_squared'],
            success=data['success'],
            message=data['message']
        )
    
    @staticmethod
    def load_reference_concentrations(csv_path: str) -> Dict[str, float]:
        """
        Load reference concentrations from NIST CSV file
        
        Args:
            csv_path: Path to CSV file with columns: Element, Symbol, Concentration_mg_kg
            
        Returns:
            Dict of {element_symbol: concentration_ppm}
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # Extract concentrations in ppm (mg/kg)
        concentrations = {}
        for _, row in df.iterrows():
            symbol = row['Symbol']
            conc = row['Concentration_mg_kg']
            
            # Only include elements with valid concentrations
            if pd.notna(conc) and conc > 0:
                concentrations[symbol] = float(conc)
        
        return concentrations
