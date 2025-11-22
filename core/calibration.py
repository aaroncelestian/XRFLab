"""
Instrument calibration using reference standards
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass
from typing import Dict, List, Tuple
from core.peak_fitting import PeakFitter
from core.xray_data import get_element_lines


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
                  initial_params: Dict = None) -> CalibrationResult:
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
        p0 = [
            initial_params['fwhm_0'],
            initial_params['epsilon'],
            initial_params['voigt_gamma_ratio']
        ]
        
        # Parameter bounds
        bounds = (
            [0.020, 0.0005, 0.05],  # Lower bounds
            [0.200, 0.0100, 0.50]   # Upper bounds
        )
        
        # Prepare element data
        try:
            element_data = self._prepare_element_data(
                reference_concentrations, 
                excitation_energy
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
            result = optimize.minimize(
                self._objective_function,
                p0,
                args=(energy, counts, element_data),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'disp': True}
            )
            
            # Extract optimized parameters
            fwhm_0_opt = result.x[0]
            epsilon_opt = result.x[1]
            gamma_ratio_opt = result.x[2]
            
            # Calculate final fit quality
            calculated_spectrum = self._calculate_spectrum(
                energy, element_data, result.x
            )
            
            residuals = counts - calculated_spectrum
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
    
    def _prepare_element_data(self, 
                              concentrations: Dict[str, float],
                              excitation_energy: float) -> List[Dict]:
        """
        Prepare element emission line data with relative intensities
        
        Args:
            concentrations: Dict of {element: concentration_ppm}
            excitation_energy: Excitation energy (keV)
            
        Returns:
            List of dicts with element, line, energy, relative_intensity
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
        fwhm_0, epsilon, gamma_ratio = params
        
        spectrum = np.zeros_like(energy)
        
        for line_data in element_data:
            line_energy = line_data['energy']
            intensity = line_data['relative_intensity']
            
            # Calculate FWHM at this energy
            fwhm = np.sqrt(fwhm_0**2 + 2.35 * epsilon * line_energy)
            sigma = fwhm / 2.355
            gamma = sigma * gamma_ratio
            
            # Add Voigt peak
            spectrum += self.peak_fitter.voigt(
                energy, intensity, line_energy, sigma, gamma
            )
        
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
        calculated = self._calculate_spectrum(energy, element_data, params)
        
        # Chi-squared (weighted by counts for Poisson statistics)
        weights = 1.0 / np.maximum(measured_counts, 1.0)
        chi_squared = np.sum(weights * (measured_counts - calculated)**2)
        
        return chi_squared
    
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
