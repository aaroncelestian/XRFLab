"""
Instrument calibration using reference standards
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import json
from datetime import datetime
from core.background import BackgroundModeler
from core.peak_fitting import PeakFitter, Peak
from core.xray_data import get_element_lines, get_tube_lines
from core.fundamental_parameters import FundamentalParameters
from core.fisx_integration import FisxCalculator, convert_fisx_to_element_data
from core.fwhm_calibration import FWHMCalibration, load_fwhm_calibration, get_fwhm_initial_params


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
    fwhm_model_type: str = 'detector'  # FWHM model type
    fwhm_calibration: Dict = None  # Full FWHM calibration data
    calibration_date: str = None  # ISO format timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.calibration_date is None:
            data['calibration_date'] = datetime.now().isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationResult':
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, filepath: str):
        """Save calibration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CalibrationResult':
        """Load calibration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class InstrumentCalibrator:
    """Calibrate instrument parameters using reference standards"""
    
    def __init__(self, fwhm_calibration: FWHMCalibration = None):
        """
        Initialize calibrator
        
        Args:
            fwhm_calibration: Pre-calibrated FWHM model (optional)
                             If provided, FWHM_0 and epsilon will be fixed during optimization
        """
        self.peak_fitter = PeakFitter()
        self.fwhm_calibration = fwhm_calibration
    
    def calibrate(self, 
                  energy: np.ndarray,
                  counts: np.ndarray,
                  reference_concentrations: Dict[str, float],
                  excitation_energy: float = 50.0,
                  initial_params: Dict = None,
                  use_measured_intensities: bool = True,
                  experimental_params: Dict = None,
                  bg_params: Dict = None) -> CalibrationResult:
        """
        Calibrate instrument parameters using reference spectrum
        
        OPTIMIZATION STRATEGY:
        1. Pre-calculate all emission line intensities using fisx (ONCE)
        2. During optimization, only calculate peak shapes (Gaussians)
        3. Optimize detector parameters: FWHM_0, epsilon, intensity_scale, rh_scatter
        4. Future: Add sum peaks, escape peaks, detector efficiency curve
        
        Args:
            energy: Energy array (keV)
            counts: Measured counts
            reference_concentrations: Dict of element concentrations
            excitation_energy: Tube voltage (keV)
            use_measured_intensities: Use measured peak intensities vs fisx calculation
            experimental_params: Dict with tube_element, current, live_time, etc.
            initial_params: Initial parameter guesses
            bg_params: Background parameters (optional)
            
        Returns:
            CalibrationResult with optimized parameters
        """
        # Set initial parameter guesses
        if initial_params is None:
            # Use FWHM calibration if available
            if self.fwhm_calibration is not None:
                fwhm_params = get_fwhm_initial_params(self.fwhm_calibration)
                initial_params = {
                    'fwhm_0': fwhm_params['fwhm_0'],
                    'epsilon': fwhm_params['epsilon'],
                    'voigt_gamma_ratio': 0.15,
                }
            else:
                initial_params = {
                    'fwhm_0': 0.050,  # keV
                    'epsilon': 0.0015,  # keV
                    'voigt_gamma_ratio': 0.15,  # gamma/sigma
                }
        
        # Convert to parameter array for optimization
        # Parameters: FWHM_0, epsilon, intensity_scale, rh_scatter_scale
        # Background is pre-subtracted, so no background parameters needed
        
        # If FWHM calibration is provided, use those values and narrow bounds
        if self.fwhm_calibration is not None:
            fwhm_0_cal = initial_params['fwhm_0']
            epsilon_cal = initial_params['epsilon']
            
            p0 = [
                fwhm_0_cal,  # Use calibrated FWHM_0
                epsilon_cal,  # Use calibrated epsilon
                1000.0,       # Overall intensity scaling factor
                0.01          # Rh tube scatter scaling (start small)
            ]
            
            # Narrow bounds around calibrated values (±20%)
            bounds = [
                (fwhm_0_cal * 0.8, fwhm_0_cal * 1.2),
                (epsilon_cal * 0.8, epsilon_cal * 1.2),
                (10.0, 100000.0),
                (0.0, 0.5)
            ]
        else:
            # Wide bounds for uncalibrated case
            p0 = [
                0.110,      # FWHM_0: Start at 110 eV (middle of range)
                0.00001,    # epsilon: Start at 0.00001 (Fano factor ~0.12 for Si)
                1000.0,     # Overall intensity scaling factor
                0.01        # Rh tube scatter scaling (start small)
            ]
            
            bounds = [
                (0.070, 0.150),     # FWHM_0: 70-150 eV (realistic SDD range)
                (0.000001, 0.0001), # EPSILON: Fano-like factor (much smaller with 2.355² multiplier)
                (10.0, 100000.0),   # Intensity scale: much wider range
                (0.0, 0.5)          # Rh scatter scale: 0-50% (allow stronger Rh lines)
            ]
        
        # Prepare element data
        try:
            if use_measured_intensities:
                # Extract intensities from measured spectrum
                # This is faster and works well for calibration
                element_data = self._prepare_element_data_from_spectrum(
                    energy, counts, reference_concentrations, excitation_energy, initial_params
                )
            else:
                # Use fisx fundamental parameters to calculate intensities
                # This is slower but more physically accurate
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
        print(f"  Pre-calculated intensities - will NOT recalculate during optimization")
        print(f"  Initial FWHM_0: {p0[0]:.4f} keV")
        print(f"  Initial EPSILON: {p0[1]:.6f} keV")
        
        # Optimize parameters
        try:
            # Test objective function first
            print("Testing objective function...")
            test_chi2 = self._objective_function(p0, energy, counts, element_data)
            print(f"  Initial χ²: {test_chi2:.2f}")
            
            # Callback to show progress
            self.iteration_count = 0
            def callback(xk):
                self.iteration_count += 1
                chi2 = self._objective_function(xk, energy, counts, element_data, experimental_params)
                if self.iteration_count % 5 == 0:  # Print every 5 iterations
                    print(f"  Iteration {self.iteration_count}: FWHM={xk[0]:.4f} keV, ε={xk[1]:.6f} keV, scale={xk[2]:.1f}, Rh={xk[3]:.4f}, χ²={chi2:.2f}")
            
            print("Starting optimization...")
            try:
                result = optimize.minimize(
                    self._objective_function,
                    p0,
                    args=(energy, counts, element_data, experimental_params),
                    method='L-BFGS-B',
                    bounds=bounds,
                    callback=callback,
                    options={'maxiter': 500, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
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
            if len(result.x) != 4:
                raise ValueError(f"Expected 4 optimized parameters, got {len(result.x)}: {result.x}")
            
            fwhm_0_opt = result.x[0]
            epsilon_opt = result.x[1]
            intensity_scale_opt = result.x[2]
            rh_scatter_scale_opt = result.x[3]
            gamma_ratio_opt = 0.0  # Not used for Gaussian peaks
            
            # Calculate final fit quality
            calculated_spectrum = self._calculate_spectrum(
                energy, element_data, result.x, experimental_params
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
            print(f"  Optimized FWHM_0: {fwhm_0_opt:.4f} keV ({fwhm_0_opt*1000:.1f} eV)")
            print(f"  Optimized EPSILON: {epsilon_opt:.6f} keV")
            print(f"  Optimized intensity scale: {intensity_scale_opt:.1f}")
            print(f"  Optimized Rh scatter scale: {rh_scatter_scale_opt:.4f}")
            print(f"  R²: {r_squared:.4f}")
            print(f"  χ²: {chi_squared:.2f}")
            
            return CalibrationResult(
                fwhm_0=fwhm_0_opt,
                epsilon=epsilon_opt,
                voigt_gamma_ratio=gamma_ratio_opt,
                efficiency_params={
                    'intensity_scale': intensity_scale_opt,
                    'rh_scatter_scale': rh_scatter_scale_opt
                },
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
    
    def _add_tube_scatter_lines(self, tube_element: str, excitation_energy: float) -> List[Dict]:
        """
        Add tube scatter lines (Rh K, L lines from X-ray tube)
        
        These are scattered tube photons that reach the detector.
        They should be included in calculated spectrum but marked as tube lines.
        
        Args:
            tube_element: Tube anode element (e.g., 'Rh', 'W', 'Mo')
            excitation_energy: Tube voltage (keV)
            
        Returns:
            List of tube scatter line dicts
        """
        from core.xray_data import get_element_lines
        
        tube_z_map = {'Rh': 45, 'W': 74, 'Mo': 42, 'Ag': 47, 'Cu': 29, 'Cr': 24}
        z = tube_z_map.get(tube_element, 45)
        
        tube_lines = []
        try:
            lines_data = get_element_lines(tube_element, z)
            
            # Estimate scatter intensity as ~0.1-0.5% of fluorescence signal
            # This is a rough approximation - actual scatter depends on:
            # - Tube-sample-detector geometry
            # - Sample matrix (more scatter from light elements)
            # - Tube filters
            # Typical range: 0.001-0.005 (0.1-0.5%)
            scatter_fraction = 0.002  # 0.2% is more realistic for filtered tubes
            
            # Add K lines if below excitation energy
            for line in lines_data.get('K', []):
                if line['energy'] < excitation_energy:
                    tube_lines.append({
                        'element': f'{tube_element}_tube',  # Mark as tube line
                        'line': line['name'],
                        'energy': line['energy'],
                        'relative_intensity': scatter_fraction * line.get('relative_intensity', 1.0),
                        'is_tube_scatter': True
                    })
            
            # Add L lines
            for line in lines_data.get('L', []):
                if line['energy'] < excitation_energy:
                    tube_lines.append({
                        'element': f'{tube_element}_tube',
                        'line': line['name'],
                        'energy': line['energy'],
                        'relative_intensity': scatter_fraction * line.get('relative_intensity', 1.0),
                        'is_tube_scatter': True
                    })
            
            if tube_lines:
                print(f"  Added {len(tube_lines)} {tube_element} tube scatter lines")
                
        except Exception as e:
            print(f"  Warning: Could not add tube scatter lines: {e}")
        
        return tube_lines
    
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
            tube_element = 'Rh'  # Default Rhodium tube
            if experimental_params:
                incident_angle = experimental_params.get('incident_angle', 45.0)
                takeoff_angle = experimental_params.get('takeoff_angle', 45.0)
                tube_element = experimental_params.get('tube_element', 'Rh')
            
            fisx_calc = FisxCalculator(
                excitation_energy=excitation_energy,
                tube_element=tube_element,
                incident_angle=incident_angle,
                takeoff_angle=takeoff_angle
            )
            intensities = fisx_calc.calculate_intensities(composition, thickness=0.1)  # Infinite thickness
            
            # Convert fisx results to element_data format
            element_data = convert_fisx_to_element_data(intensities, excitation_energy)
            
            # TODO: Add tube scatter lines (currently disabled - intensities too high)
            # tube_scatter_lines = self._add_tube_scatter_lines(tube_element, excitation_energy)
            # element_data.extend(tube_scatter_lines)
            
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
                           params: np.ndarray,
                           experimental_params: Dict = None) -> np.ndarray:
        """
        Calculate synthetic spectrum with given parameters
        
        Args:
            energy: Energy array
            element_data: List of element line data
            params: [fwhm_0, epsilon, intensity_scale, rh_scatter_scale, bg_const, bg_exp]
            experimental_params: Experimental parameters (for tube element)
            
        Returns:
            Calculated spectrum
        """
        try:
            fwhm_0, epsilon, intensity_scale, rh_scatter_scale = params
        except ValueError as e:
            print(f"Error unpacking params: {params}")
            print(f"  Expected 4 values, got {len(params)}")
            raise
        
        # Start with zero background (background is pre-subtracted)
        spectrum = np.zeros_like(energy)
        
        # Add fluorescence lines
        for line_data in element_data:
            try:
                line_energy = line_data['energy']
                
                # Apply energy-dependent efficiency correction
                # Simple model: efficiency drops at mid-energies (3-5 keV)
                # This accounts for detector window absorption, dead layer, etc.
                # Reduced from 0.5 to 0.2 (was too aggressive)
                eff_correction = 1.0 - 0.2 * np.exp(-((line_energy - 4.0)**2) / 2.0)
                
                intensity = line_data['relative_intensity'] * intensity_scale * eff_correction
                
                # Calculate FWHM at this energy: FWHM² = FWHM₀² + 2.355² · ε · E
                # Standard detector resolution formula with Fano statistics
                # But use energy-scaled FWHM_0 for better low-energy resolution
                fwhm_0_scaled = fwhm_0 * np.sqrt(line_energy / 6.0)  # Scale with sqrt(E/6keV)
                fwhm_0_scaled = max(fwhm_0_scaled, 0.070)  # Minimum 70 eV
                fwhm = np.sqrt(fwhm_0_scaled**2 + 2.355**2 * epsilon * line_energy)
                sigma = fwhm / 2.355
                
                # Use Voigt profile (Gaussian ⊗ Lorentzian)
                # Voigt naturally has extended tails and asymmetry
                # gamma controls the Lorentzian contribution (tail strength)
                gamma_ratio = 0.05  # gamma/sigma ratio (5% Lorentzian character)
                gamma = gamma_ratio * sigma
                
                voigt_peak = self.peak_fitter.voigt(
                    energy, intensity, line_energy, sigma, gamma
                )
                spectrum += voigt_peak
            except Exception as e:
                print(f"Error calculating peak for {line_data}: {e}")
                raise
        
        # Add Compton scatter from Rh tube
        # Compton scatter is inelastic - photon loses energy
        # E_compton = E₀ / (1 + E₀/511 * (1 - cos(θ)))
        # For Rh Kα (20.2 keV) at 90°: E_compton ≈ 18.8 keV
        if rh_scatter_scale > 0 and experimental_params:
            rh_ka_energy = 20.216  # keV
            scatter_angle = 90.0  # degrees (typical geometry)
            cos_theta = np.cos(np.radians(scatter_angle))
            
            # Compton formula
            compton_energy = rh_ka_energy / (1 + rh_ka_energy / 511.0 * (1 - cos_theta))
            
            # Compton peak is broader and weaker than elastic scatter
            compton_intensity = rh_scatter_scale * 0.5  # 50% of elastic scatter (increased)
            fwhm_compton = 0.250  # Compton peaks are broader (~250 eV)
            sigma_compton = fwhm_compton / 2.355
            
            spectrum += self.peak_fitter.gaussian(
                energy, compton_intensity, compton_energy, sigma_compton
            )
        
        # Add Rh tube scatter lines if rh_scatter_scale > 0
        if rh_scatter_scale > 0 and experimental_params:
            tube_element = experimental_params.get('tube_element', 'Rh')
            excitation_energy = experimental_params.get('excitation_energy', 50.0)
            
            # Get Rh lines dynamically
            rh_lines = self._get_tube_scatter_lines(tube_element, excitation_energy, rh_scatter_scale)
            for line_data in rh_lines:
                line_energy = line_data['energy']
                # Rh scatter lines do NOT get efficiency correction
                # They are scattered from the tube, not fluorescence from sample
                intensity = line_data['relative_intensity']
                
                # Use energy-scaled FWHM for Rh lines too
                fwhm_0_scaled = fwhm_0 * np.sqrt(line_energy / 6.0)
                fwhm_0_scaled = max(fwhm_0_scaled, 0.070)
                fwhm = np.sqrt(fwhm_0_scaled**2 + 2.355**2 * epsilon * line_energy)
                sigma = fwhm / 2.355
                
                spectrum += self.peak_fitter.gaussian(
                    energy, intensity, line_energy, sigma
                )
        
        return spectrum
    
    def _get_tube_scatter_lines(self, tube_element: str, excitation_energy: float, scatter_scale: float) -> List[Dict]:
        """Get tube scatter lines with given scaling"""
        from core.xray_data import get_element_lines
        
        tube_z_map = {'Rh': 45, 'W': 74, 'Mo': 42, 'Ag': 47, 'Cu': 29, 'Cr': 24}
        z = tube_z_map.get(tube_element, 45)
        
        tube_lines = []
        try:
            lines_data = get_element_lines(tube_element, z)
            
            # Add K lines if below excitation energy
            for line in lines_data.get('K', []):
                if line['energy'] < excitation_energy:
                    tube_lines.append({
                        'element': f'{tube_element}_tube',
                        'line': line['name'],
                        'energy': line['energy'],
                        'relative_intensity': scatter_scale * line.get('relative_intensity', 1.0)
                    })
            
            # Add L lines
            for line in lines_data.get('L', []):
                if line['energy'] < excitation_energy:
                    tube_lines.append({
                        'element': f'{tube_element}_tube',
                        'line': line['name'],
                        'energy': line['energy'],
                        'relative_intensity': scatter_scale * line.get('relative_intensity', 1.0)
                    })
        except Exception as e:
            pass  # Silently skip if can't get tube lines
        
        return tube_lines
    
    def _objective_function(self,
                           params: np.ndarray,
                           energy: np.ndarray,
                           measured_counts: np.ndarray,
                           element_data: List[Dict],
                           experimental_params: Dict = None) -> float:
        """
        Objective function to minimize (chi-squared)
        
        Args:
            params: [fwhm_0, epsilon, intensity_scale, rh_scatter_scale]
            energy: Energy array
            measured_counts: Measured spectrum
            element_data: Element line data
            experimental_params: Experimental parameters
            
        Returns:
            Chi-squared value
        """
        try:
            # Downsample for speed during optimization (every 4th point)
            downsample = 4
            energy_ds = energy[::downsample]
            measured_ds = measured_counts[::downsample]
            
            # Calculate spectrum with Rh lines added dynamically
            calculated = self._calculate_spectrum(energy_ds, element_data, params, experimental_params)
            
            if np.max(calculated) == 0:
                return 1e10  # No calculated signal
            
            # Chi-squared with Poisson weighting
            # Only use regions where there's signal
            mask = measured_ds > 10  # Only fit where we have signal
            
            if np.sum(mask) == 0:
                print("Warning: No signal points found!")
                return 1e10  # Return large penalty
            
            # Weighted residuals (Poisson statistics: weight = 1/sqrt(counts))
            # No additional scaling - intensity_scale parameter handles this
            residuals = measured_ds[mask] - calculated[mask]
            weights = 1.0 / np.sqrt(measured_ds[mask])  # Poisson weighting
            weighted_residuals = residuals * weights
            chi_squared = np.sum(weighted_residuals**2) / np.sum(mask)
            
            # No regularization - let the data drive the fit
            # The optimizer will find the best FWHM based on the measured peaks
            total_cost = chi_squared
            
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
            'message': result.message,
            'fwhm_model_type': result.fwhm_model_type,
            'fwhm_calibration': result.fwhm_calibration
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
            message=data['message'],
            fwhm_model_type=data.get('fwhm_model_type', 'detector'),
            fwhm_calibration=data.get('fwhm_calibration', None)
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
