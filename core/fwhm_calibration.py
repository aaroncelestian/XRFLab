"""
FWHM vs Energy Calibration Module

This module provides detector resolution calibration using pure element standards.
The calibrated FWHM model is used throughout the application for accurate peak fitting
and quantitative analysis.

Integration with InstrumentCalibrator:
- Provides initial FWHM_0 and epsilon values
- Can be used standalone or as part of full instrument calibration
- Supports multiple model types (detector, linear, quadratic, etc.)
"""

import numpy as np
import json
from pathlib import Path
from scipy import optimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class FWHMCalibration:
    """
    FWHM calibration results
    
    This stores the detector resolution model that describes how FWHM varies with energy.
    Used throughout the application for peak fitting and quantification.
    """
    model_type: str  # 'detector', 'linear', 'quadratic', 'exponential', 'power'
    parameters: Dict[str, float]  # Model-specific parameters
    parameter_errors: Dict[str, float]  # Uncertainties
    r_squared: float  # Fit quality
    rmse: float  # Root mean square error (keV)
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    n_peaks: int  # Number of peaks used
    energy_range: Tuple[float, float]  # Min, max energy (keV)
    calibration_date: str  # ISO format timestamp
    
    def predict_fwhm(self, energy: float) -> float:
        """
        Predict FWHM at given energy using calibrated model
        
        Args:
            energy: Photon energy in keV
            
        Returns:
            FWHM in keV
        """
        if self.model_type == 'detector':
            fwhm_0 = self.parameters['fwhm_0']
            epsilon = self.parameters['epsilon']
            return np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * energy)
            
        elif self.model_type == 'linear':
            a = self.parameters['intercept']
            b = self.parameters['slope']
            return a + b * energy
            
        elif self.model_type == 'quadratic':
            a = self.parameters['intercept']
            b = self.parameters['linear_coef']
            c = self.parameters['quadratic_coef']
            return a + b * energy + c * energy**2
            
        elif self.model_type == 'exponential':
            a = self.parameters['amplitude']
            b = self.parameters['exponent']
            return a * np.exp(b * energy)
            
        elif self.model_type == 'power':
            a = self.parameters['amplitude']
            b = self.parameters['power']
            return a * energy**b
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict_fwhm_array(self, energies: np.ndarray) -> np.ndarray:
        """Vectorized FWHM prediction"""
        return np.array([self.predict_fwhm(e) for e in energies])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FWHMCalibration':
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, filepath: str):
        """Save calibration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FWHMCalibration':
        """Load calibration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self):
        if self.model_type == 'detector':
            fwhm_0_ev = self.parameters['fwhm_0'] * 1000
            epsilon_ev = self.parameters['epsilon'] * 1000
            return (f"FWHMCalibration(model=detector, "
                   f"FWHM₀={fwhm_0_ev:.1f}eV, ε={epsilon_ev:.2f}eV/keV, "
                   f"R²={self.r_squared:.4f})")
        else:
            return (f"FWHMCalibration(model={self.model_type}, "
                   f"R²={self.r_squared:.4f}, RMSE={self.rmse*1000:.1f}eV)")


def convert_peak_shape_calibration(peak_shape_results: Dict) -> FWHMCalibration:
    """
    Convert results from calibrate_peak_shape.py to FWHMCalibration
    
    Args:
        peak_shape_results: Dict from PeakShapeCalibrator.fit_resolution_model()
        
    Returns:
        FWHMCalibration object
    """
    from datetime import datetime
    
    model_type = peak_shape_results.get('model', 'detector')
    
    # Extract parameters (model-specific)
    parameters = {}
    parameter_errors = {}
    
    for key, value in peak_shape_results.items():
        if key.endswith('_err'):
            param_name = key[:-4]  # Remove '_err'
            parameter_errors[param_name] = value
        elif key not in ['model', 'r_squared', 'rmse', 'aic', 'bic']:
            parameters[key] = value
    
    return FWHMCalibration(
        model_type=model_type,
        parameters=parameters,
        parameter_errors=parameter_errors,
        r_squared=peak_shape_results['r_squared'],
        rmse=peak_shape_results['rmse'],
        aic=peak_shape_results['aic'],
        bic=peak_shape_results['bic'],
        n_peaks=peak_shape_results.get('n_peaks', 0),
        energy_range=peak_shape_results.get('energy_range', (0.0, 20.0)),
        calibration_date=datetime.now().isoformat()
    )


def load_fwhm_calibration(filepath: str) -> FWHMCalibration:
    """
    Load FWHM calibration from file
    
    Supports both:
    - New format (FWHMCalibration JSON)
    - Legacy format (peak_shape_calibration.json from calibrate_peak_shape.py)
    
    Args:
        filepath: Path to calibration file
        
    Returns:
        FWHMCalibration object
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if it's new format (has model_type field)
    if 'model_type' in data:
        return FWHMCalibration.from_dict(data)
    
    # Legacy format - convert
    from datetime import datetime
    
    # Determine model type from available fields
    if 'fwhm_0_keV' in data or 'fwhm_0_eV' in data:
        model_type = 'detector'
        
        # Handle both keV and eV formats
        if 'fwhm_0_keV' in data:
            fwhm_0 = data['fwhm_0_keV']
            epsilon = data.get('epsilon_keV', data.get('epsilon_eV_per_keV', 0.003) / 1000)
        else:
            fwhm_0 = data['fwhm_0_eV'] / 1000
            epsilon = data.get('epsilon_eV_per_keV', 3.5) / 1000
        
        parameters = {
            'fwhm_0': fwhm_0,
            'epsilon': epsilon
        }
        
        parameter_errors = {
            'fwhm_0': data.get('fwhm_0_error_eV', 0.0) / 1000,
            'epsilon': data.get('epsilon_error_eV_per_keV', 0.0) / 1000
        }
        
    else:
        raise ValueError("Unknown calibration file format")
    
    return FWHMCalibration(
        model_type=model_type,
        parameters=parameters,
        parameter_errors=parameter_errors,
        r_squared=data.get('r_squared', 0.0),
        rmse=data.get('rmse_eV', 0.0) / 1000,
        aic=data.get('aic', 0.0),
        bic=data.get('bic', 0.0),
        n_peaks=data.get('n_peaks', 0),
        energy_range=(0.0, 20.0),
        calibration_date=data.get('calibration_date', datetime.now().isoformat())
    )


def create_default_fwhm_calibration() -> FWHMCalibration:
    """
    Create default FWHM calibration for typical SDD
    
    Returns:
        FWHMCalibration with typical values for modern SDD
    """
    from datetime import datetime
    
    return FWHMCalibration(
        model_type='detector',
        parameters={
            'fwhm_0': 0.120,  # 120 eV
            'epsilon': 0.0035  # 3.5 eV/keV
        },
        parameter_errors={
            'fwhm_0': 0.005,
            'epsilon': 0.0002
        },
        r_squared=0.0,  # Not fitted
        rmse=0.0,
        aic=0.0,
        bic=0.0,
        n_peaks=0,
        energy_range=(0.0, 20.0),
        calibration_date=datetime.now().isoformat()
    )


# Convenience functions for integration with InstrumentCalibrator

def get_fwhm_initial_params(calibration: FWHMCalibration) -> Dict[str, float]:
    """
    Extract initial parameters for InstrumentCalibrator
    
    Args:
        calibration: FWHMCalibration object
        
    Returns:
        Dict with 'fwhm_0' and 'epsilon' for detector model
    """
    if calibration.model_type == 'detector':
        return {
            'fwhm_0': calibration.parameters['fwhm_0'],
            'epsilon': calibration.parameters['epsilon']
        }
    else:
        # For non-detector models, estimate equivalent detector parameters
        # by evaluating at a reference energy (e.g., 6 keV)
        ref_energy = 6.0
        fwhm_at_ref = calibration.predict_fwhm(ref_energy)
        
        # Rough approximation: FWHM ≈ sqrt(FWHM_0^2 + 2.355^2 * epsilon * E)
        # At 6 keV with typical epsilon=0.0035: FWHM ≈ sqrt(FWHM_0^2 + 0.33)
        # Solve for FWHM_0 assuming typical epsilon
        epsilon = 0.0035
        fwhm_0 = np.sqrt(max(0, fwhm_at_ref**2 - 2.355**2 * epsilon * ref_energy))
        
        return {
            'fwhm_0': fwhm_0,
            'epsilon': epsilon
        }


def apply_fwhm_calibration_to_peak_fitter(calibration: FWHMCalibration, peak_fitter):
    """
    Apply FWHM calibration to PeakFitter instance
    
    Args:
        calibration: FWHMCalibration object
        peak_fitter: PeakFitter instance
    """
    # Store calibration in peak fitter
    peak_fitter.fwhm_calibration = calibration
    
    # Update default FWHM calculation method
    def calculate_fwhm(energy):
        return calibration.predict_fwhm(energy)
    
    peak_fitter.calculate_fwhm = calculate_fwhm
