"""
Peak Shape vs Energy Calibration

Uses pure element standards to calibrate detector resolution:
FWHM(E) = sqrt(FWHM_0^2 + 2.355^2 * epsilon * E)

Where:
- FWHM_0: Electronic noise contribution (eV)
- epsilon: Fano factor contribution (eV/keV)
- E: Photon energy (keV)

Standards used:
- Fe.txt: Fe Kα (6.40 keV), Fe Kβ (7.06 keV)
- Cu.txt: Cu Kα (8.05 keV), Cu Kβ (8.91 keV)
- Ti.txt: Ti Kα (4.51 keV), Ti Kβ (4.93 keV)
- Zn.txt: Zn Kα (8.64 keV), Zn Kβ (9.57 keV)
- Mg.txt: Mg Kα (1.25 keV)
- cubic zirconia.txt: Zr Lα (2.04 keV), Zr Kα (15.75 keV)

All except cubic zirconia also have Al Kα (1.49 keV) from sample holder
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import optimize
from scipy.signal import find_peaks
from typing import Dict, List, Tuple
from dataclasses import dataclass

from utils.spectrum_loader import load_spectrum
from core.background import BackgroundModeler
from core.peak_fitting import PeakFitter


@dataclass
class PeakMeasurement:
    """Measured peak properties"""
    element: str
    line: str
    energy: float  # keV
    fwhm: float    # keV
    intensity: float
    fit_quality: float  # R²


class PeakShapeCalibrator:
    """Calibrate detector resolution vs energy"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.bg_modeler = BackgroundModeler()
        self.peak_fitter = PeakFitter()
        self.measurements: List[PeakMeasurement] = []
        
        # Expected peak energies (keV) for each element
        self.expected_peaks = {
            'Fe': [
                ('Fe Kα1', 6.404),
                ('Fe Kα2', 6.391),
                ('Fe Kβ1', 7.058),
                ('Al Kα', 1.487)
            ],
            'Cu': [
                ('Cu Kα1', 8.048),
                ('Cu Kα2', 8.028),
                ('Cu Kβ1', 8.905),
                ('Al Kα', 1.487)
            ],
            'Ti': [
                ('Ti Kα1', 4.511),
                ('Ti Kα2', 4.505),
                ('Ti Kβ1', 4.932),
                ('Al Kα', 1.487)
            ],
            'Zn': [
                ('Zn Kα1', 8.639),
                ('Zn Kα2', 8.616),
                ('Zn Kβ1', 9.572),
                ('Al Kα', 1.487)
            ],
            'Mg': [
                ('Mg Kα', 1.254),
                ('Al Kα', 1.487)
            ],
            'cubic zirconia': [
                # Skip Zr L lines - too much overlap and matrix effects
                # ('Zr Lα1', 2.042),
                # ('Zr Lβ1', 2.124),
                ('Zr Kα1', 15.775),
                ('Zr Kβ1', 17.668)  # Keep Kβ - good for high energy calibration
            ]
        }
    
    def load_and_process_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load spectrum and subtract background"""
        filepath = self.data_dir / filename
        
        # Load spectrum
        energy, counts = load_spectrum(str(filepath))
        
        # Subtract background
        background = self.bg_modeler.estimate_background(
            energy, counts, method='snip', window_length=50
        )
        counts_bg_sub = counts - background
        
        return energy, counts, counts_bg_sub
    
    def measure_peak_width(self, 
                          energy: np.ndarray, 
                          counts: np.ndarray,
                          peak_energy: float,
                          element: str,
                          line: str,
                          window_width: float = 0.3,
                          min_counts: float = 100) -> PeakMeasurement:
        """
        Measure FWHM of a peak by Gaussian fitting
        
        Args:
            energy: Energy array (keV)
            counts: Background-subtracted counts
            peak_energy: Expected peak energy (keV)
            element: Element name
            line: Line name
            window_width: Energy window around peak (keV)
        
        Returns:
            PeakMeasurement with fitted FWHM
        """
        # Extract region around peak
        mask = np.abs(energy - peak_energy) < window_width
        e_region = energy[mask]
        c_region = counts[mask]
        
        if len(e_region) < 10:
            raise ValueError(f"Not enough points around {peak_energy:.3f} keV")
        
        # Find peak maximum
        max_idx = np.argmax(c_region)
        peak_pos = e_region[max_idx]
        peak_height = c_region[max_idx]
        
        if peak_height < min_counts:  # Minimum signal threshold
            raise ValueError(f"Peak too weak at {peak_energy:.3f} keV (counts={peak_height:.0f}, need>{min_counts})")
        
        # Initial guess for Gaussian fit
        # p = [amplitude, center, sigma]
        # Use reasonable initial FWHM estimate
        initial_fwhm = 0.150  # keV, typical for SDD
        initial_sigma = initial_fwhm / 2.355
        p0 = [peak_height, peak_pos, initial_sigma]
        
        # Fit Gaussian with reasonable bounds
        try:
            def gaussian_model(x, amp, mu, sigma):
                return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)
            
            # Reasonable bounds for SDD detector
            min_fwhm = 0.080  # 80 eV minimum (excellent SDD)
            max_fwhm = 0.300  # 300 eV maximum (allows some flexibility)
            
            popt, pcov = optimize.curve_fit(
                gaussian_model, e_region, c_region, p0=p0,
                bounds=([peak_height*0.3, peak_energy-0.1, min_fwhm/2.355], 
                       [peak_height*2.0, peak_energy+0.1, max_fwhm/2.355]),
                maxfev=5000
            )
            
            amp_fit, mu_fit, sigma_fit = popt
            fwhm_fit = sigma_fit * 2.355
            
            # Calculate R²
            c_fit = gaussian_model(e_region, *popt)
            ss_res = np.sum((c_region - c_fit)**2)
            ss_tot = np.sum((c_region - np.mean(c_region))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return PeakMeasurement(
                element=element,
                line=line,
                energy=mu_fit,
                fwhm=fwhm_fit,
                intensity=amp_fit,
                fit_quality=r_squared
            )
            
        except Exception as e:
            raise ValueError(f"Fit failed for {line} at {peak_energy:.3f} keV: {e}")
    
    def process_all_files(self):
        """Process all standard files and measure peak widths"""
        print("Processing XRF standards for peak shape calibration...")
        print("=" * 70)
        
        for filename, expected in self.expected_peaks.items():
            print(f"\n{filename}:")
            
            # Load data
            try:
                energy, counts_raw, counts_bg_sub = self.load_and_process_file(f"{filename}.txt")
            except Exception as e:
                print(f"  ❌ Failed to load: {e}")
                continue
            
            # Measure each expected peak
            for line_name, peak_energy in expected:
                try:
                    # Use stricter thresholds for high-energy peaks
                    min_counts = 150 if peak_energy > 10 else 80
                    
                    measurement = self.measure_peak_width(
                        energy, counts_bg_sub, peak_energy, filename, line_name,
                        min_counts=min_counts
                    )
                    
                    # Only accept good fits with reasonable FWHM
                    # FWHM should be 80-300 eV for typical SDD (allow some flexibility)
                    fwhm_ev = measurement.fwhm * 1000
                    if measurement.fit_quality > 0.85 and 80 < fwhm_ev < 300:
                        self.measurements.append(measurement)
                        print(f"  ✓ {line_name:12s} @ {measurement.energy:.3f} keV: "
                              f"FWHM = {fwhm_ev:.1f} eV (R² = {measurement.fit_quality:.3f})")
                    else:
                        reason = "Poor fit" if measurement.fit_quality <= 0.85 else f"Unrealistic FWHM ({fwhm_ev:.1f} eV)"
                        print(f"  ⚠ {line_name:12s} @ {peak_energy:.3f} keV: "
                              f"{reason} (R² = {measurement.fit_quality:.3f})")
                        
                except Exception as e:
                    print(f"  ✗ {line_name:12s} @ {peak_energy:.3f} keV: {e}")
        
        print(f"\n{'=' * 70}")
        print(f"Total successful measurements: {len(self.measurements)}")
    
    def _remove_outliers(self, energies: np.ndarray, fwhms: np.ndarray, 
                        threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using iterative fitting with residual threshold
        
        Args:
            energies: Energy array
            fwhms: FWHM array
            threshold: Number of standard deviations for outlier detection
        
        Returns:
            Filtered (energies, fwhms) arrays
        """
        def resolution_model(E, fwhm_0, epsilon):
            return np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * E)
        
        # Initial fit
        p0 = [0.100, 0.001]
        popt, _ = optimize.curve_fit(
            resolution_model, energies, fwhms, p0=p0,
            bounds=([0.050, 0.0001], [0.200, 0.01])
        )
        
        # Calculate residuals
        fwhm_predicted = resolution_model(energies, *popt)
        residuals = fwhms - fwhm_predicted
        std_residual = np.std(residuals)
        
        # Find outliers
        outlier_mask = np.abs(residuals) > threshold * std_residual
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0:
            print(f"  Found {n_outliers} outlier(s):")
            for i, is_outlier in enumerate(outlier_mask):
                if is_outlier:
                    elem = self.measurements[i].element
                    line = self.measurements[i].line
                    resid_ev = residuals[i] * 1000
                    print(f"    - {elem} {line} @ {energies[i]:.2f} keV: "
                          f"residual = {resid_ev:+.1f} eV ({abs(resid_ev/std_residual/1000):.1f}σ)")
            
            # Remove outliers
            energies = energies[~outlier_mask]
            fwhms = fwhms[~outlier_mask]
            
            # Update measurements list
            self.measurements = [m for i, m in enumerate(self.measurements) if not outlier_mask[i]]
            
            print(f"  Removed {n_outliers} outlier(s), {len(energies)} measurements remaining")
        else:
            print("  No outliers detected")
        
        return energies, fwhms
    
    def fit_resolution_model(self, remove_outliers: bool = True, model: str = 'detector') -> Dict[str, float]:
        """
        Fit detector resolution model with various functional forms
        
        Args:
            remove_outliers: If True, remove outliers using iterative fitting
            model: Model type - 'detector', 'linear', 'quadratic', 'exponential', 'power'
        
        Returns:
            Dict with fit parameters and statistics
        
        Models:
            - 'detector': FWHM(E) = sqrt(FWHM_0^2 + 2.355^2 * epsilon * E)  [Standard detector model]
            - 'linear': FWHM(E) = a + b*E
            - 'quadratic': FWHM(E) = a + b*E + c*E^2
            - 'exponential': FWHM(E) = a * exp(b*E)
            - 'power': FWHM(E) = a * E^b
        """
        if len(self.measurements) < 3:
            raise ValueError("Need at least 3 peak measurements for calibration")
        
        # Extract energies and FWHMs
        energies = np.array([m.energy for m in self.measurements])
        fwhms = np.array([m.fwhm for m in self.measurements])
        
        # Remove outliers if requested (use detector model for outlier detection)
        if remove_outliers and len(energies) > 5:
            print("\nChecking for outliers...")
            energies, fwhms = self._remove_outliers(energies, fwhms)
        
        # Select model and fit
        if model == 'detector':
            # Standard detector model: FWHM(E) = sqrt(FWHM_0^2 + 2.355^2 * epsilon * E)
            def fit_func(E, fwhm_0, epsilon):
                return np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * E)
            p0 = [0.100, 0.001]
            bounds = ([0.050, 0.0001], [0.200, 0.01])
            param_names = ['fwhm_0', 'epsilon']
            
        elif model == 'linear':
            # Linear model: FWHM(E) = a + b*E
            def fit_func(E, a, b):
                return a + b * E
            p0 = [0.100, 0.005]
            bounds = ([0.050, 0.0], [0.200, 0.02])
            param_names = ['intercept', 'slope']
            
        elif model == 'quadratic':
            # Quadratic model: FWHM(E) = a + b*E + c*E^2
            def fit_func(E, a, b, c):
                return a + b * E + c * E**2
            p0 = [0.100, 0.005, 0.0001]
            bounds = ([0.050, -0.01, -0.001], [0.200, 0.02, 0.001])
            param_names = ['intercept', 'linear_coef', 'quadratic_coef']
            
        elif model == 'exponential':
            # Exponential model: FWHM(E) = a * exp(b*E)
            def fit_func(E, a, b):
                return a * np.exp(b * E)
            p0 = [0.100, 0.02]
            bounds = ([0.050, 0.0], [0.200, 0.1])
            param_names = ['amplitude', 'exponent']
            
        elif model == 'power':
            # Power law model: FWHM(E) = a * E^b
            def fit_func(E, a, b):
                return a * E**b
            p0 = [0.100, 0.3]
            bounds = ([0.050, 0.0], [0.200, 1.0])
            param_names = ['amplitude', 'power']
            
        else:
            raise ValueError(f"Unknown model: {model}. Choose from: detector, linear, quadratic, exponential, power")
        
        # Fit
        try:
            popt, pcov = optimize.curve_fit(
                fit_func, energies, fwhms, p0=p0, bounds=bounds
            )
            
            # Calculate fit quality
            fwhm_predicted = fit_func(energies, *popt)
            residuals = fwhms - fwhm_predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((fwhms - np.mean(fwhms))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Standard errors
            perr = np.sqrt(np.diag(pcov))
            
            # Build results dict with model-specific parameters
            results = {
                'model': model,
                'r_squared': r_squared,
                'rmse': np.sqrt(np.mean(residuals**2)),
                'aic': len(energies) * np.log(ss_res / len(energies)) + 2 * len(popt),  # Akaike Information Criterion
                'bic': len(energies) * np.log(ss_res / len(energies)) + len(popt) * np.log(len(energies))  # Bayesian Information Criterion
            }
            
            # Add model-specific parameters
            for i, (name, value, error) in enumerate(zip(param_names, popt, perr)):
                results[name] = value
                results[f'{name}_err'] = error
            
            # For backward compatibility, add fwhm_0 and epsilon for detector model
            if model == 'detector':
                results['fwhm_0'] = popt[0]
                results['fwhm_0_err'] = perr[0]
                results['epsilon'] = popt[1]
                results['epsilon_err'] = perr[1]
            
            return results
            
        except Exception as e:
            raise ValueError(f"Resolution model fit failed: {e}")
    
    def plot_calibration(self, results: Dict[str, float], save_path: str = None):
        """Plot FWHM vs Energy with fitted model"""
        
        # Extract data
        energies = np.array([m.energy for m in self.measurements])
        fwhms = np.array([m.fwhm * 1000 for m in self.measurements])  # Convert to eV
        elements = [m.element for m in self.measurements]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot 1: FWHM vs Energy
        ax1.scatter(energies, fwhms, s=100, alpha=0.6, c='blue', edgecolors='black')
        
        # Add labels for each point
        for i, (e, f, elem) in enumerate(zip(energies, fwhms, elements)):
            ax1.annotate(elem, (e, f), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        # Plot fitted model based on model type
        e_model = np.linspace(min(energies), max(energies), 200)
        model_type = results.get('model', 'detector')
        
        if model_type == 'detector':
            fwhm_0 = results['fwhm_0']
            epsilon = results['epsilon']
            fwhm_model = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * e_model) * 1000
            fwhm_predicted = np.sqrt(fwhm_0**2 + 2.355**2 * epsilon * energies) * 1000
            label = f"FWHM(E) = √(FWHM₀² + 2.355² ε E)"
            
        elif model_type == 'linear':
            a, b = results['intercept'], results['slope']
            fwhm_model = (a + b * e_model) * 1000
            fwhm_predicted = (a + b * energies) * 1000
            label = f"FWHM(E) = {a*1000:.1f} + {b*1000:.2f}·E"
            
        elif model_type == 'quadratic':
            a, b, c = results['intercept'], results['linear_coef'], results['quadratic_coef']
            fwhm_model = (a + b * e_model + c * e_model**2) * 1000
            fwhm_predicted = (a + b * energies + c * energies**2) * 1000
            label = f"FWHM(E) = {a*1000:.1f} + {b*1000:.2f}·E + {c*1000:.3f}·E²"
            
        elif model_type == 'exponential':
            a, b = results['amplitude'], results['exponent']
            fwhm_model = a * np.exp(b * e_model) * 1000
            fwhm_predicted = a * np.exp(b * energies) * 1000
            label = f"FWHM(E) = {a*1000:.1f}·exp({b:.3f}·E)"
            
        elif model_type == 'power':
            a, b = results['amplitude'], results['power']
            fwhm_model = a * e_model**b * 1000
            fwhm_predicted = a * energies**b * 1000
            label = f"FWHM(E) = {a*1000:.1f}·E^{b:.3f}"
        
        ax1.plot(e_model, fwhm_model, 'r-', linewidth=2, label=label)
        
        ax1.set_xlabel('Energy (keV)', fontsize=12)
        ax1.set_ylabel('FWHM (eV)', fontsize=12)
        ax1.set_title(f'Detector Resolution Calibration ({model_type.title()} Model)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Add text box with results
        if model_type == 'detector':
            textstr = '\n'.join([
                f"FWHM₀ = {results['fwhm_0']*1000:.1f} ± {results['fwhm_0_err']*1000:.1f} eV",
                f"ε = {results['epsilon']*1000:.2f} ± {results['epsilon_err']*1000:.2f} eV/keV",
                f"R² = {results['r_squared']:.4f}",
                f"RMSE = {results['rmse']*1000:.1f} eV",
                f"AIC = {results['aic']:.1f}",
                f"BIC = {results['bic']:.1f}"
            ])
        else:
            # Generic parameter display
            param_lines = []
            for key, value in results.items():
                if key.endswith('_err') or key in ['model', 'r_squared', 'rmse', 'aic', 'bic']:
                    continue
                if key in results and f'{key}_err' in results:
                    param_lines.append(f"{key} = {value:.4f} ± {results[f'{key}_err']:.4f}")
            
            textstr = '\n'.join(param_lines + [
                f"R² = {results['r_squared']:.4f}",
                f"RMSE = {results['rmse']*1000:.1f} eV",
                f"AIC = {results['aic']:.1f}",
                f"BIC = {results['bic']:.1f}"
            ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Plot 2: Residuals
        residuals = fwhms - fwhm_predicted
        
        ax2.scatter(energies, residuals, s=100, alpha=0.6, c='blue', edgecolors='black')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Energy (keV)', fontsize=12)
        ax2.set_ylabel('Residual (eV)', fontsize=12)
        ax2.set_title('Fit Residuals', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nCalibration plot saved to: {save_path}")
        
        plt.show()
    
    def save_calibration(self, results: Dict[str, float], filepath: str):
        """Save calibration results to JSON file"""
        import json
        from datetime import datetime
        
        model_type = results.get('model', 'detector')
        
        # Base output structure
        output = {
            'calibration_date': datetime.now().isoformat(),
            'detector_model': 'XGT7200 SDD',
            'model_type': model_type,
            'r_squared': results['r_squared'],
            'rmse_eV': results['rmse'] * 1000,
            'aic': results.get('aic', 0.0),
            'bic': results.get('bic', 0.0),
            'n_peaks': len(self.measurements),
            'measurements': [
                {
                    'element': m.element,
                    'line': m.line,
                    'energy_keV': m.energy,
                    'fwhm_eV': m.fwhm * 1000,
                    'fit_quality': m.fit_quality
                }
                for m in self.measurements
            ]
        }
        
        # Add model-specific parameters
        if model_type == 'detector':
            output.update({
                'fwhm_0_keV': results['fwhm_0'],
                'fwhm_0_eV': results['fwhm_0'] * 1000,
                'fwhm_0_error_eV': results['fwhm_0_err'] * 1000,
                'epsilon_keV': results['epsilon'],
                'epsilon_eV_per_keV': results['epsilon'] * 1000,
                'epsilon_error_eV_per_keV': results['epsilon_err'] * 1000,
            })
        else:
            # For other models, save all parameters generically
            parameters = {}
            for key, value in results.items():
                if key.endswith('_err') or key in ['model', 'r_squared', 'rmse', 'aic', 'bic']:
                    continue
                parameters[key] = value
                if f'{key}_err' in results:
                    parameters[f'{key}_err'] = results[f'{key}_err']
            
            output['parameters'] = parameters
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Calibration data saved to: {filepath}")


def main():
    """Run peak shape calibration"""
    
    # Set paths
    data_dir = Path("sample_data/data")
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create calibrator
    calibrator = PeakShapeCalibrator(data_dir)
    
    # Process all files
    calibrator.process_all_files()
    
    if len(calibrator.measurements) < 3:
        print("\n❌ Not enough successful measurements for calibration!")
        return
    
    # Fit resolution model
    print("\nFitting detector resolution model...")
    print("=" * 70)
    try:
        results = calibrator.fit_resolution_model()
        
        print(f"\n✓ Calibration successful!")
        print(f"  FWHM₀ = {results['fwhm_0']*1000:.1f} ± {results['fwhm_0_err']*1000:.1f} eV")
        print(f"  ε = {results['epsilon']*1000:.2f} ± {results['epsilon_err']*1000:.2f} eV/keV")
        print(f"  R² = {results['r_squared']:.4f}")
        print(f"  RMSE = {results['rmse']*1000:.1f} eV")
        
        # Example predictions
        print(f"\nExample FWHM predictions:")
        for E in [1.5, 5.0, 10.0, 15.0]:
            fwhm = np.sqrt(results['fwhm_0']**2 + 2.355**2 * results['epsilon'] * E)
            print(f"  {E:5.1f} keV → {fwhm*1000:6.1f} eV")
        
        # Save results
        calibrator.save_calibration(results, output_dir / "peak_shape_calibration.json")
        
        # Plot
        calibrator.plot_calibration(results, output_dir / "peak_shape_calibration.png")
        
    except Exception as e:
        print(f"\n❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
