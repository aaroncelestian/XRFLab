"""
Centralized batch spectral fitting and quantification

This module handles bulk processing of multiple XRF spectra with consistent
fitting parameters and quantification methods.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from core.fitting import SpectrumFitter
from core.spectrum import Spectrum
from core.calibration import CalibrationResult
from utils.io_handler import IOHandler


@dataclass
class BatchFitResult:
    """Results from fitting a single spectrum in batch mode"""
    spectrum_name: str
    spectrum_path: str
    fit_success: bool
    chi_squared: float
    r_squared: float
    elements_found: List[str]
    concentrations: Dict[str, float]  # {element: concentration}
    concentration_errors: Dict[str, float]  # {element: error}
    peak_areas: Dict[str, Dict[str, float]]  # {element: {line: area}}
    fitted_spectrum: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    measured_counts: Optional[np.ndarray] = None
    element_contributions: Optional[Dict[str, np.ndarray]] = None  # {element: counts_array}
    fit_time: float = 0.0
    error_message: str = ""


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    # Element selection
    elements: List[str] = field(default_factory=list)
    
    # Experimental parameters
    excitation_energy: float = 20.0  # keV
    tube_current: float = 1.0  # mA
    live_time: float = 30.0  # seconds
    incident_angle: float = 45.0  # degrees
    takeoff_angle: float = 45.0  # degrees
    
    # Fitting parameters
    background_method: str = "snip"
    peak_shape: str = "voigt"
    include_escape_peaks: bool = True
    include_pileup: bool = False
    tube_element: str = "Rh"
    include_tube_lines: bool = True
    
    # Calibration
    use_calibration: bool = False
    calibration_result: Optional[CalibrationResult] = None
    
    # Processing options
    save_individual_fits: bool = True
    save_plots: bool = False
    output_directory: Optional[Path] = None


class BatchProcessor:
    """Handles batch processing of multiple XRF spectra"""
    
    def __init__(self, config: BatchProcessingConfig):
        """
        Initialize batch processor
        
        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.fitter = SpectrumFitter()
        self.io_handler = IOHandler()
        self.results: List[BatchFitResult] = []
        
    def process_directory(self, directory: Path, 
                         file_pattern: str = "*.txt",
                         progress_callback=None) -> List[BatchFitResult]:
        """
        Process all spectra in a directory
        
        Args:
            directory: Directory containing spectrum files
            file_pattern: Glob pattern for spectrum files
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            List of batch fit results
        """
        # Find all spectrum files
        spectrum_files = sorted(Path(directory).glob(file_pattern))
        
        if not spectrum_files:
            raise ValueError(f"No spectrum files found matching {file_pattern} in {directory}")
        
        total = len(spectrum_files)
        self.results = []
        
        for i, file_path in enumerate(spectrum_files):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing {file_path.name}...")
            
            try:
                result = self.process_single_spectrum(file_path)
                self.results.append(result)
            except Exception as e:
                # Create failed result
                result = BatchFitResult(
                    spectrum_name=file_path.stem,
                    spectrum_path=str(file_path),
                    fit_success=False,
                    chi_squared=float('inf'),
                    r_squared=0.0,
                    elements_found=[],
                    concentrations={},
                    concentration_errors={},
                    peak_areas={},
                    error_message=str(e)
                )
                self.results.append(result)
        
        return self.results
    
    def process_file_list(self, file_paths: List[Path],
                         progress_callback=None) -> List[BatchFitResult]:
        """
        Process a list of spectrum files
        
        Args:
            file_paths: List of paths to spectrum files
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            List of batch fit results
        """
        total = len(file_paths)
        self.results = []
        
        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing {file_path.name}...")
            
            try:
                result = self.process_single_spectrum(file_path)
                self.results.append(result)
            except Exception as e:
                result = BatchFitResult(
                    spectrum_name=file_path.stem,
                    spectrum_path=str(file_path),
                    fit_success=False,
                    chi_squared=float('inf'),
                    r_squared=0.0,
                    elements_found=[],
                    concentrations={},
                    concentration_errors={},
                    peak_areas={},
                    error_message=str(e)
                )
                self.results.append(result)
        
        return self.results
    
    def process_single_spectrum(self, file_path: Path) -> BatchFitResult:
        """
        Process a single spectrum file
        
        Args:
            file_path: Path to spectrum file
            
        Returns:
            Batch fit result
        """
        start_time = datetime.now()
        
        # Load spectrum
        spectrum = self.io_handler.load_spectrum(str(file_path))
        
        # Fit spectrum
        fit_result = self.fitter.fit_spectrum(
            spectrum=spectrum,
            elements=self.config.elements,
            background_method=self.config.background_method,
            peak_shape=self.config.peak_shape,
            include_escape_peaks=self.config.include_escape_peaks,
            include_pileup=self.config.include_pileup,
            tube_element=self.config.tube_element if self.config.include_tube_lines else None
        )
        
        # Calculate concentrations if calibration available
        concentrations = {}
        concentration_errors = {}
        
        if self.config.use_calibration and self.config.calibration_result:
            # Quantification logic here
            # For now, placeholder
            for element in self.config.elements:
                concentrations[element] = 0.0
                concentration_errors[element] = 0.0
        
        # Extract element contributions for visualization
        element_contributions = {}
        if hasattr(fit_result, 'element_spectra'):
            element_contributions = fit_result.element_spectra
        
        # Calculate fit time
        fit_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = BatchFitResult(
            spectrum_name=file_path.stem,
            spectrum_path=str(file_path),
            fit_success=fit_result.success if hasattr(fit_result, 'success') else True,
            chi_squared=fit_result.chi_squared if hasattr(fit_result, 'chi_squared') else 0.0,
            r_squared=fit_result.r_squared if hasattr(fit_result, 'r_squared') else 0.0,
            elements_found=self.config.elements,
            concentrations=concentrations,
            concentration_errors=concentration_errors,
            peak_areas={},  # Extract from fit_result
            fitted_spectrum=fit_result.fitted_spectrum if hasattr(fit_result, 'fitted_spectrum') else None,
            residuals=fit_result.residuals if hasattr(fit_result, 'residuals') else None,
            energy=spectrum.energy,
            measured_counts=spectrum.counts,
            element_contributions=element_contributions,
            fit_time=fit_time
        )
        
        return result
    
    def export_results(self, output_path: Path, format: str = "csv"):
        """
        Export batch results to file
        
        Args:
            output_path: Path to output file
            format: Export format ('csv', 'excel', 'json')
        """
        if format == "csv":
            self._export_csv(output_path)
        elif format == "excel":
            self._export_excel(output_path)
        elif format == "json":
            self._export_json(output_path)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def _export_csv(self, output_path: Path):
        """Export results to CSV"""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            # Determine all elements
            all_elements = set()
            for result in self.results:
                all_elements.update(result.concentrations.keys())
            all_elements = sorted(all_elements)
            
            # Write header
            fieldnames = ['Spectrum', 'Success', 'Chi²', 'R²', 'Fit Time (s)']
            for element in all_elements:
                fieldnames.append(f'{element} (wt%)')
                fieldnames.append(f'{element} Error')
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data
            for result in self.results:
                row = {
                    'Spectrum': result.spectrum_name,
                    'Success': result.fit_success,
                    'Chi²': f'{result.chi_squared:.4f}',
                    'R²': f'{result.r_squared:.4f}',
                    'Fit Time (s)': f'{result.fit_time:.2f}'
                }
                
                for element in all_elements:
                    conc = result.concentrations.get(element, 0.0)
                    error = result.concentration_errors.get(element, 0.0)
                    row[f'{element} (wt%)'] = f'{conc:.4f}'
                    row[f'{element} Error'] = f'{error:.4f}'
                
                writer.writerow(row)
    
    def _export_excel(self, output_path: Path):
        """Export results to Excel"""
        try:
            import pandas as pd
            
            # Create DataFrame
            data = []
            for result in self.results:
                row = {
                    'Spectrum': result.spectrum_name,
                    'Success': result.fit_success,
                    'Chi²': result.chi_squared,
                    'R²': result.r_squared,
                    'Fit Time (s)': result.fit_time
                }
                
                # Add concentrations
                for element, conc in result.concentrations.items():
                    row[f'{element} (wt%)'] = conc
                    row[f'{element} Error'] = result.concentration_errors.get(element, 0.0)
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
            
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel export")
    
    def _export_json(self, output_path: Path):
        """Export results to JSON"""
        import json
        
        data = []
        for result in self.results:
            data.append({
                'spectrum_name': result.spectrum_name,
                'spectrum_path': result.spectrum_path,
                'fit_success': result.fit_success,
                'chi_squared': result.chi_squared,
                'r_squared': result.r_squared,
                'elements_found': result.elements_found,
                'concentrations': result.concentrations,
                'concentration_errors': result.concentration_errors,
                'fit_time': result.fit_time,
                'error_message': result.error_message
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics for batch results
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.fit_success]
        failed = [r for r in self.results if not r.fit_success]
        
        stats = {
            'total_spectra': len(self.results),
            'successful_fits': len(successful),
            'failed_fits': len(failed),
            'success_rate': len(successful) / len(self.results) * 100,
            'average_chi_squared': np.mean([r.chi_squared for r in successful]) if successful else 0,
            'average_r_squared': np.mean([r.r_squared for r in successful]) if successful else 0,
            'average_fit_time': np.mean([r.fit_time for r in self.results]),
            'total_processing_time': sum([r.fit_time for r in self.results])
        }
        
        return stats
