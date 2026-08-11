"""
Centralized batch spectral fitting and quantification

This module handles bulk processing of multiple XRF spectra with consistent
fitting parameters and semi-quantitative (area-normalized) intensities.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from core.fitting import SpectrumFitter
from core.instrument_state import InstrumentState
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
    concentrations: Dict[str, float]  # {element: relative intensity %}
    concentration_errors: Dict[str, float]  # empty for semi-quant
    peak_areas: Dict[str, Dict[str, float]]  # {element: {line: area}}
    fitted_spectrum: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    measured_counts: Optional[np.ndarray] = None
    element_contributions: Optional[Dict[str, np.ndarray]] = None
    fit_time: float = 0.0
    error_message: str = ""
    quantification_method: str = "semi_quant_area"


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    # Element selection — list of symbols or dicts with symbol/z
    elements: List[Any] = field(default_factory=list)

    # Experimental parameters
    excitation_energy: float = 20.0  # keV
    tube_current: float = 1.0  # mA
    live_time: float = 30.0  # seconds
    incident_angle: float = 45.0  # degrees
    takeoff_angle: float = 45.0  # degrees

    # Fitting parameters (aligned with SpectrumFitter.fit_spectrum)
    background_method: str = "snip"
    peak_shape: str = "voigt"
    include_escape_peaks: bool = True
    include_pileup: bool = False
    tube_element: str = "Rh"
    include_tube_lines: bool = True
    include_compton: bool = True
    auto_find_peaks: bool = True
    excitation_kv: float = 50.0

    # Instrument state (detector FWHM, tube profiles)
    instrument_state: Optional[InstrumentState] = None

    # Legacy calibration fields (reserved; batch uses semi-quant areas)
    use_calibration: bool = False
    calibration_result: Optional[CalibrationResult] = None

    # Processing options
    save_individual_fits: bool = True
    save_plots: bool = False
    output_directory: Optional[Path] = None


class BatchProcessor:
    """Handles batch processing of multiple XRF spectra"""

    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.fitter = SpectrumFitter()
        if config.instrument_state is not None:
            self.fitter.apply_instrument_state(config.instrument_state)
        self.io_handler = IOHandler()
        self.results: List[BatchFitResult] = []

    def _normalize_elements(self) -> List[Dict[str, Any]]:
        """Convert config.elements to [{symbol, z}, ...] for SpectrumFitter."""
        from core.advanced_peak_fitting import get_element_z

        normalized = []
        for item in self.config.elements or []:
            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("element")
                z = item.get("z") or (get_element_z(symbol) if symbol else None)
                if symbol and z:
                    normalized.append({"symbol": symbol, "z": int(z)})
            elif isinstance(item, str) and item.strip():
                symbol = item.strip()
                z = get_element_z(symbol)
                if z:
                    normalized.append({"symbol": symbol, "z": int(z)})
        return normalized

    def process_directory(
        self,
        directory: Path,
        file_pattern: str = "*.txt",
        progress_callback=None,
    ) -> List[BatchFitResult]:
        spectrum_files = sorted(Path(directory).glob(file_pattern))

        if not spectrum_files:
            raise ValueError(
                f"No spectrum files found matching {file_pattern} in {directory}"
            )

        return self.process_file_list(spectrum_files, progress_callback=progress_callback)

    def process_file_list(
        self, file_paths: List[Path], progress_callback=None
    ) -> List[BatchFitResult]:
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
                    spectrum_name=Path(file_path).stem,
                    spectrum_path=str(file_path),
                    fit_success=False,
                    chi_squared=float("inf"),
                    r_squared=0.0,
                    elements_found=[],
                    concentrations={},
                    concentration_errors={},
                    peak_areas={},
                    error_message=str(e),
                )
                self.results.append(result)

        return self.results

    def process_single_spectrum(self, file_path: Path) -> BatchFitResult:
        start_time = datetime.now()
        file_path = Path(file_path)

        if self.config.instrument_state is not None:
            self.fitter.apply_instrument_state(self.config.instrument_state)
        else:
            self.fitter.peak_fitter.activate()

        spectrum = self.io_handler.load_spectrum(str(file_path))
        elements = self._normalize_elements()

        excitation_kv = self.config.excitation_kv
        if self.config.excitation_energy and self.config.excitation_energy > 0:
            # Prefer explicit kV when set; else use excitation_energy as kV proxy
            if self.config.excitation_kv == 50.0 and self.config.excitation_energy != 20.0:
                excitation_kv = float(self.config.excitation_energy)

        fit_result = self.fitter.fit_spectrum(
            energy=spectrum.energy,
            counts=spectrum.counts,
            elements=elements,
            background_method=self.config.background_method,
            peak_shape=self.config.peak_shape,
            auto_find_peaks=self.config.auto_find_peaks,
            tube_element=self.config.tube_element if self.config.include_tube_lines else None,
            excitation_kv=excitation_kv,
            include_tube_lines=self.config.include_tube_lines,
            include_compton=self.config.include_compton,
        )

        stats = fit_result.statistics or {}
        chi_squared = float(stats.get("chi_squared", 0.0))
        r_squared = float(stats.get("r_squared", 0.0))

        # Semi-quantitative relative intensities (same as Analysis tab)
        quant = self.fitter.quantify_elements(fit_result.peaks, None)
        concentrations = {
            el: float(data.get("relative_intensity_pct", data.get("concentration", 0.0)))
            for el, data in quant.items()
        }
        concentration_errors: Dict[str, float] = {}

        peak_areas: Dict[str, Dict[str, float]] = {}
        for peak in fit_result.peaks:
            if peak.is_tube_line or not peak.element:
                continue
            peak_areas.setdefault(peak.element, {})
            line = peak.line or "unknown"
            peak_areas[peak.element][line] = float(peak.area)

        elements_found = sorted(concentrations.keys())
        fit_time = (datetime.now() - start_time).total_seconds()

        return BatchFitResult(
            spectrum_name=file_path.stem,
            spectrum_path=str(file_path),
            fit_success=True,
            chi_squared=chi_squared,
            r_squared=r_squared,
            elements_found=elements_found,
            concentrations=concentrations,
            concentration_errors=concentration_errors,
            peak_areas=peak_areas,
            fitted_spectrum=fit_result.fitted_spectrum,
            residuals=fit_result.residuals,
            energy=spectrum.energy,
            measured_counts=spectrum.counts,
            element_contributions=None,
            fit_time=fit_time,
            quantification_method="semi_quant_area",
        )

    def export_results(self, output_path: Path, format: str = "csv"):
        if format == "csv":
            self._export_csv(output_path)
        elif format == "excel":
            self._export_excel(output_path)
        elif format == "json":
            self._export_json(output_path)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def _export_csv(self, output_path: Path):
        import csv

        with open(output_path, "w", newline="") as f:
            all_elements = set()
            for result in self.results:
                all_elements.update(result.concentrations.keys())
            all_elements = sorted(all_elements)

            fieldnames = [
                "Spectrum",
                "Success",
                "Chi²",
                "R²",
                "Fit Time (s)",
                "Method",
            ]
            for element in all_elements:
                fieldnames.append(f"{element} (rel %)")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "Spectrum": result.spectrum_name,
                    "Success": result.fit_success,
                    "Chi²": f"{result.chi_squared:.4f}",
                    "R²": f"{result.r_squared:.4f}",
                    "Fit Time (s)": f"{result.fit_time:.2f}",
                    "Method": result.quantification_method,
                }
                for element in all_elements:
                    conc = result.concentrations.get(element, 0.0)
                    row[f"{element} (rel %)"] = f"{conc:.4f}"
                writer.writerow(row)

    def _export_excel(self, output_path: Path):
        try:
            import pandas as pd

            data = []
            for result in self.results:
                row = {
                    "Spectrum": result.spectrum_name,
                    "Success": result.fit_success,
                    "Chi²": result.chi_squared,
                    "R²": result.r_squared,
                    "Fit Time (s)": result.fit_time,
                    "Method": result.quantification_method,
                }
                for element, conc in result.concentrations.items():
                    row[f"{element} (rel %)"] = conc
                data.append(row)

            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)

        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel export")

    def _export_json(self, output_path: Path):
        import json

        data = []
        for result in self.results:
            data.append(
                {
                    "spectrum_name": result.spectrum_name,
                    "spectrum_path": result.spectrum_path,
                    "fit_success": result.fit_success,
                    "chi_squared": result.chi_squared,
                    "r_squared": result.r_squared,
                    "elements_found": result.elements_found,
                    "concentrations": result.concentrations,
                    "concentration_errors": result.concentration_errors,
                    "quantification_method": result.quantification_method,
                    "fit_time": result.fit_time,
                    "error_message": result.error_message,
                }
            )

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_summary_statistics(self) -> Dict:
        if not self.results:
            return {}

        successful = [r for r in self.results if r.fit_success]
        failed = [r for r in self.results if not r.fit_success]

        return {
            "total_spectra": len(self.results),
            "successful_fits": len(successful),
            "failed_fits": len(failed),
            "success_rate": len(successful) / len(self.results) * 100,
            "average_chi_squared": np.mean([r.chi_squared for r in successful])
            if successful
            else 0,
            "average_r_squared": np.mean([r.r_squared for r in successful])
            if successful
            else 0,
            "average_fit_time": np.mean([r.fit_time for r in self.results]),
            "total_processing_time": sum([r.fit_time for r in self.results]),
            "quantification_method": "semi_quant_area",
        }
