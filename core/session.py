"""
Non-Qt analysis session / document model.

Holds the working spectrum, element selection, fit result, concentrations,
and instrument state so the UI can stay thin and batch/CLI can share state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.instrument_state import InstrumentState


@dataclass
class AnalysisSession:
    """In-memory analysis document for the current workspace."""

    spectrum: Any = None  # Optional Spectrum
    elements: List[Dict[str, Any]] = field(default_factory=list)
    fit_result: Any = None  # Optional FitResult
    concentrations: Dict[str, Any] = field(default_factory=dict)
    instrument: InstrumentState = field(default_factory=InstrumentState)
    spectrum_path: Optional[str] = None
    quantification_method: str = "semi_quant_area"

    def clear_fit(self) -> None:
        self.fit_result = None
        self.concentrations = {}

    def set_spectrum(self, spectrum, path: Optional[str] = None) -> None:
        self.spectrum = spectrum
        self.spectrum_path = path
        self.clear_fit()

    def set_elements(self, elements: List[Dict[str, Any]]) -> None:
        self.elements = list(elements or [])

    def set_fit_result(self, fit_result) -> None:
        self.fit_result = fit_result
        self.concentrations = {}

    def set_concentrations(self, concentrations: Dict[str, Any], method: str = None) -> None:
        self.concentrations = dict(concentrations or {})
        if method:
            self.quantification_method = method

    def apply_instrument_to_fitter(self, fitter) -> None:
        """Push current instrument calibrations onto a fitter."""
        self.instrument.apply_to_fitter(fitter)
        peak_fitter = getattr(fitter, "peak_fitter", None)
        if peak_fitter is not None and hasattr(peak_fitter, "activate"):
            peak_fitter.activate()
