"""
Injectable detector / instrument state for peak fitting.

Peak fitting helpers historically read class-level PeakFitter attributes.
InstrumentState owns the real configuration; PeakFitter.activate() syncs
class-level helpers so Analysis and Batch can each carry their own copy
without silently sharing mutable globals across sessions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DetectorModel:
    """Energy-dependent detector resolution model."""

    fwhm_0: float = 0.050  # keV electronic noise term
    epsilon: float = 0.0015  # Fano * w term
    voigt_gamma_ratio: float = 0.15
    use_calibrated_shapes: bool = False
    fwhm_calibration: Any = None  # Optional FWHMCalibration

    def predict_fwhm(self, energy: float) -> float:
        """Return FWHM (keV) at the given energy."""
        if self.fwhm_calibration is not None:
            return float(self.fwhm_calibration.predict_fwhm(energy))
        e = float(energy)
        return float(np.sqrt(self.fwhm_0**2 + (2.355**2) * self.epsilon * e))

    def apply_fwhm_calibration(self, calibration) -> None:
        """Attach an FWHMCalibration and lock shapes when present."""
        self.fwhm_calibration = calibration
        if calibration is None:
            self.use_calibrated_shapes = False
            return

        if getattr(calibration, "model_type", None) == "detector":
            params = getattr(calibration, "parameters", {}) or {}
            if "fwhm_0" in params:
                self.fwhm_0 = float(params["fwhm_0"])
            if "epsilon" in params:
                self.epsilon = float(params["epsilon"])

        self.use_calibrated_shapes = True

    def copy(self) -> "DetectorModel":
        return DetectorModel(
            fwhm_0=self.fwhm_0,
            epsilon=self.epsilon,
            voigt_gamma_ratio=self.voigt_gamma_ratio,
            use_calibrated_shapes=self.use_calibrated_shapes,
            fwhm_calibration=self.fwhm_calibration,
        )


@dataclass
class InstrumentState:
    """Full instrument context used by fitting and quantification."""

    detector: DetectorModel = field(default_factory=DetectorModel)
    tube_profile_library: Any = None  # Optional TubeProfileLibrary
    standards_calibration: Any = None  # Optional CalibrationResult

    def apply_fwhm_calibration(self, calibration) -> None:
        self.detector.apply_fwhm_calibration(calibration)

    def copy(self) -> "InstrumentState":
        return InstrumentState(
            detector=self.detector.copy(),
            tube_profile_library=self.tube_profile_library,
            standards_calibration=self.standards_calibration,
        )

    def apply_to_fitter(self, fitter) -> None:
        """
        Push this instrument state onto a SpectrumFitter / PeakFitter.

        Args:
            fitter: SpectrumFitter (preferred) or PeakFitter instance
        """
        peak_fitter = getattr(fitter, "peak_fitter", fitter)
        if hasattr(peak_fitter, "set_detector"):
            peak_fitter.set_detector(self.detector.copy())
        if hasattr(fitter, "set_tube_profile_library"):
            fitter.set_tube_profile_library(self.tube_profile_library)
        elif hasattr(fitter, "tube_profile_library"):
            fitter.tube_profile_library = self.tube_profile_library
