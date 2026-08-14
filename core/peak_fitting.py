"""
Peak fitting for XRF spectra
"""

import numpy as np
from scipy import signal, optimize
from scipy.special import wofz
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars/containers so Peak.to_dict() is JSON-serializable."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class Peak:
    """Represents a fitted peak"""
    energy: float  # keV
    amplitude: float
    fwhm: float  # keV
    area: float
    element: str = None
    line: str = None
    shape: str = 'gaussian'  # Peak shape used for fitting
    shape_params: dict = None  # Parameters specific to the shape (sigma, gamma, etc.)
    is_tube_line: bool = False  # True if this is from X-ray tube, not sample
    fixed_fwhm: float = None  # If set, width was locked (e.g. Compton)
    
    def __post_init__(self):
        if self.shape_params is None:
            self.shape_params = {}

    def to_dict(self) -> dict:
        """JSON-friendly representation for project files."""
        return {
            "energy": float(self.energy),
            "amplitude": float(self.amplitude),
            "fwhm": float(self.fwhm),
            "area": float(self.area),
            "element": self.element,
            "line": self.line,
            "shape": self.shape,
            "shape_params": _json_safe(self.shape_params or {}),
            "is_tube_line": bool(self.is_tube_line),
            "fixed_fwhm": (
                None if self.fixed_fwhm is None else float(self.fixed_fwhm)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Peak":
        return cls(
            energy=float(data["energy"]),
            amplitude=float(data.get("amplitude", 0.0)),
            fwhm=float(data.get("fwhm", 0.0)),
            area=float(data.get("area", 0.0)),
            element=data.get("element"),
            line=data.get("line"),
            shape=str(data.get("shape") or "gaussian"),
            shape_params=dict(data.get("shape_params") or {}),
            is_tube_line=bool(data.get("is_tube_line", False)),
            fixed_fwhm=(
                None
                if data.get("fixed_fwhm") is None
                else float(data["fixed_fwhm"])
            ),
        )
    
    def __str__(self):
        tube_marker = " [TUBE]" if self.is_tube_line else ""
        if self.fixed_fwhm is not None:
            tube_marker = (tube_marker + " [wide]").replace(" [TUBE] [wide]", " [TUBE, wide]")
        if self.element and self.line:
            return f"{self.element}-{self.line}: {self.energy:.3f} keV (Area: {self.area:.1f}){tube_marker}"
        return f"Peak at {self.energy:.3f} keV (Area: {self.area:.1f}){tube_marker}"


class PeakFitter:
    """Peak fitting for XRF spectra"""
    
    # Detector parameters for energy-dependent FWHM
    # FWHM(E) = sqrt(FWHM_0^2 + 2.355^2 * epsilon * E)  (standard detector model)
    # Prefer FWHMCalibration via set_fwhm_calibration() when available.
    # Class attrs remain for tube_constraints helpers; prefer instance DetectorModel
    # via set_detector() / activate() for session-scoped state.
    FWHM_0 = 0.050  # keV at 0 keV (noise contribution)
    EPSILON = 0.0015  # Fano factor * w (eV per e-h pair)
    VOIGT_GAMMA_RATIO = 0.15  # gamma/sigma ratio for Voigt peaks
    USE_CALIBRATED_SHAPES = False  # If True, fix peak shapes during fitting
    # Max allowed center shift during LS (keV). Weak peaks otherwise wander
    # within the old ±0.2 keV window onto neighbors / noise.
    CENTER_SHIFT_FRACTION = 0.25  # fraction of local FWHM
    CENTER_SHIFT_MIN_KEV = 0.010  # 10 eV floor
    CENTER_SHIFT_MAX_KEV = 0.040  # 40 eV cap
    # Known (element/tube) line centers stay even tighter
    KNOWN_LINE_CENTER_SHIFT_KEV = 0.020  # 20 eV
    # Reject electronic zero/noise. C Kα is ~0.277 keV (some instruments);
    # Na Kα is ~1.04 keV (typical EDXRF). 0.17 keV sits below C so both work.
    MIN_PEAK_ENERGY_KEV = 0.17
    _fwhm_calibration = None  # Optional FWHMCalibration shared by all fits
    _active = None  # Currently activated PeakFitter instance
    
    def __init__(self, detector=None):
        from core.instrument_state import DetectorModel
        self.detector = detector if detector is not None else DetectorModel()
        self.fwhm_calibration = self.detector.fwhm_calibration
        self._sync_instance_from_detector()

    def _sync_instance_from_detector(self):
        """Mirror DetectorModel fields onto this instance."""
        d = self.detector
        self.fwhm_calibration = d.fwhm_calibration

    def set_detector(self, detector):
        """Install an injectable DetectorModel and sync class helpers if active."""
        from core.instrument_state import DetectorModel
        self.detector = detector if detector is not None else DetectorModel()
        self._sync_instance_from_detector()
        if PeakFitter._active is self:
            self.activate()

    def activate(self):
        """
        Make this fitter's detector the active class-level configuration.

        Static helpers (tube_constraints, calculate_fwhm) read class attrs;
        activate() keeps them aligned with this instance's DetectorModel.
        """
        PeakFitter._active = self
        PeakFitter._sync_class_from_detector(self.detector)

    @classmethod
    def _sync_class_from_detector(cls, detector):
        cls.FWHM_0 = float(detector.fwhm_0)
        cls.EPSILON = float(detector.epsilon)
        cls.VOIGT_GAMMA_RATIO = float(detector.voigt_gamma_ratio)
        cls.USE_CALIBRATED_SHAPES = bool(detector.use_calibrated_shapes)
        cls._fwhm_calibration = detector.fwhm_calibration

    @classmethod
    def set_fwhm_calibration(cls, calibration):
        """
        Apply a detector FWHM calibration for all subsequent peak fits.
        
        Uses the calibrated model for width predictions and, when present,
        fixes peak shapes to those widths during fitting.
        """
        from core.instrument_state import DetectorModel

        if cls._active is not None:
            cls._active.detector.apply_fwhm_calibration(calibration)
            cls._active._sync_instance_from_detector()
            cls._sync_class_from_detector(cls._active.detector)
            return

        # No active instance: update class defaults and a fresh detector snapshot
        detector = DetectorModel()
        detector.apply_fwhm_calibration(calibration)
        cls._sync_class_from_detector(detector)
    
    @classmethod
    def calculate_fwhm(cls, energy):
        """Calculate energy-dependent FWHM for detector"""
        if cls._active is not None:
            return cls._active.detector.predict_fwhm(energy)
        if cls._fwhm_calibration is not None:
            return cls._fwhm_calibration.predict_fwhm(energy)
        return np.sqrt(cls.FWHM_0**2 + (2.355 ** 2) * cls.EPSILON * energy)
    
    @staticmethod
    def gaussian(x, amplitude, center, sigma):
        """Gaussian peak function"""
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    @staticmethod
    def lorentzian(x, amplitude, center, gamma):
        """Lorentzian peak function"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    @staticmethod
    def voigt(x, amplitude, center, sigma, gamma):
        """
        Voigt profile (convolution of Gaussian and Lorentzian)
        More accurate for X-ray peaks
        """
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def pseudo_voigt(x, amplitude, center, sigma, eta):
        """
        Pseudo-Voigt profile (linear combination of Gaussian and Lorentzian)
        Faster approximation of Voigt profile
        
        Args:
            eta: Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
        """
        gaussian = PeakFitter.gaussian(x, 1, center, sigma)
        lorentzian = PeakFitter.lorentzian(x, 1, center, sigma)
        return amplitude * (eta * lorentzian + (1 - eta) * gaussian)
    
    @staticmethod
    def hypermet(x, amplitude, center, sigma, tail_amplitude, tail_slope):
        """
        Hypermet function for XRF peaks with low-energy tail
        Combines Gaussian with exponential tail for incomplete charge collection
        
        Args:
            tail_amplitude: Relative amplitude of tail (0-1)
            tail_slope: Decay slope of tail (keV^-1)
        """
        # Main Gaussian peak
        gaussian = PeakFitter.gaussian(x, amplitude, center, sigma)
        
        # Low-energy exponential tail
        tail = np.zeros_like(x)
        mask = x < center
        if np.any(mask):
            tail[mask] = amplitude * tail_amplitude * np.exp(tail_slope * (x[mask] - center))
        
        return gaussian + tail
    
    @staticmethod
    def tail_gaussian(x, amplitude, center, sigma, tail_fraction, tail_sigma):
        """
        Gaussian with tail component (simplified hypermet)
        More stable for fitting than full hypermet
        
        Args:
            tail_fraction: Fraction of intensity in tail (0-1)
            tail_sigma: Width of tail relative to main peak (typically 2-5x sigma)
        """
        # Main Gaussian
        main_peak = (1 - tail_fraction) * PeakFitter.gaussian(x, amplitude, center, sigma)
        
        # Tail component (wider Gaussian on low-energy side)
        tail_peak = tail_fraction * PeakFitter.gaussian(x, amplitude, center - 0.5 * sigma, tail_sigma)
        
        return main_peak + tail_peak
    
    @staticmethod
    def find_peaks(energy, counts, prominence=None, distance=None, height=None,
                   prominence_percent=None, min_separation_ev=None,
                   min_energy_kev=None):
        """
        Find peaks in spectrum using scipy peak detection
        
        Args:
            energy: Energy array
            counts: Counts array (background-subtracted recommended)
            prominence: Absolute minimum peak prominence (counts)
            distance: Minimum distance between peaks (in indices)
            height: Minimum peak height
            prominence_percent: If set, prominence = percent/100 * max(counts)
            min_separation_ev: If set, convert to channel distance using energy spacing
            min_energy_kev: Ignore detections below this energy (default
                MIN_PEAK_ENERGY_KEV = 0.17 keV, below C Kα)
            
        Returns:
            List of (energy, height) tuples for detected peaks
        """
        counts = np.asarray(counts, dtype=float)
        energy = np.asarray(energy, dtype=float)
        if min_energy_kev is None:
            min_energy_kev = PeakFitter.MIN_PEAK_ENERGY_KEV
        else:
            min_energy_kev = float(min_energy_kev)
        
        if prominence is None:
            if prominence_percent is not None:
                prominence = np.max(counts) * (float(prominence_percent) / 100.0)
            else:
                # Auto-calculate prominence as 2% of max (more sensitive default)
                prominence = np.max(counts) * 0.02
        
        if distance is None:
            if min_separation_ev is not None and len(energy) > 1:
                de_kev = float(np.median(np.diff(energy)))
                if de_kev > 0:
                    distance = max(1, int(round((float(min_separation_ev) / 1000.0) / de_kev)))
                else:
                    distance = 10
            else:
                # Default: at least 10 channels apart
                distance = 10
        
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            counts,
            prominence=prominence,
            distance=distance,
            height=height
        )
        
        # Extract peak information (skip electronic zero/noise below floor)
        peaks = []
        for idx in peak_indices:
            peak_energy = float(energy[idx])
            if peak_energy < min_energy_kev:
                continue
            peak_height = counts[idx]
            peaks.append((peak_energy, peak_height))
        
        return peaks
    
    @classmethod
    def center_shift_limit(cls, energy_kev, known_line=False, center_tolerance=None):
        """
        Allowed ±center shift (keV) during least-squares refinement.

        Known tabulated lines stay tighter; auto-found peaks use a
        FWHM-fraction window capped at CENTER_SHIFT_MAX_KEV.
        """
        if center_tolerance is not None:
            return float(center_tolerance)
        if known_line:
            return float(cls.KNOWN_LINE_CENTER_SHIFT_KEV)
        fwhm = float(cls.calculate_fwhm(energy_kev))
        return float(np.clip(
            cls.CENTER_SHIFT_FRACTION * fwhm,
            cls.CENTER_SHIFT_MIN_KEV,
            cls.CENTER_SHIFT_MAX_KEV,
        ))

    @staticmethod
    def fit_single_peak(energy, counts, initial_center, shape='gaussian', 
                       bounds=None, known_line=False, fix_center=False,
                       center_tolerance=None, fixed_fwhm=None):
        """
        Fit a single peak
        
        Args:
            energy: Energy array
            counts: Counts array
            initial_center: Initial guess for peak center
            shape: 'gaussian', 'lorentzian', 'voigt', or 'pseudo_voigt'
            bounds: Parameter bounds (optional override)
            known_line: If True, use tighter center bounds (tabulated line)
            fix_center: If True, hold center at initial_center (amp/width only)
            center_tolerance: Optional explicit ±center shift in keV
            fixed_fwhm: If set (keV), lock width to this value (e.g. Compton)
            
        Returns:
            Peak object with fitted parameters
        """
        if float(initial_center) < PeakFitter.MIN_PEAK_ENERGY_KEV:
            return None

        # Define fitting window around peak
        # Use appropriate window for peak width (±3 FWHM is standard)
        if fixed_fwhm is not None:
            fwhm_estimate = float(fixed_fwhm)
        else:
            fwhm_estimate = PeakFitter.calculate_fwhm(initial_center)
        
        # Use wider window for low-energy peaks due to more overlap
        # Compton / fixed-wide peaks need a window matching their breadth
        if fixed_fwhm is not None:
            window_width = 3.0 * fwhm_estimate
        elif initial_center < 3.0:  # Low energy (Si, Al, Mg, Na)
            window_width = 5.0 * fwhm_estimate  # Wider window
        else:
            window_width = 3.0 * fwhm_estimate  # Standard window
        
        mask = np.abs(energy - initial_center) < window_width
        
        if np.sum(mask) < 5:
            # Not enough points
            return None
        
        x_fit = energy[mask]
        y_fit = counts[mask]
        
        # Initial parameter guesses
        amplitude_guess = np.max(y_fit)
        center_guess = float(initial_center)
        # Use energy-dependent FWHM for better initial guess (or locked Compton width)
        fwhm_guess = fwhm_estimate
        sigma_guess = fwhm_guess / 2.355  # Convert FWHM to sigma
        # Lock width when calibrated shapes are on OR a per-peak fixed_fwhm is given
        lock_width = PeakFitter.USE_CALIBRATED_SHAPES or (fixed_fwhm is not None)
        dE = 0.0 if fix_center else PeakFitter.center_shift_limit(
            center_guess, known_line=known_line, center_tolerance=center_tolerance
        )
        c_lo, c_hi = center_guess - dE, center_guess + dE
        
        try:
            shape_params = {}
            
            if shape == 'gaussian':
                if lock_width:
                    sigma_fixed = sigma_guess
                    if fix_center:
                        def gaussian_amp_only(x, amplitude):
                            return PeakFitter.gaussian(
                                x, amplitude, center_guess, sigma_fixed
                            )
                        popt, _ = optimize.curve_fit(
                            gaussian_amp_only, x_fit, y_fit,
                            p0=[amplitude_guess],
                            bounds=([0], [np.inf]),
                            maxfev=5000,
                        )
                        amplitude = popt[0]
                        center = center_guess
                    else:
                        def gaussian_fixed_shape(x, amplitude, center):
                            return PeakFitter.gaussian(x, amplitude, center, sigma_fixed)
                        
                        p0 = [amplitude_guess, center_guess]
                        if bounds is None:
                            bounds = ([0, c_lo], [np.inf, c_hi])
                        
                        popt, _ = optimize.curve_fit(
                            gaussian_fixed_shape, x_fit, y_fit, p0=p0, bounds=bounds,
                            maxfev=5000
                        )
                        amplitude, center = popt
                    sigma = sigma_fixed
                else:
                    if fix_center:
                        def gaussian_fixed_center(x, amplitude, sigma):
                            return PeakFitter.gaussian(
                                x, amplitude, center_guess, sigma
                            )
                        p0 = [amplitude_guess, sigma_guess]
                        if bounds is None:
                            bounds = (
                                [0, sigma_guess * 0.5],
                                [np.inf, sigma_guess * 2.0],
                            )
                        popt, _ = optimize.curve_fit(
                            gaussian_fixed_center, x_fit, y_fit, p0=p0,
                            bounds=bounds, maxfev=5000,
                        )
                        amplitude, sigma = popt
                        center = center_guess
                    else:
                        p0 = [amplitude_guess, center_guess, sigma_guess]
                        if bounds is None:
                            # Allow FWHM to refine within reasonable physical limits
                            bounds = ([0, c_lo, sigma_guess * 0.5],
                                     [np.inf, c_hi, sigma_guess * 2.0])
                        
                        popt, _ = optimize.curve_fit(
                            PeakFitter.gaussian, x_fit, y_fit, p0=p0, bounds=bounds,
                            maxfev=5000
                        )
                        amplitude, center, sigma = popt
                fwhm = 2.355 * sigma  # FWHM = 2.355 * sigma for Gaussian
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma}
            
            elif shape == 'voigt':
                # Use calibrated gamma ratio if available
                gamma_guess = sigma_guess * PeakFitter.VOIGT_GAMMA_RATIO
                
                if lock_width:
                    # Fix peak shape, only fit amplitude and center
                    sigma_fixed = sigma_guess
                    gamma_fixed = gamma_guess
                    
                    if fix_center:
                        def voigt_amp_only(x, amplitude):
                            return PeakFitter.voigt(
                                x, amplitude, center_guess, sigma_fixed, gamma_fixed
                            )
                        popt, _ = optimize.curve_fit(
                            voigt_amp_only, x_fit, y_fit,
                            p0=[amplitude_guess],
                            bounds=([0], [np.inf]),
                            maxfev=5000,
                        )
                        amplitude = popt[0]
                        center = center_guess
                    else:
                        def voigt_fixed_shape(x, amplitude, center):
                            return PeakFitter.voigt(x, amplitude, center, sigma_fixed, gamma_fixed)
                        
                        p0 = [amplitude_guess, center_guess]
                        if bounds is None:
                            bounds = ([0, c_lo], [np.inf, c_hi])
                        
                        popt, _ = optimize.curve_fit(
                            voigt_fixed_shape, x_fit, y_fit, p0=p0, bounds=bounds,
                            maxfev=5000
                        )
                        amplitude, center = popt
                    sigma = sigma_fixed
                    gamma = gamma_fixed
                else:
                    # Fit all parameters including shape
                    if fix_center:
                        def voigt_fixed_center(x, amplitude, sigma, gamma):
                            return PeakFitter.voigt(
                                x, amplitude, center_guess, sigma, gamma
                            )
                        p0 = [amplitude_guess, sigma_guess, gamma_guess]
                        if bounds is None:
                            bounds = (
                                [0, sigma_guess * 0.5, 0.001],
                                [np.inf, sigma_guess * 2.0, sigma_guess * 2.0],
                            )
                        popt, _ = optimize.curve_fit(
                            voigt_fixed_center, x_fit, y_fit, p0=p0,
                            bounds=bounds, maxfev=5000,
                        )
                        amplitude, sigma, gamma = popt
                        center = center_guess
                    else:
                        p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess]
                        if bounds is None:
                            bounds = ([0, c_lo, sigma_guess * 0.5, 0.001],
                                     [np.inf, c_hi, sigma_guess * 2.0, sigma_guess * 2.0])
                        
                        popt, _ = optimize.curve_fit(
                            PeakFitter.voigt, x_fit, y_fit, p0=p0, bounds=bounds,
                            maxfev=5000
                        )
                        amplitude, center, sigma, gamma = popt
                
                # Approximate Voigt FWHM from Gaussian/Lorentzian components
                fwhm_g = 2.355 * sigma
                fwhm_l = 2.0 * gamma
                fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma, 'gamma': gamma}
            
            elif shape == 'pseudo_voigt':
                # Start with more Gaussian character (eta=0.3)
                if lock_width:
                    sigma_fixed = sigma_guess
                    eta_fixed = 0.3
                    if fix_center:
                        def pv_amp_only(x, amplitude):
                            return PeakFitter.pseudo_voigt(
                                x, amplitude, center_guess, sigma_fixed, eta_fixed
                            )
                        popt, _ = optimize.curve_fit(
                            pv_amp_only, x_fit, y_fit,
                            p0=[amplitude_guess],
                            bounds=([0], [np.inf]),
                            maxfev=5000,
                        )
                        amplitude = popt[0]
                        center = center_guess
                    else:
                        def pv_fixed_shape(x, amplitude, center):
                            return PeakFitter.pseudo_voigt(
                                x, amplitude, center, sigma_fixed, eta_fixed
                            )
                        p0 = [amplitude_guess, center_guess]
                        if bounds is None:
                            bounds = ([0, c_lo], [np.inf, c_hi])
                        popt, _ = optimize.curve_fit(
                            pv_fixed_shape, x_fit, y_fit, p0=p0,
                            bounds=bounds, maxfev=5000,
                        )
                        amplitude, center = popt
                    sigma = sigma_fixed
                    eta = eta_fixed
                elif fix_center:
                    def pv_fixed_center(x, amplitude, sigma, eta):
                        return PeakFitter.pseudo_voigt(
                            x, amplitude, center_guess, sigma, eta
                        )
                    p0 = [amplitude_guess, sigma_guess, 0.3]
                    if bounds is None:
                        bounds = (
                            [0, sigma_guess * 0.3, 0],
                            [np.inf, sigma_guess * 3.0, 1],
                        )
                    popt, _ = optimize.curve_fit(
                        pv_fixed_center, x_fit, y_fit, p0=p0,
                        bounds=bounds, maxfev=5000,
                    )
                    amplitude, sigma, eta = popt
                    center = center_guess
                else:
                    p0 = [amplitude_guess, center_guess, sigma_guess, 0.3]
                    if bounds is None:
                        bounds = ([0, c_lo, sigma_guess * 0.3, 0],
                                 [np.inf, c_hi, sigma_guess * 3.0, 1])
                    
                    popt, _ = optimize.curve_fit(
                        PeakFitter.pseudo_voigt, x_fit, y_fit, p0=p0, bounds=bounds,
                        maxfev=5000
                    )
                    amplitude, center, sigma, eta = popt
                fwhm = 2.355 * sigma
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma, 'eta': eta}
            
            elif shape == 'hypermet':
                if lock_width:
                    sigma_fixed = sigma_guess
                    tail_amp_fixed = 0.1
                    tail_slope_fixed = 2.0
                    if fix_center:
                        def hypermet_amp_only(x, amplitude):
                            return PeakFitter.hypermet(
                                x, amplitude, center_guess, sigma_fixed,
                                tail_amp_fixed, tail_slope_fixed
                            )
                        popt, _ = optimize.curve_fit(
                            hypermet_amp_only, x_fit, y_fit,
                            p0=[amplitude_guess],
                            bounds=([0], [np.inf]),
                            maxfev=5000,
                        )
                        amplitude = popt[0]
                        center = center_guess
                    else:
                        def hypermet_fixed_shape(x, amplitude, center):
                            return PeakFitter.hypermet(
                                x, amplitude, center, sigma_fixed,
                                tail_amp_fixed, tail_slope_fixed
                            )
                        p0 = [amplitude_guess, center_guess]
                        if bounds is None:
                            bounds = ([0, c_lo], [np.inf, c_hi])
                        popt, _ = optimize.curve_fit(
                            hypermet_fixed_shape, x_fit, y_fit, p0=p0,
                            bounds=bounds, maxfev=5000,
                        )
                        amplitude, center = popt
                    sigma = sigma_fixed
                    tail_amp = tail_amp_fixed
                    tail_slope = tail_slope_fixed
                elif fix_center:
                    def hypermet_fixed_center(x, amplitude, sigma, tail_amp, tail_slope):
                        return PeakFitter.hypermet(
                            x, amplitude, center_guess, sigma, tail_amp, tail_slope
                        )
                    p0 = [amplitude_guess, sigma_guess, 0.1, 2.0]
                    if bounds is None:
                        bounds = (
                            [0, sigma_guess * 0.3, 0, 0.5],
                            [np.inf, sigma_guess * 3.0, 0.5, 10],
                        )
                    popt, _ = optimize.curve_fit(
                        hypermet_fixed_center, x_fit, y_fit, p0=p0,
                        bounds=bounds, maxfev=5000,
                    )
                    amplitude, sigma, tail_amp, tail_slope = popt
                    center = center_guess
                else:
                    p0 = [amplitude_guess, center_guess, sigma_guess, 0.1, 2.0]
                    if bounds is None:
                        bounds = ([0, c_lo, sigma_guess * 0.3, 0, 0.5],
                                 [np.inf, c_hi, sigma_guess * 3.0, 0.5, 10])
                    
                    popt, _ = optimize.curve_fit(
                        PeakFitter.hypermet, x_fit, y_fit, p0=p0, bounds=bounds,
                        maxfev=5000
                    )
                    amplitude, center, sigma, tail_amp, tail_slope = popt
                fwhm = 2.355 * sigma
                area = amplitude * sigma * np.sqrt(2 * np.pi) * (1 + tail_amp)
                shape_params = {'sigma': sigma, 'tail_amplitude': tail_amp, 'tail_slope': tail_slope}
            
            elif shape == 'tail_gaussian':
                if lock_width:
                    sigma_fixed = sigma_guess
                    tail_frac_fixed = 0.15
                    tail_sigma_fixed = sigma_guess * 3
                    if fix_center:
                        def tg_amp_only(x, amplitude):
                            return PeakFitter.tail_gaussian(
                                x, amplitude, center_guess, sigma_fixed,
                                tail_frac_fixed, tail_sigma_fixed
                            )
                        popt, _ = optimize.curve_fit(
                            tg_amp_only, x_fit, y_fit,
                            p0=[amplitude_guess],
                            bounds=([0], [np.inf]),
                            maxfev=5000,
                        )
                        amplitude = popt[0]
                        center = center_guess
                    else:
                        def tg_fixed_shape(x, amplitude, center):
                            return PeakFitter.tail_gaussian(
                                x, amplitude, center, sigma_fixed,
                                tail_frac_fixed, tail_sigma_fixed
                            )
                        p0 = [amplitude_guess, center_guess]
                        if bounds is None:
                            bounds = ([0, c_lo], [np.inf, c_hi])
                        popt, _ = optimize.curve_fit(
                            tg_fixed_shape, x_fit, y_fit, p0=p0,
                            bounds=bounds, maxfev=5000,
                        )
                        amplitude, center = popt
                    sigma = sigma_fixed
                    tail_frac = tail_frac_fixed
                    tail_sigma = tail_sigma_fixed
                elif fix_center:
                    def tg_fixed_center(x, amplitude, sigma, tail_frac, tail_sigma):
                        return PeakFitter.tail_gaussian(
                            x, amplitude, center_guess, sigma, tail_frac, tail_sigma
                        )
                    p0 = [amplitude_guess, sigma_guess, 0.15, sigma_guess * 3]
                    if bounds is None:
                        bounds = (
                            [0, sigma_guess * 0.3, 0, sigma_guess],
                            [np.inf, sigma_guess * 3.0, 0.5, sigma_guess * 10],
                        )
                    popt, _ = optimize.curve_fit(
                        tg_fixed_center, x_fit, y_fit, p0=p0,
                        bounds=bounds, maxfev=5000,
                    )
                    amplitude, sigma, tail_frac, tail_sigma = popt
                    center = center_guess
                else:
                    p0 = [amplitude_guess, center_guess, sigma_guess, 0.15, sigma_guess * 3]
                    if bounds is None:
                        bounds = ([0, c_lo, sigma_guess * 0.3, 0, sigma_guess],
                                 [np.inf, c_hi, sigma_guess * 3.0, 0.5, sigma_guess * 10])
                    
                    popt, _ = optimize.curve_fit(
                        PeakFitter.tail_gaussian, x_fit, y_fit, p0=p0, bounds=bounds,
                        maxfev=5000
                    )
                    amplitude, center, sigma, tail_frac, tail_sigma = popt
                fwhm = 2.355 * sigma
                area = amplitude * sigma * np.sqrt(2 * np.pi)
                shape_params = {'sigma': sigma, 'tail_fraction': tail_frac, 'tail_sigma': tail_sigma}
            
            else:
                raise ValueError(f"Unknown peak shape: {shape}")

            if float(center) < PeakFitter.MIN_PEAK_ENERGY_KEV:
                return None
            
            return Peak(
                energy=center,
                amplitude=amplitude,
                fwhm=fwhm,
                area=area,
                shape=shape,
                shape_params=shape_params,
                fixed_fwhm=float(fixed_fwhm) if fixed_fwhm is not None else None,
            )
        
        except Exception as e:
            print(f"Peak fitting failed at {initial_center:.3f} keV: {e}")
            return None
    
    @staticmethod
    def fit_multiple_peaks(energy, counts, peak_positions, shape='gaussian'):
        """
        Fit multiple peaks simultaneously
        
        Args:
            energy: Energy array
            counts: Counts array
            peak_positions: List of initial peak center energies
            shape: Peak shape to use
            
        Returns:
            List of fitted Peak objects
        """
        fitted_peaks = []
        
        for center in peak_positions:
            peak = PeakFitter.fit_single_peak(
                energy, counts, center, shape=shape
            )
            if peak is not None:
                fitted_peaks.append(peak)
        
        return fitted_peaks
    
    @staticmethod
    def calculate_residuals(energy, counts, fitted_peaks, background, shape='gaussian'):
        """
        Calculate residuals between data and fit
        
        Args:
            energy: Energy array
            counts: Original counts
            fitted_peaks: List of fitted Peak objects
            background: Background array
            shape: Peak shape used
            
        Returns:
            Residuals array
        """
        # Reconstruct fitted spectrum
        fitted_spectrum = np.copy(background)
        
        for peak in fitted_peaks:
            sigma = peak.fwhm / 2.355  # Convert FWHM to sigma
            
            if shape == 'gaussian':
                fitted_spectrum += PeakFitter.gaussian(
                    energy, peak.amplitude, peak.energy, sigma
                )
        
        residuals = counts - fitted_spectrum
        return residuals
    
    @staticmethod
    def calculate_fit_statistics(counts, fitted_counts, n_params):
        """
        Calculate goodness-of-fit statistics
        
        Args:
            counts: Original counts
            fitted_counts: Fitted counts
            n_params: Number of fit parameters
            
        Returns:
            Dictionary with chi-squared, reduced chi-squared, and R-squared
        """
        residuals = counts - fitted_counts
        
        # Chi-squared (assuming Poisson statistics)
        # Avoid division by zero
        variance = np.where(counts > 0, counts, 1)
        chi_squared = np.sum(residuals**2 / variance)
        
        # Reduced chi-squared
        n_points = len(counts)
        dof = n_points - n_params  # Degrees of freedom
        reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((counts - np.mean(counts))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'r_squared': r_squared,
            'dof': dof
        }
