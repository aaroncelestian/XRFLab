"""
Empirical element calibration from certified reference standards.

Workflow
--------
1. Each standard has a certified composition (wt%) and one or more replicate
   spot spectra.
2. Every spectrum is fitted with the same element list (union of certified
   elements) using the shared SpectrumFitter, so net peak areas are extracted
   consistently. Areas are normalised to counts per second.
3. For each element the replicate spots of a standard collapse to one
   calibration point: mean intensity, replicate standard deviation (instrument
   precision at that concentration) and the certified concentration.
4. A weighted regression C = f(I) is fitted per element over the *included*
   standards. Curves, points and raw spot intensities are kept so standards or
   points can be toggled and the curves rebuilt instantly without re-fitting
   spectra. Weak curves can drop a few wrecking standards (greedy
   leave-one-out) when that produces a usable positive-slope fit.
5. `StandardsCalibration.quantify` converts fitted peaks of an unknown into
   wt% with a propagated uncertainty.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CALIBRATION_TYPE = "standards_curves"

MODEL_LINEAR = "linear"
MODEL_THROUGH_ORIGIN = "through_origin"
MODEL_QUADRATIC = "quadratic"
MODELS = (MODEL_LINEAR, MODEL_THROUGH_ORIGIN, MODEL_QUADRATIC)

LINE_SERIES = "series"  # every line of the principal series: K if excited, else L, else M
LINE_AUTO = "auto"      # Kα if excited, else Lα, else Mα
LINE_ALL = "all"        # every fitted line of the element (mixes series)
LINE_DEFAULT = LINE_SERIES

# Minimum Z we try to calibrate (lighter elements are not measurable in air)
MIN_Z = 11
# Tube kV / K-edge below this → K fluorescence is too weak; use L (or M).
# At 50 kV this keeps Fe–Sn on K and moves Ba / REE onto L.
K_OVERVOLTAGE_MIN = 1.5
# R² below this is reported as a weak-correlation warning (still fitted).
WEAK_R_SQUARED = 0.5
# Currie / IUPAC factor: I_MDC = 3·σ, then mapped through the (positive) slope.
MDC_SIGMA = 3.0
# Outlier search: leave already-good curves alone, stop once R² reaches the
# target, and never drop a point unless it improves R² by at least this much.
OUTLIER_SKIP_R2 = 0.80
OUTLIER_TARGET_R2 = 0.80
OUTLIER_MIN_R2_GAIN = 0.08
OUTLIER_MAX_DROP_FRAC = 1.0 / 3.0
# Currie / IUPAC factor: I_MDC = 3·σ, then mapped through the (positive) slope.
MDC_SIGMA = 3.0


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class StandardRecord:
    """A certified reference material and the spectra measured on it."""

    name: str
    concentrations: Dict[str, float]           # element → wt%
    spectrum_paths: List[str] = field(default_factory=list)
    enabled: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardRecord":
        return cls(
            name=str(data.get("name", "")),
            concentrations={
                str(k): float(v) for k, v in (data.get("concentrations") or {}).items()
            },
            spectrum_paths=[str(p) for p in data.get("spectrum_paths") or []],
            enabled=bool(data.get("enabled", True)),
            notes=str(data.get("notes", "") or ""),
        )


@dataclass
class SpotIntensity:
    """Net intensity of one element in one spectrum."""

    spectrum: str            # file name (display)
    path: str
    live_time: float
    area: float              # net counts
    area_err: float          # counting-statistics σ on the area
    cps: float               # area / live_time (or area if unnormalised)
    cps_err: float
    lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpotIntensity":
        return cls(
            spectrum=str(d.get("spectrum", "")),
            path=str(d.get("path", "")),
            live_time=float(d.get("live_time", 0.0) or 0.0),
            area=float(d.get("area", 0.0)),
            area_err=float(d.get("area_err", 0.0)),
            cps=float(d.get("cps", 0.0)),
            cps_err=float(d.get("cps_err", 0.0)),
            lines=[str(x) for x in d.get("lines") or []],
        )


@dataclass
class StandardPoint:
    """One standard collapsed to a single calibration point for an element."""

    standard: str
    concentration: float          # certified wt%
    intensity: float              # mean cps over spots
    intensity_sd: float           # replicate SD (instrument precision)
    intensity_sem: float          # SD/√n, floored by counting statistics
    n_spots: int
    rsd_percent: float            # 100·SD/mean
    included: bool = True
    predicted: Optional[float] = None   # wt% from the fitted curve
    residual: Optional[float] = None    # predicted − certified (wt%)
    spots: List[SpotIntensity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spots"] = [s.to_dict() for s in self.spots]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardPoint":
        return cls(
            standard=str(d.get("standard", "")),
            concentration=float(d.get("concentration", 0.0)),
            intensity=float(d.get("intensity", 0.0)),
            intensity_sd=float(d.get("intensity_sd", 0.0)),
            intensity_sem=float(d.get("intensity_sem", 0.0)),
            n_spots=int(d.get("n_spots", 0)),
            rsd_percent=float(d.get("rsd_percent", 0.0)),
            included=bool(d.get("included", True)),
            predicted=d.get("predicted"),
            residual=d.get("residual"),
            spots=[SpotIntensity.from_dict(s) for s in d.get("spots") or []],
        )


@dataclass
class ElementCurve:
    """Calibration curve C(wt%) = c0 + c1·I + c2·I² for one element."""

    element: str
    line_group: str                       # e.g. 'Kα', 'Lα', 'all'
    model: str                            # one of MODELS
    coefficients: List[float]             # [c0, c1, c2]; unused terms are 0
    coefficient_errors: List[float]
    r_squared: float
    rmse: float                           # wt%, over included points
    n_standards: int
    n_spectra: int
    mean_rsd_percent: float               # average replicate RSD → precision
    residual_variance: float              # wt%², for prediction uncertainty
    covariance: List[List[float]]         # of fitted coefficients (p×p)
    points: List[StandardPoint]
    enabled: bool = True
    fitted: bool = False
    message: str = ""

    # ------------------------------------------------------------------ #
    @property
    def slope(self) -> float:
        """dC/dI at I = 0 (wt% per cps)."""
        return self.coefficients[1] if len(self.coefficients) > 1 else 0.0

    @property
    def intercept(self) -> float:
        return self.coefficients[0] if self.coefficients else 0.0

    @property
    def sensitivity(self) -> float:
        """cps per wt% (inverse of slope), 0 if undefined."""
        return 1.0 / self.slope if self.slope else 0.0

    def included_points(self) -> List[StandardPoint]:
        return [p for p in self.points if p.included]

    def slope_at(self, intensity: float = 0.0) -> float:
        """dC/dI at a given intensity (wt% per cps)."""
        c = list(self.coefficients) + [0.0, 0.0, 0.0]
        if self.model == MODEL_QUADRATIC:
            return c[1] + 2.0 * c[2] * float(intensity)
        return c[1]

    def evaluate(self, intensity: float) -> float:
        c = list(self.coefficients) + [0.0, 0.0, 0.0]
        return c[0] + c[1] * intensity + c[2] * intensity ** 2

    def minimum_detectable_concentration(self) -> Optional[float]:
        """
        Predicted minimum detectable concentration (wt%).

        Always ≥ 0 when defined. Currie 3σ intensity is mapped through the
        positive slope as |C(I_MDC) − C(0)| so a negative intercept cannot
        produce a negative MDC. Residual scatter is preferred; the median
        replicate SEM is the fallback when there is no residual dof.
        """
        if not self.fitted:
            return None
        pts = self.included_points()
        if self.model == MODEL_QUADRATIC and pts:
            i_ref = float(np.mean([p.intensity for p in pts]))
        else:
            i_ref = 0.0
        dcdi = self.slope_at(i_ref)
        if dcdi <= 0:
            return None
        s = math.sqrt(max(self.residual_variance, 0.0))
        dof = len(pts) - _min_points(self.model)
        if s > 0 and dof > 0:
            i_mdc = MDC_SIGMA * s / dcdi
            mdc = abs(self.evaluate(i_mdc) - self.evaluate(0.0))
        else:
            sems = [p.intensity_sem for p in pts if p.intensity_sem > 0]
            if not sems:
                return None
            mdc = dcdi * MDC_SIGMA * float(np.median(sems))
        if not math.isfinite(mdc) or mdc < 0:
            return None
        return float(mdc)

    def predict(self, intensity: float, intensity_err: float = 0.0) -> Tuple[float, float]:
        """
        Return (wt%, 1σ uncertainty) for a measured intensity.

        Uncertainty combines the curve's residual scatter, the coefficient
        covariance, and the measurement error propagated through dC/dI.
        """
        if not self.fitted:
            return float("nan"), float("nan")
        conc = self.evaluate(intensity)
        x = _design_row(intensity, self.model)
        cov = np.asarray(self.covariance, dtype=float)
        var_coef = 0.0
        if cov.size and cov.shape[0] == len(x):
            var_coef = float(x @ cov @ x)
        c = list(self.coefficients) + [0.0, 0.0, 0.0]
        dcdi = c[1] + 2.0 * c[2] * intensity
        var_meas = (dcdi * intensity_err) ** 2
        var = max(self.residual_variance, 0.0) + max(var_coef, 0.0) + var_meas
        return float(conc), float(math.sqrt(var))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [p.to_dict() for p in self.points]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ElementCurve":
        return cls(
            element=str(d.get("element", "")),
            line_group=str(d.get("line_group", LINE_AUTO)),
            model=str(d.get("model", MODEL_LINEAR)),
            coefficients=[float(x) for x in d.get("coefficients") or [0.0, 0.0, 0.0]],
            coefficient_errors=[float(x) for x in d.get("coefficient_errors") or [0.0, 0.0, 0.0]],
            r_squared=float(d.get("r_squared", 0.0)),
            rmse=float(d.get("rmse", 0.0)),
            n_standards=int(d.get("n_standards", 0)),
            n_spectra=int(d.get("n_spectra", 0)),
            mean_rsd_percent=float(d.get("mean_rsd_percent", 0.0)),
            residual_variance=float(d.get("residual_variance", 0.0)),
            covariance=[[float(v) for v in row] for row in d.get("covariance") or []],
            points=[StandardPoint.from_dict(p) for p in d.get("points") or []],
            enabled=bool(d.get("enabled", True)),
            fitted=bool(d.get("fitted", False)),
            message=str(d.get("message", "") or ""),
        )


# --------------------------------------------------------------------------- #
# Regression helpers
# --------------------------------------------------------------------------- #
def _design_row(intensity: float, model: str) -> np.ndarray:
    """Row of the design matrix over the *free* parameters of the model."""
    if model == MODEL_THROUGH_ORIGIN:
        return np.array([intensity])
    if model == MODEL_QUADRATIC:
        return np.array([1.0, intensity, intensity ** 2])
    return np.array([1.0, intensity])


def _min_points(model: str) -> int:
    return {MODEL_THROUGH_ORIGIN: 1, MODEL_LINEAR: 2, MODEL_QUADRATIC: 3}.get(model, 2)


def fit_curve(
    intensities: Sequence[float],
    concentrations: Sequence[float],
    intensity_errors: Optional[Sequence[float]] = None,
    model: str = MODEL_LINEAR,
    weighted: bool = True,
) -> Dict[str, Any]:
    """
    Weighted least squares of C = f(I).

    Errors live on the intensity axis, so a two-pass scheme is used: an
    unweighted fit gives dC/dI, which converts σ_I into σ_C for the weights.

    Returns dict with coefficients [c0, c1, c2], errors, covariance (over the
    free parameters, ordered like the non-zero coefficients), r_squared, rmse,
    residual_variance, predicted, residuals.
    """
    I = np.asarray(intensities, dtype=float)
    C = np.asarray(concentrations, dtype=float)
    n = I.size
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'")
    if n < _min_points(model):
        raise ValueError(
            f"{model} needs at least {_min_points(model)} standard(s); got {n}"
        )

    if model == MODEL_THROUGH_ORIGIN:
        X = I[:, None]
    elif model == MODEL_QUADRATIC:
        X = np.column_stack([np.ones(n), I, I ** 2])
    else:
        X = np.column_stack([np.ones(n), I])
    p = X.shape[1]

    def _solve(w: np.ndarray):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], C * sw, rcond=None)
        return beta

    w = np.ones(n)
    beta = _solve(w)

    sig_I = None
    if weighted and intensity_errors is not None:
        sig_I = np.asarray(intensity_errors, dtype=float)
        sig_I = np.where(np.isfinite(sig_I), sig_I, 0.0)
        if np.any(sig_I > 0):
            # Effective σ on the concentration axis
            if model == MODEL_THROUGH_ORIGIN:
                dcdi = np.full(n, beta[0])
            elif model == MODEL_QUADRATIC:
                dcdi = beta[1] + 2.0 * beta[2] * I
            else:
                dcdi = np.full(n, beta[1])
            sig_C = np.abs(dcdi) * sig_I
            positive = sig_C[sig_C > 0]
            floor = 0.1 * np.median(positive) if positive.size else 1.0
            sig_C = np.maximum(sig_C, floor if floor > 0 else 1e-12)
            w = 1.0 / sig_C ** 2
            w = w / np.mean(w)
            beta = _solve(w)

    predicted = X @ beta
    resid = predicted - C
    dof = n - p
    chi2 = float(np.sum(w * resid ** 2))
    # Scale covariance by reduced chi² (standard WLS with unknown scale)
    scale = chi2 / dof if dof > 0 else 0.0
    try:
        xtwx_inv = np.linalg.inv((X * w[:, None]).T @ X)
    except np.linalg.LinAlgError:
        xtwx_inv = np.zeros((p, p))
    cov = xtwx_inv * scale
    # With no redundancy (dof = 0) the errors are undefined; report 0 and let
    # the caller flag the exact fit rather than emitting NaN into JSON.
    errs = np.sqrt(np.clip(np.diag(cov), 0, None)) if dof > 0 else np.zeros(p)
    cov = np.where(np.isfinite(cov), cov, 0.0)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((C - np.mean(C)) ** 2))
    if model == MODEL_THROUGH_ORIGIN and n < 2:
        r2 = 1.0
    elif ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0 if ss_res == 0 else 0.0
    rmse = math.sqrt(ss_res / n) if n else 0.0
    resid_var = ss_res / dof if dof > 0 else 0.0

    coefficients = [0.0, 0.0, 0.0]
    coef_errors = [0.0, 0.0, 0.0]
    if model == MODEL_THROUGH_ORIGIN:
        coefficients[1] = float(beta[0])
        coef_errors[1] = float(errs[0])
    else:
        for i, b in enumerate(beta):
            coefficients[i] = float(b)
            coef_errors[i] = float(errs[i])

    return {
        "coefficients": coefficients,
        "coefficient_errors": coef_errors,
        "covariance": cov.tolist(),
        "r_squared": float(r2),
        "rmse": float(rmse),
        "residual_variance": float(resid_var),
        "predicted": predicted.tolist(),
        "residuals": resid.tolist(),
        "dof": int(dof),
    }


def _usable_fit(result: Dict[str, Any]) -> bool:
    """Same acceptance rule as `build_curves`: positive slope and R² ≥ 0."""
    return result["coefficients"][1] > 0 and result["r_squared"] >= 0.0


def _concentration_span_ok(points: Sequence[StandardPoint], model: str) -> bool:
    need = 1 if model == MODEL_THROUGH_ORIGIN else 2
    return len({round(p.concentration, 9) for p in points}) >= need


def suggest_outlier_exclusions(
    points: Sequence[StandardPoint],
    model: str = MODEL_LINEAR,
    weighted: bool = True,
) -> List[str]:
    """
    Standards whose removal would rescue a weak or failed curve.

    Greedy leave-one-out: drop the point that most improves R², repeating
    until the fit is usable (positive slope, R² ≥ 0) and preferably
    R² ≥ OUTLIER_TARGET_R2. Already-good curves are left alone. Returns []
    when no subset with redundancy yields a usable fit — so a genuinely
    uncorrelated element (below detection, wrong line) is not "fixed" by
    throwing points away.
    """
    candidates = [p for p in points if p.included]
    # Keep at least one degree of freedom so a 2-point linear fit cannot
    # report a perfect R² just by dropping down to an exact line.
    need = _min_points(model) + 1
    if len(candidates) <= need:
        return []

    def _fit(subset: Sequence[StandardPoint]) -> Dict[str, Any]:
        return fit_curve(
            [p.intensity for p in subset],
            [p.concentration for p in subset],
            [p.intensity_sem for p in subset],
            model=model,
            weighted=weighted,
        )

    def _score(result: Dict[str, Any]) -> float:
        if result["coefficients"][1] <= 0:
            return float("-inf")
        return float(result["r_squared"])

    try:
        base = _fit(candidates)
    except Exception:
        return []
    if _usable_fit(base) and base["r_squared"] >= OUTLIER_SKIP_R2:
        return []

    dropped: List[str] = []
    remaining = list(candidates)
    score = _score(base)
    max_drop = max(1, int(len(candidates) * OUTLIER_MAX_DROP_FRAC))

    while len(dropped) < max_drop and len(remaining) > need:
        best: Optional[Tuple[float, StandardPoint, Dict[str, Any]]] = None
        for pt in remaining:
            trial = [q for q in remaining if q is not pt]
            if not _concentration_span_ok(trial, model):
                continue
            try:
                result = _fit(trial)
            except Exception:
                continue
            s = _score(result)
            if best is None or s > best[0]:
                best = (s, pt, result)
        if best is None or not math.isfinite(best[0]):
            break
        s, pt, result = best
        if s < score + OUTLIER_MIN_R2_GAIN:
            break
        dropped.append(pt.standard)
        remaining = [q for q in remaining if q is not pt]
        score = s
        if _usable_fit(result) and result["r_squared"] >= OUTLIER_TARGET_R2:
            break

    if not dropped:
        return []
    try:
        final = _fit(remaining)
    except Exception:
        return []
    if not _usable_fit(final) or not _concentration_span_ok(remaining, model):
        return []
    if _usable_fit(base) and final["r_squared"] < base["r_squared"] + OUTLIER_MIN_R2_GAIN:
        return []
    return dropped


# --------------------------------------------------------------------------- #
# Intensity extraction
# --------------------------------------------------------------------------- #
def line_group_of(line_name: Optional[str]) -> str:
    """'Kα1' → 'Kα', 'Lβ2' → 'Lβ', 'Compton Kα' → 'Compton'."""
    if not line_name:
        return "?"
    s = str(line_name)
    if s.startswith("Compton"):
        return "Compton"
    return s[:2] if len(s) >= 2 else s


def k_series_preferred(element: str, excitation_kv: float) -> bool:
    """True when the tube overvoltage is high enough for a useful K series."""
    from core.advanced_peak_fitting import get_element_z
    from core.xray_data import edge_energy_kev

    z = get_element_z(element)
    edge = edge_energy_kev(z, "K") if z else None
    if edge is None:
        return True
    return float(excitation_kv) >= float(edge) * K_OVERVOLTAGE_MIN


def choose_line_group(
    groups_present: Iterable[str],
    mode: str = LINE_AUTO,
    *,
    element: Optional[str] = None,
    excitation_kv: Optional[float] = None,
) -> str:
    """
    Pick the line group key for an element from the families fitted.

    mode LINE_AUTO   → 'Kα', else 'Lα', else 'Mα' (single family)
    mode LINE_SERIES → 'K', else 'L', else 'M' (whole series summed)
    Falls back to LINE_ALL when nothing matches.

    When `element` and `excitation_kv` are given, K is skipped unless the
    tube overvoltage is at least K_OVERVOLTAGE_MIN (Ba / REE at 50 kV → L).
    """
    present = {g for g in groups_present if g and g != "Compton"}
    skip_k = bool(
        element and excitation_kv is not None
        and not k_series_preferred(element, excitation_kv)
    )
    if mode == LINE_SERIES:
        series = {g[0] for g in present if g[0] in "KLM"}
        order = ("L", "M") if skip_k else ("K", "L", "M")
        for s in order:
            if s in series:
                return s
        if skip_k and "K" in series:
            return "K"
        return LINE_ALL
    families = ("Lα", "Mα") if skip_k else ("Kα", "Lα", "Mα")
    for g in families:
        if g in present:
            return g
    if skip_k and "Kα" in present:
        return "Kα"
    return LINE_ALL


def line_matches_group(line_name: Optional[str], group: str) -> bool:
    """
    Does a fitted line belong to a curve's line group?

    group is LINE_ALL, a series letter ('K'), or a family ('Kα').
    Compton humps never match.
    """
    g = line_group_of(line_name)
    if g == "Compton" or g == "?":
        return False
    if group == LINE_ALL:
        return True
    if len(group) == 1:
        return g[0] == group
    return g == group


def element_list_for_fit(
    standards: Iterable[StandardRecord],
    *,
    excitation_kv: float = 50.0,
    min_z: int = MIN_Z,
    extra_elements: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Union of certified elements (Z ≥ min_z, with an excitable line)."""
    from core.advanced_peak_fitting import get_element_z
    from core.xray_data import get_element_lines

    symbols = set(extra_elements)
    for std in standards:
        if not std.enabled:
            continue
        symbols.update(std.concentrations.keys())

    out: List[Dict[str, Any]] = []
    for sym in sorted(symbols, key=lambda s: get_element_z(s) or 999):
        z = get_element_z(sym)
        if not z or z < min_z:
            continue
        try:
            lines = get_element_lines(sym, z)
        except Exception:
            continue
        energies = [
            ln["energy"]
            for series in ("K", "L", "M")
            for ln in lines.get(series, [])
            if ln.get("energy", 0) < excitation_kv
        ]
        if not energies:
            continue
        out.append({"symbol": sym, "z": z})
    return out


def suggest_elements(
    standards: Iterable[StandardRecord],
    *,
    excitation_kv: float = 50.0,
    min_standards: int = 2,
    min_wt_pct: float = 0.01,
) -> List[str]:
    """
    Elements worth calibrating by default: certified in ≥ min_standards enabled
    standards and reaching ≥ min_wt_pct (100 ppm) in at least one of them.
    """
    stds = [s for s in standards if s.enabled]
    candidates = element_list_for_fit(stds, excitation_kv=excitation_kv)
    out = []
    for e in candidates:
        sym = e["symbol"]
        concs = [s.concentrations[sym] for s in stds if sym in s.concentrations]
        if len(concs) >= min_standards and max(concs) >= min_wt_pct:
            out.append(sym)
    return out


def _counting_error(area: float, background: np.ndarray, energy: np.ndarray,
                    center: float, fwhm: float) -> float:
    """σ_area ≈ √(net + background under ±FWHM)."""
    half = max(float(fwhm), 0.02)
    mask = np.abs(energy - center) <= half
    bg = float(np.sum(background[mask])) if background is not None and background.size == energy.size else 0.0
    return math.sqrt(max(area, 0.0) + max(bg, 0.0))


def extract_spot_intensities(
    spectrum,
    fit_result,
    elements: Iterable[str],
    *,
    line_groups: Dict[str, str],
    normalise: str = "live_time",
    path: str = "",
    name: str = "",
) -> Dict[str, SpotIntensity]:
    """
    Sum fitted peak areas per element for the chosen line group.

    line_groups maps element → 'K' / 'L' (whole series), 'Kα' / 'Lα' … (one
    family) or 'all'. normalise: 'live_time', 'real_time' or 'none'.
    """
    energy = np.asarray(spectrum.energy, dtype=float)
    background = np.asarray(getattr(fit_result, "background", None), dtype=float) \
        if getattr(fit_result, "background", None) is not None else None

    if normalise == "live_time":
        t = float(getattr(spectrum, "live_time", 0.0) or 0.0)
    elif normalise == "real_time":
        t = float(getattr(spectrum, "real_time", 0.0) or 0.0)
    else:
        t = 1.0
    if not t or t <= 0:
        t = 1.0

    out: Dict[str, SpotIntensity] = {}
    for sym in elements:
        group = line_groups.get(sym, LINE_ALL)
        area = 0.0
        var = 0.0
        lines: List[str] = []
        for pk in fit_result.peaks:
            if pk.element != sym or getattr(pk, "is_tube_line", False):
                continue
            if not line_matches_group(pk.line, group):
                continue
            a = float(pk.area or 0.0)
            area += a
            var += _counting_error(a, background, energy, pk.energy, pk.fwhm) ** 2
            lines.append(str(pk.line))
        err = math.sqrt(var)
        out[sym] = SpotIntensity(
            spectrum=name or Path(path).name,
            path=path,
            live_time=t,
            area=area,
            area_err=err,
            cps=area / t,
            cps_err=err / t,
            lines=lines,
        )
    return out


def collapse_spots(standard: str, concentration: float,
                   spots: Sequence[SpotIntensity]) -> StandardPoint:
    """Replicate spots → one point with mean, SD, SEM and RSD."""
    vals = np.array([s.cps for s in spots], dtype=float)
    n = vals.size
    mean = float(np.mean(vals)) if n else 0.0
    counting = math.sqrt(sum(s.cps_err ** 2 for s in spots)) / n if n else 0.0
    if n >= 2:
        sd = float(np.std(vals, ddof=1))
        sem = sd / math.sqrt(n)
    else:
        sd = counting
        sem = counting
    sem = max(sem, counting)  # never claim better than counting statistics
    rsd = 100.0 * sd / mean if mean > 0 else 0.0
    return StandardPoint(
        standard=standard,
        concentration=float(concentration),
        intensity=mean,
        intensity_sd=sd,
        intensity_sem=sem,
        n_spots=n,
        rsd_percent=rsd,
        spots=list(spots),
    )


# --------------------------------------------------------------------------- #
# Calibration container
# --------------------------------------------------------------------------- #
@dataclass
class StandardsCalibration:
    """
    Per-element calibration curves plus everything needed to rebuild them.

    `intensities[standard][spectrum_path][element]` holds the raw spot
    intensities from the last spectrum-fitting pass. Curves are rebuilt from
    those with `build_curves`, honouring `StandardRecord.enabled`,
    per-element point exclusions, and the current model/line settings.
    """

    standards: Dict[str, StandardRecord] = field(default_factory=dict)
    curves: Dict[str, ElementCurve] = field(default_factory=dict)
    intensities: Dict[str, Dict[str, Dict[str, SpotIntensity]]] = field(default_factory=dict)
    line_groups: Dict[str, str] = field(default_factory=dict)
    # element → set of standard names excluded from that element's curve
    excluded_points: Dict[str, List[str]] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    fit_settings: Dict[str, Any] = field(default_factory=dict)
    calibration_date: Optional[str] = None
    fwhm_calibration: Optional[Dict[str, Any]] = None
    overlap_models: List[Any] = field(default_factory=list)
    message: str = ""

    type: str = CALIBRATION_TYPE

    # ---- compatibility with the legacy CalibrationResult surface --------- #
    @property
    def success(self) -> bool:
        return any(c.fitted and c.enabled for c in self.curves.values())

    @property
    def fwhm_0(self) -> float:
        params = (self.fwhm_calibration or {}).get("parameters") or {}
        return float(params.get("fwhm_0", 0.0) or 0.0)

    @property
    def epsilon(self) -> float:
        params = (self.fwhm_calibration or {}).get("parameters") or {}
        return float(params.get("epsilon", 0.0) or 0.0)

    # ---- standards management ------------------------------------------- #
    def add_standard(self, record: StandardRecord) -> None:
        self.standards[record.name] = record

    def remove_standard(self, name: str) -> None:
        self.standards.pop(name, None)
        self.intensities.pop(name, None)
        for excl in self.excluded_points.values():
            if name in excl:
                excl.remove(name)

    def enabled_standards(self) -> List[StandardRecord]:
        return [s for s in self.standards.values() if s.enabled and s.spectrum_paths]

    def set_point_included(self, element: str, standard: str, included: bool) -> None:
        excl = self.excluded_points.setdefault(element, [])
        if included and standard in excl:
            excl.remove(standard)
        elif not included and standard not in excl:
            excl.append(standard)

    def is_point_included(self, element: str, standard: str) -> bool:
        return standard not in self.excluded_points.get(element, [])

    def clear_point_exclusions(self, element: Optional[str] = None) -> None:
        """Re-include every excluded standard, or just those of one element."""
        if element is None:
            self.excluded_points.clear()
        else:
            self.excluded_points.pop(element, None)

    def exclude_outliers(
        self,
        elements: Optional[Iterable[str]] = None,
        *,
        rebuild: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Drop wrecking standards on weak or failed curves.

        Only commits a drop list when the remaining points produce a usable
        fit (positive slope, R² ≥ 0). Returns {element: [standard, …]}.
        """
        if not self.curves:
            self.build_curves()
        wanted = None if elements is None else {str(e) for e in elements}
        weighted = bool(self.settings.get("weighted", True))
        applied: Dict[str, List[str]] = {}
        for el, curve in self.curves.items():
            if wanted is not None and el not in wanted:
                continue
            drops = suggest_outlier_exclusions(
                curve.points, model=curve.model, weighted=weighted,
            )
            if not drops:
                continue
            for name in drops:
                self.set_point_included(el, name, False)
            applied[el] = drops
        if rebuild and applied:
            self.build_curves()
            for el, names in applied.items():
                curve = self.curves.get(el)
                if curve is None:
                    continue
                note = "Excluded outlier(s): " + ", ".join(names)
                if curve.message:
                    curve.message = f"{curve.message}; {note}"
                else:
                    curve.message = note
        return applied

    def has_intensities(self) -> bool:
        return any(self.intensities.values())

    # ---- spectrum fitting pass ------------------------------------------ #
    def fit_spectra(
        self,
        spectra: Dict[str, List[Tuple[str, Any]]],
        fitter,
        *,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        elements: Optional[Iterable[str]] = None,
        line_selection: str = LINE_DEFAULT,
        normalise: str = "live_time",
        progress: Optional[Callable[[str, int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Fit every spot spectrum of every enabled standard and cache intensities.

        spectra:  {standard_name: [(path, Spectrum), ...]}
        fitter:   core.fitting.SpectrumFitter (with instrument state applied)
        elements: symbols to calibrate; default = every certified element
                  (Z ≥ 11) in the enabled standards. Fewer elements → faster.
        """
        fit_kwargs = dict(fit_kwargs or {})
        excitation = float(fit_kwargs.get("excitation_kv", 50.0))
        if elements is None:
            element_dicts = element_list_for_fit(self.standards.values(), excitation_kv=excitation)
        else:
            from core.advanced_peak_fitting import get_element_z

            element_dicts = [
                {"symbol": s, "z": get_element_z(s)}
                for s in elements if get_element_z(s)
            ]
        symbols = [e["symbol"] for e in element_dicts]
        if not symbols:
            raise ValueError("No calibratable elements (Z ≥ 11) in the enabled standards.")

        jobs = [
            (name, path, spec)
            for name, entries in spectra.items()
            if name in self.standards and self.standards[name].enabled
            for path, spec in entries
        ]
        total = len(jobs)
        if total == 0:
            raise ValueError("No spectra to fit — add spot spectra to at least one enabled standard.")

        fit_results: Dict[Tuple[str, str], Any] = {}
        for i, (name, path, spec) in enumerate(jobs, start=1):
            if should_stop and should_stop():
                raise InterruptedError("Calibration cancelled")
            if progress:
                progress(f"Fitting {name}: {Path(path).name}", i, total)
            result = fitter.fit_spectrum(
                energy=spec.energy,
                counts=spec.counts,
                elements=element_dicts,
                background_method=fit_kwargs.get("background_method", "snip"),
                peak_shape=fit_kwargs.get("peak_shape", "tail_gaussian"),
                auto_find_peaks=False,
                tube_element=fit_kwargs.get("tube_element", "Rh"),
                excitation_kv=excitation,
                include_tube_lines=fit_kwargs.get("include_tube_lines", True),
                include_compton=fit_kwargs.get("include_compton", True),
                grouped_lines=bool(fit_kwargs.get("grouped_lines", True)),
                reference_composition=dict(self.standards[name].concentrations),
            )
            fit_results[(name, path)] = result

        # Decide one line group per element (consistent across all spectra).
        # LINE_SERIES sums every line of the principal series (Kα+Kβ …): with
        # grouped fitting that is the sub-shell amplitude itself; with released
        # ratios it is far less sensitive to an overlap on one line.
        line_groups: Dict[str, str] = {}
        for sym in symbols:
            if line_selection == LINE_ALL:
                line_groups[sym] = LINE_ALL
                continue
            present = set()
            for res in fit_results.values():
                for pk in res.peaks:
                    if pk.element == sym and not getattr(pk, "is_tube_line", False):
                        present.add(line_group_of(pk.line))
            line_groups[sym] = choose_line_group(
                present, line_selection, element=sym, excitation_kv=excitation,
            )
        self.line_groups = line_groups

        new_intensities: Dict[str, Dict[str, Dict[str, SpotIntensity]]] = {}
        for (name, path), result in fit_results.items():
            spec = next(s for p, s in spectra[name] if p == path)
            spots = extract_spot_intensities(
                spec, result, symbols,
                line_groups=line_groups, normalise=normalise,
                path=path, name=Path(path).name,
            )
            new_intensities.setdefault(name, {})[path] = spots
        # Keep cached intensities for disabled standards so re-enabling is instant
        for name, per_path in self.intensities.items():
            if name not in new_intensities and name in self.standards:
                new_intensities[name] = per_path
        self.intensities = new_intensities

        self.settings.update({
            "line_selection": line_selection,
            "normalise": normalise,
        })
        self.fit_settings = {
            "background_method": fit_kwargs.get("background_method", "snip"),
            "peak_shape": fit_kwargs.get("peak_shape", "tail_gaussian"),
            "grouped_lines": bool(fit_kwargs.get("grouped_lines", True)),
            "tube_element": fit_kwargs.get("tube_element", "Rh"),
            "excitation_kv": excitation,
            "elements": symbols,
        }
        self.calibration_date = datetime.now().isoformat()

    # ---- curve building --------------------------------------------------- #
    def build_curves(
        self,
        model: Optional[str] = None,
        weighted: Optional[bool] = None,
        *,
        min_standards: int = 2,
    ) -> None:
        """Rebuild every element curve from cached intensities."""
        if model is None:
            model = self.settings.get("model", MODEL_LINEAR)
        if weighted is None:
            weighted = bool(self.settings.get("weighted", True))
        self.settings["model"] = model
        self.settings["weighted"] = bool(weighted)

        elements = set()
        for name, per_path in self.intensities.items():
            std = self.standards.get(name)
            if std is None or not std.enabled:
                continue
            for spots in per_path.values():
                elements.update(spots.keys())

        previous = self.curves
        new_curves: Dict[str, ElementCurve] = {}
        for sym in sorted(elements, key=_z_sort_key):
            points: List[StandardPoint] = []
            for name, per_path in self.intensities.items():
                std = self.standards.get(name)
                if std is None or not std.enabled or sym not in std.concentrations:
                    continue
                spots = [spots[sym] for spots in per_path.values() if sym in spots]
                # Only spots whose spectra are still attached to the standard
                spots = [s for s in spots if s.path in std.spectrum_paths] or spots
                if not spots:
                    continue
                pt = collapse_spots(name, std.concentrations[sym], spots)
                pt.included = self.is_point_included(sym, name)
                points.append(pt)
            points.sort(key=lambda p: p.concentration)

            curve = ElementCurve(
                element=sym,
                line_group=self.line_groups.get(sym, LINE_ALL),
                model=model,
                coefficients=[0.0, 0.0, 0.0],
                coefficient_errors=[0.0, 0.0, 0.0],
                r_squared=0.0,
                rmse=0.0,
                n_standards=0,
                n_spectra=0,
                mean_rsd_percent=0.0,
                residual_variance=0.0,
                covariance=[],
                points=points,
                enabled=previous.get(sym).enabled if sym in previous else True,
            )
            used = [p for p in points if p.included]
            curve.n_standards = len(used)
            curve.n_spectra = sum(p.n_spots for p in used)
            if used:
                curve.mean_rsd_percent = float(np.mean([p.rsd_percent for p in used]))

            need = max(min_standards, _min_points(model))
            if len(used) < need:
                curve.message = f"Needs ≥{need} included standards (have {len(used)})"
            elif len({round(p.concentration, 9) for p in used}) < (1 if model == MODEL_THROUGH_ORIGIN else 2):
                curve.message = "Standards span a single concentration"
            else:
                try:
                    res = fit_curve(
                        [p.intensity for p in used],
                        [p.concentration for p in used],
                        [p.intensity_sem for p in used],
                        model=model,
                        weighted=weighted,
                    )
                    curve.coefficients = res["coefficients"]
                    curve.coefficient_errors = res["coefficient_errors"]
                    curve.covariance = res["covariance"]
                    curve.r_squared = res["r_squared"]
                    curve.rmse = res["rmse"]
                    curve.residual_variance = res["residual_variance"]
                    for p in points:
                        p.predicted = curve.evaluate(p.intensity)
                        p.residual = p.predicted - p.concentration
                    if curve.slope <= 0:
                        curve.fitted = False
                        curve.message = (
                            "Negative or zero slope — intensity does not "
                            "increase with concentration"
                        )
                    elif curve.r_squared < 0:
                        curve.fitted = False
                        curve.message = (
                            f"No correlation (R² = {curve.r_squared:.2f}) — "
                            "wrong line or below detection"
                        )
                    else:
                        curve.fitted = True
                        if res["dof"] <= 0:
                            curve.message = "Exact fit — no redundancy; add standards"
                        elif curve.r_squared < WEAK_R_SQUARED:
                            curve.message = (
                                f"Weak correlation (R² = {curve.r_squared:.2f}) "
                                "— treat as semi-quantitative"
                            )
                except Exception as exc:  # pragma: no cover - defensive
                    curve.message = str(exc)
            new_curves[sym] = curve
        self.curves = new_curves
        self._learn_overlap_models()

    def _learn_overlap_models(self) -> None:
        """Build I = S C mixing matrices for unresolved line-group clusters."""
        from core.overlap_deconvolution import clusters_for_line_groups, learn_mixing_models
        from core.peak_fitting import PeakFitter

        if not self.line_groups or not self.intensities:
            self.overlap_models = []
            return
        clusters, _pairs, energies = clusters_for_line_groups(
            self.line_groups, PeakFitter.calculate_fwhm,
        )
        if not clusters:
            self.overlap_models = []
            return
        concs = {
            name: dict(rec.concentrations)
            for name, rec in self.standards.items()
            if rec.enabled
        }
        ints: Dict[str, Dict[str, float]] = {}
        for name, per_path in self.intensities.items():
            rec = self.standards.get(name)
            if rec is None or not rec.enabled:
                continue
            acc: Dict[str, List[float]] = {}
            for path, spots in per_path.items():
                if rec.spectrum_paths and path not in rec.spectrum_paths:
                    continue
                for el, spot in spots.items():
                    acc.setdefault(el, []).append(float(spot.cps))
            if acc:
                ints[name] = {el: float(np.mean(vals)) for el, vals in acc.items()}
        self.overlap_models = learn_mixing_models(
            clusters, concentrations=concs, intensities=ints, energies=energies,
            enabled=concs.keys(),
        )
        by_el = {el: m for m in self.overlap_models if m.usable for el in m.elements}
        for el, curve in self.curves.items():
            model = by_el.get(el)
            if model is None:
                continue
            note = model.message
            if curve.message:
                curve.message = f"{curve.message}; {note}"
            else:
                curve.message = note

    # ---- quantification of unknowns -------------------------------------- #
    def quantify(
        self,
        peaks,
        live_time: float,
        *,
        real_time: Optional[float] = None,
        fit_result=None,
        energy=None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert fitted peaks of an unknown into wt% using the element curves.

        Returns the same dict shape the Results panel expects:
        {element: {'concentration', 'error', 'lines', 'method', 'intensity_cps'}}
        """
        normalise = self.settings.get("normalise", "live_time")
        if normalise == "live_time":
            t = float(live_time or 0.0)
        elif normalise == "real_time":
            t = float(real_time or live_time or 0.0)
        else:
            t = 1.0
        if not t or t <= 0:
            t = 1.0

        background = None
        energy_arr = None
        if fit_result is not None and energy is not None:
            background = np.asarray(fit_result.background, dtype=float)
            energy_arr = np.asarray(energy, dtype=float)

        def _spot_cps(sym: str):
            group = self.line_groups.get(sym) or (
                self.curves[sym].line_group if sym in self.curves else LINE_ALL
            )
            area = 0.0
            var = 0.0
            lines: List[str] = []
            for pk in peaks:
                if pk.element != sym or getattr(pk, "is_tube_line", False):
                    continue
                if not line_matches_group(pk.line, group):
                    continue
                a = float(pk.area or 0.0)
                area += a
                if background is not None:
                    var += _counting_error(a, background, energy_arr, pk.energy, pk.fwhm) ** 2
                else:
                    var += max(a, 0.0)
                lines.append(str(pk.line))
            if not lines:
                return None
            return area / t, math.sqrt(var) / t, lines, group

        out: Dict[str, Dict[str, Any]] = {}
        overlap_done = set()
        for model in self.overlap_models or []:
            if not getattr(model, "usable", False):
                continue
            cps_map = {}
            lines_map = {}
            group_map = {}
            ok = True
            for el in model.elements:
                spot = _spot_cps(el)
                if spot is None:
                    ok = False
                    break
                cps_map[el], _err, lines_map[el], group_map[el] = spot
            if not ok:
                continue
            preds = model.predict(cps_map)
            for el, (conc, err) in preds.items():
                overlap_done.add(el)
                group = group_map.get(el, self.line_groups.get(el, LINE_ALL))
                lines = lines_map.get(el) or []
                if group == LINE_ALL:
                    label = ", ".join(lines)
                elif len(group) == 1:
                    fams = sorted({line_group_of(l) for l in lines})
                    label = f"{group} ({'+'.join(fams)})"
                else:
                    label = group
                out[el] = {
                    "concentration": conc,
                    "error": err,
                    "lines": lines,
                    "line": label,
                    "method": "standards_overlap",
                    "intensity_cps": cps_map[el],
                    "intensity_cps_err": 0.0,
                    "in_range": True,
                    "overlap": "+".join(model.elements),
                }

        for sym, curve in self.curves.items():
            if sym in overlap_done:
                continue
            if not (curve.fitted and curve.enabled):
                continue
            spot = _spot_cps(sym)
            if spot is None:
                continue
            cps, cps_err, lines, group = spot
            conc, err = curve.predict(cps, cps_err)
            if group == LINE_ALL:
                label = ", ".join(lines)
            elif len(group) == 1:
                fams = sorted({line_group_of(l) for l in lines})
                label = f"{group} ({'+'.join(fams)})"
            else:
                label = group
            out[sym] = {
                "concentration": conc,
                "error": err,
                "lines": lines,
                "line": label,
                "method": "standards_curve",
                "intensity_cps": cps,
                "intensity_cps_err": cps_err,
                "in_range": _in_range(curve, cps),
            }
        return out

    # ---- summaries ---------------------------------------------------------- #
    def fitted_curves(self) -> List[ElementCurve]:
        return [c for c in self.curves.values() if c.fitted and c.enabled]

    def summary_lines(self) -> List[str]:
        lines = []
        for c in self.curves.values():
            if c.fitted:
                flag = "" if c.enabled else " (disabled)"
                lines.append(
                    f"{c.element:>2} {c.line_group:<3} n={c.n_standards} "
                    f"R²={c.r_squared:.4f} RMSE={c.rmse:.3g} wt% "
                    f"RSD={c.mean_rsd_percent:.1f}% "
                    f"MDC={c.minimum_detectable_concentration() or float('nan'):.3g} wt%{flag}"
                )
            else:
                lines.append(f"{c.element:>2} — {c.message}")
        return lines

    def export_csv(self, path: str) -> None:
        """Write curves and per-standard points to a CSV for reports."""
        import csv

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["# Element calibration curves", self.calibration_date or ""])
            w.writerow([
                "element", "line", "model", "enabled", "fitted", "n_standards",
                "n_spectra", "c0", "c0_err", "c1", "c1_err", "c2", "c2_err",
                "r_squared", "rmse_wt_pct", "mean_replicate_rsd_pct",
                "mdc_wt_pct", "message",
            ])
            for c in self.curves.values():
                co, ce = c.coefficients + [0.0] * 3, c.coefficient_errors + [0.0] * 3
                mdc = c.minimum_detectable_concentration() if c.fitted else None
                w.writerow([
                    c.element, c.line_group, c.model, c.enabled, c.fitted,
                    c.n_standards, c.n_spectra,
                    co[0], ce[0], co[1], ce[1], co[2], ce[2],
                    c.r_squared, c.rmse, c.mean_rsd_percent,
                    "" if mdc is None else mdc, c.message,
                ])
            w.writerow([])
            w.writerow(["# Calibration points (one per standard per element)"])
            w.writerow([
                "element", "standard", "included", "certified_wt_pct",
                "mean_cps", "sd_cps", "sem_cps", "rsd_pct", "n_spots",
                "predicted_wt_pct", "residual_wt_pct",
            ])
            for c in self.curves.values():
                for p in c.points:
                    w.writerow([
                        c.element, p.standard, p.included, p.concentration,
                        p.intensity, p.intensity_sd, p.intensity_sem,
                        p.rsd_percent, p.n_spots,
                        "" if p.predicted is None else p.predicted,
                        "" if p.residual is None else p.residual,
                    ])
            w.writerow([])
            w.writerow(["# Individual spot intensities"])
            w.writerow(["element", "standard", "spectrum", "live_time_s", "net_area", "area_err", "cps", "cps_err", "lines"])
            for c in self.curves.values():
                for p in c.points:
                    for s in p.spots:
                        w.writerow([
                            c.element, p.standard, s.spectrum, s.live_time,
                            s.area, s.area_err, s.cps, s.cps_err, " ".join(s.lines),
                        ])

    # ---- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": CALIBRATION_TYPE,
            "version": 1,
            "calibration_date": self.calibration_date or datetime.now().isoformat(),
            "message": self.message,
            "settings": dict(self.settings),
            "fit_settings": dict(self.fit_settings),
            "line_groups": dict(self.line_groups),
            "excluded_points": {k: list(v) for k, v in self.excluded_points.items()},
            "fwhm_calibration": self.fwhm_calibration,
            "overlap_models": [
                m.to_dict() if hasattr(m, "to_dict") else m
                for m in (self.overlap_models or [])
            ],
            "standards": {k: v.to_dict() for k, v in self.standards.items()},
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
            "intensities": {
                std: {path: {el: s.to_dict() for el, s in spots.items()}
                      for path, spots in per_path.items()}
                for std, per_path in self.intensities.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardsCalibration":
        cal = cls(
            standards={k: StandardRecord.from_dict(v) for k, v in (data.get("standards") or {}).items()},
            curves={k: ElementCurve.from_dict(v) for k, v in (data.get("curves") or {}).items()},
            intensities={
                std: {path: {el: SpotIntensity.from_dict(s) for el, s in spots.items()}
                      for path, spots in per_path.items()}
                for std, per_path in (data.get("intensities") or {}).items()
            },
            line_groups={str(k): str(v) for k, v in (data.get("line_groups") or {}).items()},
            excluded_points={str(k): [str(x) for x in v] for k, v in (data.get("excluded_points") or {}).items()},
            settings=dict(data.get("settings") or {}),
            fit_settings=dict(data.get("fit_settings") or {}),
            calibration_date=data.get("calibration_date"),
            fwhm_calibration=data.get("fwhm_calibration"),
            message=str(data.get("message", "") or ""),
        )
        from core.overlap_deconvolution import OverlapMixingModel
        cal.overlap_models = [
            OverlapMixingModel.from_dict(m) for m in (data.get("overlap_models") or [])
            if isinstance(m, dict)
        ]
        return cal

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "StandardsCalibration":
        with open(filepath) as fh:
            return cls.from_dict(json.load(fh))


def is_standards_calibration_dict(data: Dict[str, Any]) -> bool:
    return isinstance(data, dict) and data.get("type") == CALIBRATION_TYPE


def load_any_standards_calibration(data: Dict[str, Any]):
    """
    Build the right object from a saved dict: the new curve-based
    StandardsCalibration or the legacy CalibrationResult.
    """
    if is_standards_calibration_dict(data):
        return StandardsCalibration.from_dict(data)
    from core.calibration import CalibrationResult

    return CalibrationResult.from_dict(data)


def _in_range(curve: ElementCurve, cps: float) -> bool:
    pts = curve.included_points()
    if not pts:
        return True
    lo = min(p.intensity for p in pts)
    hi = max(p.intensity for p in pts)
    span = hi - lo
    return (lo - 0.1 * span) <= cps <= (hi + 0.1 * span)


def _z_sort_key(sym: str) -> int:
    from core.advanced_peak_fitting import get_element_z

    return get_element_z(sym) or 999
