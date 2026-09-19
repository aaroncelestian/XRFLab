"""
Standardless fundamental-parameters quantification from one fitted spectrum.

Iterates cation ratios so predicted line intensities match fitted peak areas,
while a MatrixAssumptions object supplies stoichiometric O/C/H plus fixed
user H2O / OH / CO2. Instrument flux is unknown, so only intensity ratios
are used (Sherman-style relative FP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.fundamental_parameters import FundamentalParameters
from core.matrix_model import (
    MatrixAssumptions,
    expand_composition,
    format_formula_wt,
    measured_cation_wt,
)

_LIGHT = frozenset({"H", "C", "O", "N", "F"})

_LINE_ALIASES = {
    "k": "Kα1",
    "ka": "Kα1",
    "ka1": "Kα1",
    "ka2": "Kα2",
    "kb": "Kβ1",
    "kb1": "Kβ1",
    "kb2": "Kβ2",
    "kb3": "Kβ3",
    "l": "Lα1",
    "la": "Lα1",
    "la1": "Lα1",
    "la2": "Lα2",
    "lb": "Lβ1",
    "lb1": "Lβ1",
    "lb2": "Lβ2",
    "lg": "Lγ1",
    "lg1": "Lγ1",
    "m": "Mα1",
    "ma": "Mα1",
    "ma1": "Mα1",
    "ma2": "Mα2",
}


def normalize_line(name: Optional[str]) -> str:
    """Map fitted line labels onto FundamentalParameters line names."""
    if not name:
        return "Kα1"
    raw = str(name).strip()
    key = (
        raw.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("α", "a")
        .replace("β", "b")
        .replace("γ", "g")
        .replace("α", "a")
    )
    return _LINE_ALIASES.get(key, raw)


def _atomic_number(symbol: str) -> Optional[int]:
    try:
        import xraylib as xrl

        return int(xrl.SymbolToAtomicNumber(symbol))
    except Exception:
        return None


_XRL_LINE_CODES = None


def tabulated_line_energy(element: str, line: str) -> Optional[float]:
    """Tabulated emission energy (keV) for a normalized line name, or None."""
    global _XRL_LINE_CODES
    try:
        import xraylib as xrl

        if _XRL_LINE_CODES is None:
            _XRL_LINE_CODES = {
                "Kα1": xrl.KA1_LINE, "Kα2": xrl.KA2_LINE,
                "Kβ1": xrl.KB1_LINE, "Kβ2": xrl.KB2_LINE, "Kβ3": xrl.KB3_LINE,
                "Lα1": xrl.LA1_LINE, "Lα2": xrl.LA2_LINE,
                "Lβ1": xrl.LB1_LINE, "Lβ2": xrl.LB2_LINE, "Lγ1": xrl.LG1_LINE,
                "Mα1": xrl.MA1_LINE, "Mα2": xrl.MA2_LINE,
            }
        code = _XRL_LINE_CODES.get(normalize_line(line))
        if code is None:
            return None
        z = _atomic_number(element)
        if z is None:
            return None
        e = float(xrl.LineEnergy(z, code))
        return e if e > 0 else None
    except Exception:
        return None


def _line_series_rank(line: Optional[str]) -> int:
    """K before L before M, even when a lower series has a larger fitted area."""
    raw = str(line or "")
    key = (
        raw.lower()
        .replace("α", "a")
        .replace("β", "b")
        .replace("γ", "g")
        .replace("α", "a")
    )
    if key.startswith("k"):
        return 0
    if key.startswith("l"):
        return 1
    if key.startswith("m"):
        return 2
    return 3


# Relative model uncertainty of the FP prediction per series. Within a K
# series the branching ratios are known to a few %; the L model here treats
# every L line as L3 with an averaged yield, so L predictions are much
# rougher; M is a placeholder yield. These enter the per-line weights as
# σ² = N + (f·N)² with N the *predicted* counts.
SERIES_MODEL_UNCERTAINTY = {0: 0.03, 1: 0.15, 2: 0.50, 3: 1.0}

# |obs/pred − 1| beyond this (relative to the element's pooled estimate)
# is reported as a line-consistency warning (overlap / absorption edge).
LINE_CHECK_TOLERANCE = 0.25
# Robust IRLS: lines deviating by more than this are down-weighted so one
# contaminated line cannot drag the pooled estimate (Huber-style).
ROBUST_TOLERANCE = 0.5

# Lines below this energy are fully absorbed / outside the detector's
# useful efficiency range; FP predictions there are meaningless.
MIN_LINE_ENERGY_KEV = 1.0
# Lines whose predicted intensity is below this fraction of the element's
# strongest predicted line are too weak to carry information (fitted area
# is mostly noise / background residual).
MIN_PREDICTED_SHARE = 0.03
# Sub-lines closer than this × detector FWHM are unresolved: the fitter's
# split between them is arbitrary, so they are pooled into one observation.
UNRESOLVED_FWHM_FRACTION = 0.6


def _line_family(line: str) -> str:
    """'Kα1' → 'Kα', 'Kβ3' → 'Kβ', 'Lγ1' → 'Lγ'."""
    s = str(line)
    return s.rstrip("0123456789") or s


@dataclass
class LineObservation:
    """
    One fitted emission-line observation of a sample element.

    `lines` holds the tabulated lines pooled into this observation (e.g.
    ['Kα1', 'Kα2'] when the detector cannot resolve them). The FP
    prediction is the sum over `lines`.
    """

    lines: list
    area: float
    energy: float

    @property
    def line(self) -> str:
        """Display label: 'Kα' for Kα1+Kα2, else the lines joined."""
        fams = {_line_family(l) for l in self.lines}
        if len(self.lines) == 1:
            return self.lines[0]
        if len(fams) == 1:
            return fams.pop()
        return "+".join(self.lines)

    @property
    def series_rank(self) -> int:
        return min(_line_series_rank(l) for l in self.lines)

    @property
    def model_uncertainty(self) -> float:
        return SERIES_MODEL_UNCERTAINTY.get(self.series_rank, 1.0)


def _detector_fwhm(energy_kev: float) -> float:
    try:
        from core.peak_fitting import PeakFitter

        return float(PeakFitter.calculate_fwhm(float(energy_kev)))
    except Exception:
        return 0.15


def _pool_unresolved(obs: list) -> list:
    """Merge same-series lines closer than a fraction of the detector FWHM."""
    obs = sorted(obs, key=lambda o: o.energy)
    pooled: list = []
    for o in obs:
        if pooled:
            prev = pooled[-1]
            if (
                prev.series_rank == o.series_rank
                and abs(o.energy - prev.energy)
                < UNRESOLVED_FWHM_FRACTION * _detector_fwhm(0.5 * (o.energy + prev.energy))
            ):
                tot = prev.area + o.area
                prev.energy = (
                    (prev.energy * prev.area + o.energy * o.area) / tot
                    if tot > 0 else 0.5 * (prev.energy + o.energy)
                )
                prev.area = tot
                prev.lines = prev.lines + o.lines
                continue
        pooled.append(LineObservation(list(o.lines), o.area, o.energy))
    return pooled


def observed_lines_from_peaks(
    peaks,
    *,
    tube_element=None,
    sample_contains_tube_element=False,
    fp: Optional[FundamentalParameters] = None,
    min_energy_kev: float = MIN_LINE_ENERGY_KEV,
) -> Dict[str, list]:
    """
    All usable non-tube sample lines per element.

    Every K and L line the FP model can predict is kept as an independent
    observation (K and L series of one element are *not* tied to each other
    — their ratio depends on the excitation spectrum). Unresolved sub-lines
    (Kα1/Kα2, Kβ1/Kβ3, Lα1/Lα2 …) are pooled. Lines below `min_energy_kev`
    are dropped. M lines are only used when an element has nothing else.

    Returns:
        {element: [LineObservation, ...]} sorted K → L → M, strongest first
    """
    anode = tube_element
    include_anode = bool(sample_contains_tube_element)
    by_el: Dict[str, Dict[str, LineObservation]] = {}
    for peak in peaks or []:
        if getattr(peak, "is_tube_line", False):
            continue
        element = getattr(peak, "element", None)
        if not element or element in _LIGHT:
            continue
        line = str(getattr(peak, "line", None) or "Kα1")
        if line.startswith("Compton"):
            continue
        if anode and element == anode and not include_anode:
            continue
        area = float(getattr(peak, "area", 0.0) or 0.0)
        if area <= 0:
            continue
        norm = normalize_line(line)
        energy = float(getattr(peak, "energy", 0.0) or 0.0)
        if energy <= 0:
            # Stored areas without a fitted centre (batch summaries)
            energy = tabulated_line_energy(element, norm) or 0.0
        if energy < float(min_energy_kev):
            continue
        if fp is not None:
            z = _atomic_number(element)
            if z is None or fp._get_line_energy(z, norm) is None:
                continue
        bucket = by_el.setdefault(element, {})
        prev = bucket.get(norm)
        if prev is None:
            bucket[norm] = LineObservation([norm], area, energy)
        else:
            prev.area += area

    out: Dict[str, list] = {}
    for el, bucket in by_el.items():
        obs = _pool_unresolved(list(bucket.values()))
        obs.sort(key=lambda o: (o.series_rank, -o.area))
        if obs[0].series_rank < 2:
            # Drop M lines when K or L is available
            obs = [o for o in obs if o.series_rank < 2]
        out[el] = obs
    return out


def observed_areas_from_peaks(
    peaks,
    *,
    tube_element=None,
    sample_contains_tube_element=False,
) -> Dict[str, Tuple[float, str, float]]:
    """
    Best single non-tube sample peak per element (legacy helper).

    Prefers K lines over L/M even if the lower-energy peak has a larger
    fitted area. Within a series, the largest area wins.

    Returns:
        {element: (area, line, energy)}
    """
    lines = observed_lines_from_peaks(
        peaks,
        tube_element=tube_element,
        sample_contains_tube_element=sample_contains_tube_element,
    )
    return {el: (obs[0].area, obs[0].line, obs[0].energy) for el, obs in lines.items()}


def _usable(obs: list, predicted: Dict[str, float]) -> list:
    """Lines with a prediction ≥ MIN_PREDICTED_SHARE of the element's max."""
    preds = [float(predicted.get(o.line, 0.0)) for o in obs]
    top = max(preds) if preds else 0.0
    if top <= 0:
        return []
    return [o for o, p in zip(obs, preds) if p >= MIN_PREDICTED_SHARE * top]


def _pooled_scale(obs: list, predicted: Dict[str, float]) -> Optional[float]:
    """
    Robust weighted least-squares slope of fitted area vs. FP-predicted
    intensity through the origin over an element's lines: s = Σ w A I / Σ w I².

    Weights use the *predicted* counts N = s·I (Poisson) plus a per-series
    model term, σ² = N + (f·N)², iterated (IRLS). Lines whose obs/pred
    deviates by more than ROBUST_TOLERANCE are Huber-down-weighted so one
    contaminated line cannot drag the estimate.

    For a single line this reduces to A / I. Returns None if nothing usable.
    """
    use = _usable(obs, predicted)
    if not use:
        return None
    a = np.array([o.area for o in use], dtype=float)
    i_th = np.array([float(predicted[o.line]) for o in use], dtype=float)
    f = np.array([o.model_uncertainty for o in use], dtype=float)

    s = float(a.sum() / i_th.sum())
    if s <= 0:
        return None
    for _ in range(6):
        n_pred = np.maximum(s * i_th, 1.0)
        var = n_pred + (f * n_pred) ** 2
        w = 1.0 / var
        if len(use) > 1:
            dev = np.abs(a / n_pred - 1.0)
            robust = np.where(dev > ROBUST_TOLERANCE, (ROBUST_TOLERANCE / dev) ** 2, 1.0)
            w = w * robust
        den = float(np.sum(w * i_th * i_th))
        if den <= 0:
            break
        s_new = float(np.sum(w * a * i_th) / den)
        if s_new <= 0:
            break
        converged = abs(s_new - s) <= 1e-4 * s
        s = s_new
        if converged:
            break
    return s


def _line_checks(obs: list, predicted: Dict[str, float], scale: Optional[float]) -> list:
    """Per-line obs/pred relative to the element's pooled scale."""
    checks = []
    usable = {id(o) for o in _usable(obs, predicted)}
    for o in obs:
        i_th = float(predicted.get(o.line, 0.0))
        pred_area = (scale or 0.0) * i_th
        used = id(o) in usable
        ratio = (o.area / pred_area) if (pred_area > 0 and used) else None
        checks.append({
            "line": o.line,
            "lines": list(o.lines),
            "energy": o.energy,
            "area": float(o.area),
            "predicted_area": float(pred_area),
            "ratio": None if ratio is None else float(ratio),
            "used": used,
            "flag": bool(ratio is not None and abs(ratio - 1.0) > LINE_CHECK_TOLERANCE),
        })
    return checks


def _predicted_intensity(
    fp: FundamentalParameters,
    element: str,
    line,
    composition_frac: Dict[str, float],
) -> float:
    """FP intensity for one line, or the sum over a list of pooled lines."""
    z = _atomic_number(element)
    if z is None:
        return 0.0
    conc = float(composition_frac.get(element, 0.0))
    if conc <= 0:
        return 0.0
    lines = list(line) if isinstance(line, (list, tuple)) else [line]
    return float(sum(
        fp.calculate_intensity(element, z, normalize_line(l), conc, composition_frac)
        for l in lines
    ))


def _ratio_residual(
    observed: Dict[str, list],
    predicted: Dict[str, Dict[str, float]],
) -> float:
    """RMS of normalized observed vs. predicted areas over all lines."""
    obs_list = []
    pred_list = []
    for el, obs in observed.items():
        for o in obs:
            i_th = float(predicted.get(el, {}).get(o.line, 0.0))
            if i_th <= 0:
                continue
            obs_list.append(o.area)
            pred_list.append(i_th)
    if not obs_list:
        return float("inf")
    obs_a = np.array(obs_list, dtype=float)
    pred_a = np.array(pred_list, dtype=float)
    if obs_a.sum() <= 0 or pred_a.sum() <= 0:
        return float("inf")
    obs_n = obs_a / obs_a.sum()
    pred_n = pred_a / pred_a.sum()
    return float(np.sqrt(np.mean((obs_n - pred_n) ** 2)))


def _as_percent(masses: Dict[str, float]) -> Dict[str, float]:
    """Normalize positive masses to wt% (sum 100)."""
    total = float(sum(max(float(v), 0.0) for v in masses.values()))
    if total <= 0:
        return {k: 0.0 for k in masses}
    return {k: 100.0 * max(float(v), 0.0) / total for k, v in masses.items()}


@dataclass
class FPQuantResult:
    """Closed elemental composition from iterative FP + matrix assumptions."""

    success: bool
    element_wt: Dict[str, float] = field(default_factory=dict)
    formula_wt: Dict[str, float] = field(default_factory=dict)
    concentrations: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    residual: float = float("inf")
    message: str = ""
    method: str = "fp_matrix"
    lines_used: Dict[str, str] = field(default_factory=dict)
    assumptions: Optional[MatrixAssumptions] = None
    measured_cation_pct: float = 0.0
    # Human-readable per-line consistency warnings (Kβ/Kα vs predicted …)
    line_warnings: list = field(default_factory=list)

    def formula_summary(self, max_terms: int = 8) -> str:
        return format_formula_wt(self.formula_wt, max_terms=max_terms)

    def to_dict(self) -> dict:
        residual = float(self.residual)
        return {
            "success": bool(self.success),
            "element_wt": dict(self.element_wt or {}),
            "formula_wt": dict(self.formula_wt or {}),
            "concentrations": dict(self.concentrations or {}),
            "iterations": int(self.iterations),
            "residual": residual if np.isfinite(residual) else None,
            "message": self.message,
            "method": self.method,
            "lines_used": dict(self.lines_used or {}),
            "assumptions": (
                self.assumptions.to_dict() if self.assumptions is not None else None
            ),
            "measured_cation_pct": float(self.measured_cation_pct),
            "line_warnings": list(self.line_warnings or []),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["FPQuantResult"]:
        if not data:
            return None
        residual = data.get("residual")
        if residual is None:
            residual = float("inf")
        assumptions = data.get("assumptions")
        if isinstance(assumptions, dict):
            assumptions = MatrixAssumptions.from_dict(assumptions)
        return cls(
            success=bool(data.get("success", False)),
            element_wt=dict(data.get("element_wt") or {}),
            formula_wt=dict(data.get("formula_wt") or {}),
            concentrations=dict(data.get("concentrations") or {}),
            iterations=int(data.get("iterations") or 0),
            residual=float(residual),
            message=str(data.get("message") or ""),
            method=str(data.get("method") or "fp_matrix"),
            lines_used=dict(data.get("lines_used") or {}),
            assumptions=assumptions,
            measured_cation_pct=float(data.get("measured_cation_pct") or 0.0),
            line_warnings=list(data.get("line_warnings") or []),
        )


def _concentrations_dict(
    element_wt: Dict[str, float],
    observed: Dict[str, list],
    assumptions: MatrixAssumptions,
    line_checks: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    measured = set(observed)
    line_checks = line_checks or {}
    # Measured cations first, then light elements, then anything else
    order = [el for el in element_wt if el in measured]
    order += [el for el in ("O", "C", "H") if el in element_wt]
    order += [el for el in element_wt if el not in order]
    for el in order:
        wt = float(element_wt[el])
        if el in observed:
            obs = observed[el]
            area = float(sum(o.area for o in obs))
            lines = [o.line for o in obs]
            line = "+".join(lines)
            role = "measured"
        else:
            area = 0.0
            line = "assumed"
            role = "assumed"
            lines = []
        out[el] = {
            "concentration": wt,
            "relative_intensity_pct": wt,
            "error": None,
            "lines": lines,
            "total_area": area,
            "method": "fp_matrix",
            "role": role,
            "line": line,
            "matrix": assumptions.kind.value,
            "line_checks": list(line_checks.get(el, [])),
        }
    return out


def _format_line_warning(el: str, check: dict) -> str:
    ratio = check.get("ratio")
    direction = "above" if ratio and ratio > 1 else "below"
    return (
        f"{el} {check['line']} ({check.get('energy', 0.0):.2f} keV): "
        f"{ratio:.2f}× the pooled FP prediction — "
        f"{'unresolved overlap or pile-up adding counts' if direction == 'above' else 'absorption edge or overlap stealing counts'}?"
    )


def quantify_from_peaks(
    peaks,
    assumptions: Optional[MatrixAssumptions] = None,
    experimental_params: Optional[Dict[str, Any]] = None,
    *,
    max_iter: int = 20,
    damp: float = 0.7,
    tol: float = 1e-4,
    tube_element=None,
    sample_contains_tube_element: bool = False,
) -> FPQuantResult:
    """
    Invert fitted peak areas to wt% using relative fundamental parameters.

    Every predictable K and L line of each element is used as an independent
    observation: the element's scale is the weighted least-squares slope of
    fitted area vs. FP-predicted intensity across its lines (weights from
    counting statistics plus a per-series model uncertainty). K and L
    series are not tied to each other. Per-line obs/pred ratios are
    reported so an overlap or absorption-edge effect on one line is visible.

    Args:
        peaks: Fitted Peak objects (tube lines ignored)
        assumptions: Matrix kind + H2O/OH/CO2 knobs
        experimental_params: excitation_energy (tube kV), incident_angle,
            takeoff_angle, tube_element. The kV sets a polychromatic
            continuum + anode characteristic spectrum, not a mono line.
        max_iter: FP iteration cap
        damp: mixing factor toward the new cation estimate (0-1)
        tol: max relative cation change for convergence
        tube_element: Anode symbol excluded unless sample_contains_tube_element
        sample_contains_tube_element: Opt-in to quantify the tube anode
    """
    assumptions = assumptions or MatrixAssumptions()
    params = experimental_params or {}
    anode = tube_element or params.get("tube_element")

    excitation = float(params.get("excitation_energy", 50.0) or 50.0)
    incident = float(params.get("incident_angle", 45.0) or 45.0)
    takeoff = float(params.get("takeoff_angle", incident) or incident)
    poly = params.get("polychromatic")
    fp = FundamentalParameters(
        excitation_energy=excitation,
        takeoff_angle=takeoff,
        incident_angle=incident,
        tube_element=str(anode or "Rh"),
        polychromatic=True if poly is None else bool(poly),
    )

    observed = observed_lines_from_peaks(
        peaks,
        tube_element=anode,
        sample_contains_tube_element=sample_contains_tube_element
        or bool(params.get("sample_contains_tube_element")),
        fp=fp,
    )
    if not observed:
        return FPQuantResult(
            success=False,
            message="No labeled sample peaks with area > 0.",
            assumptions=assumptions,
        )

    # Seed with the strongest line of each element
    cation_masses = _as_percent({el: obs[0].area for el, obs in observed.items()})
    try:
        element_wt, formula_wt = expand_composition(cation_masses, assumptions)
    except ValueError as exc:
        return FPQuantResult(
            success=False, message=str(exc), assumptions=assumptions
        )

    last_residual = float("inf")
    predicted: Dict[str, Dict[str, float]] = {}
    scales: Dict[str, Optional[float]] = {}
    n_iter = 0
    damp = min(max(float(damp), 0.1), 1.0)

    for n_iter in range(1, max_iter + 1):
        frac = {k: v / 100.0 for k, v in element_wt.items()}
        new_cations: Dict[str, float] = {}
        predicted = {}
        scales = {}
        for el, obs in observed.items():
            predicted[el] = {
                o.line: _predicted_intensity(fp, el, o.lines, frac) for o in obs
            }
            c = max(frac.get(el, 0.0), 1e-12)
            s = _pooled_scale(obs, predicted[el])
            scales[el] = s
            if s is None or s <= 0:
                new_cations[el] = cation_masses.get(el, 1e-12)
            else:
                # I(c) is (roughly) ∝ c, so the WLS slope A/I rescales c
                new_cations[el] = s * c

        # area * c / I is a weight fraction; mix only after both sides are wt%.
        new_cations = _as_percent(new_cations)
        blended = {
            el: (1.0 - damp) * cation_masses.get(el, 0.0) + damp * new_cations[el]
            for el in new_cations
        }
        blended = _as_percent(blended)

        rel_change = 0.0
        for el, new_m in blended.items():
            old_m = cation_masses.get(el, 0.0)
            denom = max(abs(new_m), abs(old_m), 1e-9)
            rel_change = max(rel_change, abs(new_m - old_m) / denom)

        cation_masses = blended
        try:
            element_wt, formula_wt = expand_composition(cation_masses, assumptions)
        except ValueError as exc:
            return FPQuantResult(
                success=False,
                message=str(exc),
                iterations=n_iter,
                assumptions=assumptions,
            )

        last_residual = _ratio_residual(observed, predicted)
        if rel_change < tol:
            break

    # Per-line consistency against the converged composition
    line_checks: Dict[str, list] = {}
    line_warnings: list = []
    for el, obs in observed.items():
        checks = _line_checks(obs, predicted.get(el, {}), scales.get(el))
        line_checks[el] = checks
        if sum(1 for c in checks if c["used"]) > 1:
            for chk in checks:
                if chk["flag"]:
                    line_warnings.append(_format_line_warning(el, chk))

    lines_used = {
        el: "+".join(c["line"] for c in line_checks[el] if c["used"]) or obs[0].line
        for el, obs in observed.items()
    }
    concentrations = _concentrations_dict(
        element_wt, observed, assumptions, line_checks
    )
    cation_pct = measured_cation_wt(element_wt)
    n_lines = sum(len(obs) for obs in observed.values())
    return FPQuantResult(
        success=True,
        element_wt=element_wt,
        formula_wt=formula_wt,
        concentrations=concentrations,
        iterations=n_iter,
        residual=last_residual,
        message=(
            f"ok ({fp.tube_element} {excitation:g} kV "
            f"{'polychromatic' if fp.polychromatic else 'monochromatic'}; "
            f"{n_lines} lines / {len(observed)} elements)"
        ),
        lines_used=lines_used,
        assumptions=assumptions,
        measured_cation_pct=cation_pct,
        line_warnings=line_warnings,
    )


def format_formula_summary(formula_wt: Dict[str, float], max_terms: int = 10) -> str:
    return format_formula_wt(formula_wt, max_terms=max_terms)
