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


def observed_areas_from_peaks(
    peaks,
    *,
    tube_element=None,
    sample_contains_tube_element=False,
) -> Dict[str, Tuple[float, str, float]]:
    """
    Strongest non-tube sample peak per element.

    Returns:
        {element: (area, line, energy)}
    """
    anode = tube_element
    include_anode = bool(sample_contains_tube_element)
    best: Dict[str, Tuple[float, str, float]] = {}
    for peak in peaks or []:
        if getattr(peak, "is_tube_line", False):
            continue
        element = getattr(peak, "element", None)
        if not element or element in _LIGHT:
            continue
        line = getattr(peak, "line", None) or "Kα1"
        if str(line).startswith("Compton"):
            continue
        if anode and element == anode and not include_anode:
            continue
        area = float(getattr(peak, "area", 0.0) or 0.0)
        if area <= 0:
            continue
        energy = float(getattr(peak, "energy", 0.0) or 0.0)
        prev = best.get(element)
        if prev is None or area > prev[0]:
            best[element] = (area, str(line), energy)
    return best


def _predicted_intensity(
    fp: FundamentalParameters,
    element: str,
    line: str,
    composition_frac: Dict[str, float],
) -> float:
    z = _atomic_number(element)
    if z is None:
        return 0.0
    conc = float(composition_frac.get(element, 0.0))
    if conc <= 0:
        return 0.0
    return float(
        fp.calculate_intensity(
            element, z, normalize_line(line), conc, composition_frac
        )
    )


def _ratio_residual(
    observed: Dict[str, Tuple[float, str, float]],
    predicted: Dict[str, float],
) -> float:
    els = [el for el in observed if predicted.get(el, 0.0) > 0]
    if len(els) < 1:
        return float("inf")
    obs = np.array([observed[el][0] for el in els], dtype=float)
    pred = np.array([predicted[el] for el in els], dtype=float)
    if obs.sum() <= 0 or pred.sum() <= 0:
        return float("inf")
    obs_n = obs / obs.sum()
    pred_n = pred / pred.sum()
    return float(np.sqrt(np.mean((obs_n - pred_n) ** 2)))


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
        )


def _concentrations_dict(
    element_wt: Dict[str, float],
    observed: Dict[str, Tuple[float, str, float]],
    assumptions: MatrixAssumptions,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    measured = set(observed)
    # Measured cations first, then light elements, then anything else
    order = [el for el in element_wt if el in measured]
    order += [el for el in ("O", "C", "H") if el in element_wt]
    order += [el for el in element_wt if el not in order]
    for el in order:
        wt = float(element_wt[el])
        if el in observed:
            area, line, _energy = observed[el]
            role = "measured"
            lines = [line]
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
        }
    return out


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

    Args:
        peaks: Fitted Peak objects (tube lines ignored)
        assumptions: Matrix kind + H2O/OH/CO2 knobs
        experimental_params: excitation_energy, incident_angle, takeoff_angle
        max_iter: FP iteration cap
        damp: mixing factor toward the new cation estimate (0-1)
        tol: max relative cation change for convergence
        tube_element: Anode symbol excluded unless sample_contains_tube_element
        sample_contains_tube_element: Opt-in to quantify the tube anode
    """
    assumptions = assumptions or MatrixAssumptions()
    params = experimental_params or {}
    observed = observed_areas_from_peaks(
        peaks,
        tube_element=tube_element or params.get("tube_element"),
        sample_contains_tube_element=sample_contains_tube_element
        or bool(params.get("sample_contains_tube_element")),
    )
    if not observed:
        return FPQuantResult(
            success=False,
            message="No labeled sample peaks with area > 0.",
            assumptions=assumptions,
        )

    excitation = float(params.get("excitation_energy", 50.0) or 50.0)
    incident = float(params.get("incident_angle", 45.0) or 45.0)
    takeoff = float(params.get("takeoff_angle", incident) or incident)
    fp = FundamentalParameters(
        excitation_energy=excitation,
        takeoff_angle=takeoff,
        incident_angle=incident,
    )

    cation_masses = {el: area for el, (area, _line, _e) in observed.items()}
    try:
        element_wt, formula_wt = expand_composition(cation_masses, assumptions)
    except ValueError as exc:
        return FPQuantResult(
            success=False, message=str(exc), assumptions=assumptions
        )

    last_residual = float("inf")
    predicted: Dict[str, float] = {}
    n_iter = 0
    damp = min(max(float(damp), 0.1), 1.0)

    for n_iter in range(1, max_iter + 1):
        frac = {k: v / 100.0 for k, v in element_wt.items()}
        new_cations: Dict[str, float] = {}
        predicted = {}
        for el, (area, line, _energy) in observed.items():
            i_th = _predicted_intensity(fp, el, line, frac)
            predicted[el] = i_th
            c = max(frac.get(el, 0.0), 1e-12)
            if i_th <= 0:
                new_cations[el] = cation_masses.get(el, 1e-12)
            else:
                new_cations[el] = area * c / i_th

        blended = {
            el: (1.0 - damp) * cation_masses.get(el, 0.0) + damp * new_cations[el]
            for el in new_cations
        }
        # Keep relative scale of cations order-1 so expand stays well-conditioned
        s = sum(blended.values()) or 1.0
        blended = {el: 100.0 * v / s for el, v in blended.items()}

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

    lines_used = {el: line for el, (_a, line, _e) in observed.items()}
    concentrations = _concentrations_dict(element_wt, observed, assumptions)
    cation_pct = measured_cation_wt(element_wt)
    return FPQuantResult(
        success=True,
        element_wt=element_wt,
        formula_wt=formula_wt,
        concentrations=concentrations,
        iterations=n_iter,
        residual=last_residual,
        message="ok",
        lines_used=lines_used,
        assumptions=assumptions,
        measured_cation_pct=cation_pct,
    )


def format_formula_summary(formula_wt: Dict[str, float], max_terms: int = 10) -> str:
    return format_formula_wt(formula_wt, max_terms=max_terms)
