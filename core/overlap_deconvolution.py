"""
Empirical overlap deconvolution for standards calibration.

Two pieces:

1. While fitting a *standard* the certified wt% are known. Overlapping
   line-groups (principal lines within ~1.25 FWHM) get a soft amplitude
   prior  A_i / A_j = (ε_i C_i) / (ε_j C_j). That is the only unique
   split when the design columns are nearly identical.

2. After all standards are fitted, each overlap cluster learns a mixing
   matrix  I = S C  from certified concentrations. Unknowns are unmixed
   with C = S⁻¹ I, which is more stable than a univariate curve through
   traded-off amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Groups whose strongest lines are closer than this × FWHM are one blend
OVERLAP_FWHM_FRAC = 1.25
# Soft prior: 1σ of (A_i − ρ A_j) relative to the expected split
OVERLAP_PRIOR_REL_SIGMA = 0.40
# Weight of the ratio row vs a unit-height spectral column
OVERLAP_PRIOR_WEIGHT = 2.0
# Mixing matrix is usable only when S is well conditioned
MAX_CONDITION = 80.0
MIN_WT_PCT = 1e-6


@dataclass
class OverlapPair:
    """Two line-groups of different elements that sit inside one FWHM."""

    key_a: str
    key_b: str
    element_a: str
    element_b: str
    energy_a: float
    energy_b: float
    separation_kev: float


@dataclass
class OverlapMixingModel:
    """I = S C + b  for one overlap cluster (element order is `elements`)."""

    elements: List[str]
    energies: List[float]
    sensitivities: List[List[float]]
    intercepts: List[float]
    condition: float
    n_standards: int
    r_squared: float
    usable: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": list(self.elements),
            "energies": [float(x) for x in self.energies],
            "sensitivities": [list(map(float, row)) for row in self.sensitivities],
            "intercepts": [float(x) for x in self.intercepts],
            "condition": float(self.condition),
            "n_standards": int(self.n_standards),
            "r_squared": float(self.r_squared),
            "usable": bool(self.usable),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverlapMixingModel":
        return cls(
            elements=[str(x) for x in data.get("elements") or []],
            energies=[float(x) for x in data.get("energies") or []],
            sensitivities=[list(map(float, row)) for row in data.get("sensitivities") or []],
            intercepts=[float(x) for x in data.get("intercepts") or []],
            condition=float(data.get("condition", 0.0) or 0.0),
            n_standards=int(data.get("n_standards", 0) or 0),
            r_squared=float(data.get("r_squared", 0.0) or 0.0),
            usable=bool(data.get("usable", False)),
            message=str(data.get("message", "") or ""),
        )

    def predict(
        self, intensities: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Return {element: (wt%, 1σ)} from observed cps."""
        if not self.usable or not self.elements:
            return {}
        s = np.asarray(self.sensitivities, dtype=float)
        b = np.asarray(self.intercepts, dtype=float)
        i_vec = np.array(
            [float(intensities.get(el, 0.0) or 0.0) for el in self.elements]
        )
        try:
            conc = np.linalg.solve(s, i_vec - b)
        except np.linalg.LinAlgError:
            conc, *_ = np.linalg.lstsq(s, i_vec - b, rcond=None)
        resid = i_vec - (s @ conc + b)
        # Rough σ: residual scatter mapped through S⁻¹
        try:
            sinv = np.linalg.inv(s)
            sigma_i = float(np.sqrt(np.mean(resid ** 2))) if resid.size else 0.0
            err = np.sqrt(np.clip(np.sum(sinv ** 2, axis=1), 0, None)) * max(sigma_i, 1e-9)
        except np.linalg.LinAlgError:
            err = np.full(len(self.elements), 0.2 * np.maximum(np.abs(conc), 1e-6))
        out = {}
        for el, c, e in zip(self.elements, conc, err):
            out[el] = (float(max(c, 0.0)), float(max(e, 0.0)))
        return out


def theoretical_sensitivity(
    element: str, energy_kev: float, excitation_kv: float = 50.0
) -> float:
    """Relative I/wt% proxy: tabulated line weight × overvoltage above the edge."""
    from core.advanced_peak_fitting import get_element_z
    from core.xray_data import edge_energy_kev, get_element_lines

    z = get_element_z(element)
    if not z:
        return 1.0
    try:
        lines = get_element_lines(element, z)
    except Exception:
        return 1.0
    closest = None
    closest_d = 1e9
    series = "L"
    for ser, entries in (lines or {}).items():
        for ln in entries or []:
            e = float(ln.get("energy") or 0.0)
            d = abs(e - float(energy_kev))
            if d < closest_d:
                closest, closest_d, series = ln, d, ser[:1].upper() or "L"
    if closest is None:
        return 1.0
    # Sensitivity of the *principal* line of that series (not a nearby satellite)
    family = (lines or {}).get(series) or [closest]
    best = max(
        family,
        key=lambda ln: float(
            ln.get("relative_intensity_series")
            or ln.get("relative_intensity")
            or ln.get("intensity")
            or 0.0
        ),
    )
    rel = float(
        best.get("relative_intensity_series")
        or best.get("relative_intensity")
        or best.get("intensity")
        or 1.0
    )
    edge = edge_energy_kev(z, series) or float(energy_kev)
    overvolt = max(float(excitation_kv) - float(edge), 0.1)
    return max(rel * overvolt, 1e-12)


def find_overlap_pairs(
    groups,
    fwhm_fn: Callable[[float], float],
    *,
    max_fwhm_frac: float = OVERLAP_FWHM_FRAC,
) -> List[OverlapPair]:
    """Pairs of *different-element* groups whose strongest lines blend."""
    items = []
    for g in groups or []:
        if not getattr(g, "lines", None):
            continue
        strongest = g.strongest
        items.append((g.key, g.element, float(strongest.energy)))
    pairs: List[OverlapPair] = []
    for i in range(len(items)):
        key_a, el_a, e_a = items[i]
        for j in range(i + 1, len(items)):
            key_b, el_b, e_b = items[j]
            if el_a == el_b:
                continue
            fwhm = 0.5 * (float(fwhm_fn(e_a)) + float(fwhm_fn(e_b)))
            sep = abs(e_a - e_b)
            if sep <= max_fwhm_frac * max(fwhm, 0.05):
                pairs.append(OverlapPair(
                    key_a=key_a, key_b=key_b,
                    element_a=el_a, element_b=el_b,
                    energy_a=e_a, energy_b=e_b,
                    separation_kev=sep,
                ))
    return pairs


def composition_ratio_priors(
    pairs: Sequence[OverlapPair],
    composition: Dict[str, float],
    *,
    excitation_kv: float = 50.0,
    rel_sigma: float = OVERLAP_PRIOR_REL_SIGMA,
) -> List[Tuple[str, str, float, float]]:
    """
    Soft rows  A_a − ρ A_b ≈ 0  with ρ = (ε_a C_a) / (ε_b C_b).

    Returns (key_a, key_b, rho, rel_sigma) for pairs where both certified
    concentrations are usable.
    """
    out: List[Tuple[str, str, float, float]] = []
    for p in pairs:
        c_a = float(composition.get(p.element_a) or 0.0)
        c_b = float(composition.get(p.element_b) or 0.0)
        if c_a < MIN_WT_PCT or c_b < MIN_WT_PCT:
            continue
        e_a = theoretical_sensitivity(p.element_a, p.energy_a, excitation_kv)
        e_b = theoretical_sensitivity(p.element_b, p.energy_b, excitation_kv)
        rho = (e_a * c_a) / (e_b * c_b)
        if not np.isfinite(rho) or rho <= 0:
            continue
        out.append((p.key_a, p.key_b, float(rho), float(rel_sigma)))
    return out


def ratio_prior_rows(
    labels: Sequence[str],
    priors: Sequence[Tuple[str, str, float, float]],
    n_cols: int,
    *,
    weight: float = OVERLAP_PRIOR_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Stack scale-free ratio constraints for the first NNLS solve."""
    index = {lab: i for i, lab in enumerate(labels)}
    rows = []
    notes = []
    for key_a, key_b, rho, rel_sigma in priors:
        ia = index.get(key_a)
        ib = index.get(key_b)
        if ia is None or ib is None:
            continue
        w = float(weight) / max(float(rel_sigma), 0.05)
        # A_a − ρ A_b = 0
        row = np.zeros(n_cols)
        row[ia] = w
        row[ib] = -w * float(rho)
        rows.append(row)
        notes.append(
            f"{key_a}/{key_b}: composition prior A≈{rho:.3g}× "
            f"(±{100 * rel_sigma:.0f}%)"
        )
    if not rows:
        return np.zeros((0, n_cols)), np.zeros(0), notes
    return np.vstack(rows), np.zeros(len(rows)), notes


def _clusters_from_pairs(pairs: Sequence[OverlapPair]) -> List[List[str]]:
    """Connected components of element symbols."""
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in pairs:
        union(p.element_a, p.element_b)
    buckets: Dict[str, List[str]] = {}
    for el in parent:
        buckets.setdefault(find(el), []).append(el)
    return [sorted(v) for v in buckets.values() if len(v) >= 2]


def principal_energy(element: str, line_group: str) -> float:
    """Energy of the strongest line in the element's chosen group."""
    from core.advanced_peak_fitting import get_element_z
    from core.xray_data import get_element_lines

    z = get_element_z(element)
    if not z:
        return 0.0
    try:
        lines = get_element_lines(element, z)
    except Exception:
        return 0.0
    group = (line_group or "L")[:1].upper()
    series = lines.get(group) or lines.get("L") or lines.get("K") or []
    if not series:
        return 0.0
    best = max(series, key=lambda ln: float(ln.get("relative_intensity") or ln.get("intensity") or 0.0))
    return float(best.get("energy") or 0.0)


def learn_mixing_models(
    clusters: Sequence[Sequence[str]],
    *,
    concentrations: Dict[str, Dict[str, float]],
    intensities: Dict[str, Dict[str, float]],
    energies: Dict[str, float],
    enabled: Optional[Iterable[str]] = None,
) -> List[OverlapMixingModel]:
    """
    Fit I = S C + b per cluster.

    concentrations / intensities: standard → {element: value}
    Only standards in `enabled` (or all) that have every cluster element
    certified and an intensity are used.
    """
    allow = set(enabled) if enabled is not None else None
    models: List[OverlapMixingModel] = []
    for cluster in clusters:
        els = list(cluster)
        rows_c = []
        rows_i = []
        used = []
        for std, concs in concentrations.items():
            if allow is not None and std not in allow:
                continue
            ints = intensities.get(std) or {}
            if any(float(concs.get(el) or 0.0) < 0 for el in els):
                continue
            if any(el not in concs for el in els):
                continue
            if any(el not in ints for el in els):
                continue
            rows_c.append([float(concs[el]) for el in els])
            rows_i.append([float(ints[el]) for el in els])
            used.append(std)
        c_mat = np.asarray(rows_c, dtype=float)
        i_mat = np.asarray(rows_i, dtype=float)
        n_std, n_el = (c_mat.shape if c_mat.size else (0, len(els)))
        e_list = [float(energies.get(el, 0.0)) for el in els]
        if n_std < n_el:
            models.append(OverlapMixingModel(
                elements=els, energies=e_list, sensitivities=[], intercepts=[],
                condition=0.0, n_standards=n_std, r_squared=0.0, usable=False,
                message=f"Need ≥{n_el} standards with all of {', '.join(els)}",
            ))
            continue
        use_intercept = n_std >= n_el + 2
        s = np.zeros((n_el, n_el))
        b = np.zeros(n_el)
        ss_res = 0.0
        ss_tot = 0.0
        for k in range(n_el):
            if use_intercept:
                x = np.column_stack([np.ones(n_std), c_mat])
            else:
                x = c_mat
            beta, *_ = np.linalg.lstsq(x, i_mat[:, k], rcond=None)
            pred = x @ beta
            if use_intercept:
                b[k] = float(beta[0])
                s[k, :] = beta[1:]
            else:
                s[k, :] = beta
            resid = i_mat[:, k] - pred
            ss_res += float(np.sum(resid ** 2))
            ss_tot += float(np.sum((i_mat[:, k] - np.mean(i_mat[:, k])) ** 2))
        try:
            cond = float(np.linalg.cond(s))
        except np.linalg.LinAlgError:
            cond = float("inf")
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        usable = bool(np.isfinite(cond) and cond < MAX_CONDITION and r2 > 0)
        msg = (
            f"Unmix { '+'.join(els) } from {n_std} standards "
            f"(cond={cond:.1f}, R²={r2:.2f})"
            if usable else
            f"Overlap { '+'.join(els) } not invertible "
            f"(cond={cond if np.isfinite(cond) else float('inf'):.1f}, R²={r2:.2f})"
        )
        models.append(OverlapMixingModel(
            elements=els, energies=e_list,
            sensitivities=s.tolist(), intercepts=b.tolist(),
            condition=cond if np.isfinite(cond) else 1e9,
            n_standards=n_std, r_squared=float(r2),
            usable=usable, message=msg,
        ))
    return models


def clusters_for_line_groups(
    line_groups: Dict[str, str],
    fwhm_fn: Callable[[float], float],
) -> Tuple[List[List[str]], List[OverlapPair], Dict[str, float]]:
    """Build element clusters from the calibration's chosen line groups."""
    from types import SimpleNamespace

    fake_groups = []
    energies = {}
    for el, group in (line_groups or {}).items():
        e = principal_energy(el, group)
        if e <= 0:
            continue
        energies[el] = e
        fake_groups.append(SimpleNamespace(
            key=f"{el} {group}",
            element=el,
            strongest=SimpleNamespace(energy=e),
            lines=[True],
        ))
    pairs = find_overlap_pairs(fake_groups, fwhm_fn)
    return _clusters_from_pairs(pairs), pairs, energies
