"""
REE-only fitting set: lanthanides (plus Y, Sc) and their line overlaps.

A full multi-element fit is not required to read wt% from the empirical
standards curves. The curves need a consistent net intensity for each
target line, and that intensity is only trustworthy if every interferent
that sits inside ~2 FWHM of a REE line is also in the fit.

This module builds that list from tabulated line energies (and optional
calibration overlap clusters). It does *not* close the overlap graph
through the interferents — adding Fe for Dy Lα must not pull in Zn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from core.advanced_peak_fitting import get_element_z
from core.smart_peak_id import COMMON_XRF_SYMBOLS
from core.standards_calibration import k_series_preferred
from core.xray_data import edge_energy_kev, get_element_lines

# Geochemical REE suite used by the MBH REE standards (no Pm).
REE_LANTHANIDES: Tuple[str, ...] = (
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
)
REE_Y: Tuple[str, ...] = ("Y",)
REE_SC: Tuple[str, ...] = ("Sc",)

# Principal α/β lines within this × FWHM (and this absolute cap) are partners.
# The cap stops high-energy K lines (FWHM ~0.3 keV) from matching everything.
REE_OVERLAP_FWHM_FRAC = 1.25
REE_OVERLAP_MAX_SEP_KEV = 0.30
MIN_LINE_ENERGY_KEV = 1.0
MIN_REL_SERIES = 0.10
# Series is “on” if the tube is at least this far above the edge.
SERIES_OVERVOLTAGE = 1.05

# Extra interferents that are common with REE L lines but not in the survey set.
_REE_INTERFERENT_EXTRA: Tuple[str, ...] = (
    "Cs", "I", "Te", "Hf", "Ta", "Th", "U",
)


@dataclass(frozen=True)
class OverlapHit:
    """One spectroscopic reason to include an interferent."""

    element: str
    target: str
    element_line: str
    target_line: str
    element_energy: float
    target_energy: float
    separation_kev: float
    source: str = "lines"  # 'lines' or 'calibration'


@dataclass
class ReeFitSet:
    """Elements to fit for a REE-only quantification."""

    rees: List[str]
    overlaps: List[str]
    hits: List[OverlapHit] = field(default_factory=list)
    excitation_kv: float = 50.0

    @property
    def symbols(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for sym in list(self.rees) + list(self.overlaps):
            if sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
        return out

    @property
    def ree_set(self) -> set:
        return set(self.rees)

    def hits_for(self, element: str) -> List[OverlapHit]:
        return [h for h in self.hits if h.element == element]

    def summary(self) -> str:
        n_ree = len(self.rees)
        n_ov = len(self.overlaps)
        if not self.overlaps:
            return f"{n_ree} REE(s), no spectroscopic overlaps"
        partners = ", ".join(self.overlaps)
        return f"{n_ree} REE(s) + {n_ov} overlap partner(s): {partners}"


def ree_elements(*, include_y: bool = True, include_sc: bool = False) -> List[str]:
    """Y + La–Lu (no Pm). Sc is optional — its Kα sits in a crowded L-line region."""
    symbols = list(REE_LANTHANIDES)
    if include_y:
        symbols.extend(REE_Y)
    if include_sc:
        symbols.extend(REE_SC)
    return _sort_by_z(symbols)


def is_ree(symbol: str, *, include_y: bool = True, include_sc: bool = False) -> bool:
    return str(symbol) in set(ree_elements(include_y=include_y, include_sc=include_sc))


def interferent_candidates() -> List[str]:
    """Non-REE elements whose lines are checked against the REE suite."""
    skip = set(ree_elements())
    skip.add("Pm")
    seen = set()
    out: List[str] = []
    for sym in list(COMMON_XRF_SYMBOLS) + list(_REE_INTERFERENT_EXTRA):
        if sym in skip or sym in seen:
            continue
        if not get_element_z(sym):
            continue
        seen.add(sym)
        out.append(sym)
    return _sort_by_z(out)


def _is_principal_line(name: str) -> bool:
    """Kα/Kβ and Lα/Lβ families — not Lγ, Ll, Lη, or M satellites."""
    if not name or "γ" in name or name in ("Ll", "Lη"):
        return False
    return ("α" in name) or ("β" in name)


def significant_lines(
    symbol: str,
    *,
    excitation_kv: float,
    series: Optional[str] = None,
    min_rel: float = MIN_REL_SERIES,
    min_energy: float = MIN_LINE_ENERGY_KEV,
    principal_only: bool = True,
) -> List[Tuple[str, float, str]]:
    """
    Return (line_name, energy_kev, series) for lines strong enough to overlap.

    If `series` is set, only that series is returned. Otherwise every series
    the tube can excite is included (so Ti K and Ba L are both visible).
    """
    z = get_element_z(symbol)
    if not z:
        return []
    try:
        lines = get_element_lines(symbol, z)
    except Exception:
        return []
    kv = float(excitation_kv)
    wanted = (series[:1].upper() if series else None)
    out: List[Tuple[str, float, str]] = []
    for ser, entries in (lines or {}).items():
        s = str(ser)[:1].upper()
        if wanted and s != wanted:
            continue
        if not wanted and not _series_excited(z, s, kv):
            continue
        for entry in entries or []:
            name = str(entry.get("name") or "")
            energy = float(entry.get("energy") or 0.0)
            rel = float(
                entry.get("relative_intensity_series")
                or entry.get("relative_intensity")
                or 0.0
            )
            if not name or energy < min_energy or rel < min_rel:
                continue
            if principal_only and not _is_principal_line(name):
                continue
            out.append((name, energy, s))
    return out


def preferred_series(symbol: str, excitation_kv: float) -> str:
    """Analytical series for a REE target (K if overvoltage allows, else L)."""
    return "K" if k_series_preferred(symbol, excitation_kv) else "L"


def find_ree_overlaps(
    targets: Sequence[str],
    *,
    excitation_kv: float = 50.0,
    fwhm_fn: Optional[Callable[[float], float]] = None,
    candidates: Optional[Sequence[str]] = None,
    extra_clusters: Optional[Iterable[Sequence[str]]] = None,
    max_fwhm_frac: float = REE_OVERLAP_FWHM_FRAC,
    max_sep_kev: float = REE_OVERLAP_MAX_SEP_KEV,
) -> List[OverlapHit]:
    """
    1-hop spectroscopic overlaps of the target lines, plus calibration clusters.

    Only lines of `targets` are used as the search seeds, so the graph does
    not walk through Fe → Co → Ni → Cu → Zn.
    """
    fwhm = fwhm_fn or _default_fwhm
    kv = float(excitation_kv)
    target_lines: Dict[str, List[Tuple[str, float, str]]] = {}
    for sym in targets:
        series = preferred_series(sym, kv)
        lines = significant_lines(sym, excitation_kv=kv, series=series)
        if not lines:
            lines = significant_lines(sym, excitation_kv=kv)
        target_lines[sym] = lines

    cand_list = list(candidates) if candidates is not None else interferent_candidates()
    hits: List[OverlapHit] = []
    seen = set()

    for cand in cand_list:
        if cand in target_lines:
            continue
        cand_lines = significant_lines(cand, excitation_kv=kv)
        if not cand_lines:
            continue
        for t_sym, t_lines in target_lines.items():
            for t_name, t_e, _t_s in t_lines:
                for c_name, c_e, _c_s in cand_lines:
                    sep = abs(c_e - t_e)
                    if sep > max_sep_kev:
                        continue
                    width = 0.5 * (float(fwhm(c_e)) + float(fwhm(t_e)))
                    if sep > max_fwhm_frac * max(width, 0.05):
                        continue
                    key = (cand, t_sym, c_name, t_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(OverlapHit(
                        element=cand,
                        target=t_sym,
                        element_line=c_name,
                        target_line=t_name,
                        element_energy=c_e,
                        target_energy=t_e,
                        separation_kev=sep,
                        source="lines",
                    ))

    for cluster in extra_clusters or []:
        members = [str(x) for x in cluster if x]
        if not members:
            continue
        if not any(m in target_lines for m in members):
            continue
        for m in members:
            if m in target_lines:
                continue
            if any(h.element == m and h.source == "calibration" for h in hits):
                continue
            # Attach to the first target in the cluster (display only)
            target = next((x for x in members if x in target_lines), members[0])
            hits.append(OverlapHit(
                element=m,
                target=target,
                element_line="cluster",
                target_line="cluster",
                element_energy=0.0,
                target_energy=0.0,
                separation_kev=0.0,
                source="calibration",
            ))
    return hits


def build_ree_fit_set(
    *,
    excitation_kv: float = 50.0,
    include_y: bool = True,
    include_sc: bool = False,
    fwhm_fn: Optional[Callable[[float], float]] = None,
    extra_clusters: Optional[Iterable[Sequence[str]]] = None,
) -> ReeFitSet:
    """REEs plus 1-hop overlap partners, sorted by Z."""
    rees = ree_elements(include_y=include_y, include_sc=include_sc)
    hits = find_ree_overlaps(
        rees,
        excitation_kv=excitation_kv,
        fwhm_fn=fwhm_fn,
        extra_clusters=extra_clusters,
    )
    overlaps = _sort_by_z({h.element for h in hits})
    return ReeFitSet(
        rees=rees,
        overlaps=overlaps,
        hits=hits,
        excitation_kv=float(excitation_kv),
    )


def apply_ree_roles(
    concentrations: Dict[str, Dict],
    fit_set: ReeFitSet,
) -> Dict[str, Dict]:
    """Tag rows as REE vs overlap and put REEs first (then partners by Z)."""
    if not concentrations:
        return {}
    ree_set = fit_set.ree_set
    tagged: Dict[str, Dict] = {}
    for sym, row in concentrations.items():
        data = dict(row)
        data["role"] = "ree" if sym in ree_set else "overlap"
        tagged[sym] = data
    rees = [s for s in fit_set.rees if s in tagged]
    others = _sort_by_z(s for s in tagged if s not in ree_set)
    return {s: tagged[s] for s in rees + others}


def clusters_from_calibration(calibration) -> List[List[str]]:
    """Overlap-model element lists from an applied StandardsCalibration."""
    out: List[List[str]] = []
    for model in getattr(calibration, "overlap_models", None) or []:
        els = [str(x) for x in (getattr(model, "elements", None) or []) if x]
        if len(els) >= 2:
            out.append(els)
    return out


def _series_excited(z: int, series: str, excitation_kv: float) -> bool:
    if series not in ("K", "L", "M"):
        return False
    edge = edge_energy_kev(z, series)
    if edge is None:
        return series in ("K", "L")
    return float(excitation_kv) >= float(edge) * SERIES_OVERVOLTAGE


def _default_fwhm(energy: float) -> float:
    from core.peak_fitting import PeakFitter
    return float(PeakFitter.calculate_fwhm(energy))


def _sort_by_z(symbols: Iterable[str]) -> List[str]:
    return sorted({str(s) for s in symbols if s}, key=lambda s: (get_element_z(s) or 999, s))
