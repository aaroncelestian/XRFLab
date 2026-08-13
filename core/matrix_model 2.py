"""
Matrix assumptions for standardless XRF composition.

Measured cations come from the spectrum. Unmeasurable light elements
(H, C, O) are added from a stoichiometric model plus optional user knobs
for H2O, OH, and CO2. Those knobs are fixed during FP iteration.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

try:
    import xraylib as xrl

    _HAS_XRAYLIB = True
except ImportError:
    xrl = None
    _HAS_XRAYLIB = False


class MatrixKind(str, Enum):
    """How to assign unmeasured light elements to measured cations."""

    MEASURED = "measured"  # metal / sulfide; galena PbS needs nothing extra
    OXIDE = "oxide"  # silicate / oxide: stoichiometric oxygen
    CARBONATE = "carbonate"  # CO3 on carbonate formers, oxides otherwise
    HYDROXIDE = "hydroxide"  # OH-bearing formulas; Si etc. stay oxides


# Oxide formulas (cation → compound). Fe is overridden by fe_as.
_OXIDE_FORMULAS = {
    "Si": "SiO2",
    "Al": "Al2O3",
    "Fe": "FeO",
    "Mg": "MgO",
    "Ca": "CaO",
    "Na": "Na2O",
    "K": "K2O",
    "Ti": "TiO2",
    "Mn": "MnO",
    "P": "P2O5",
    "Cr": "Cr2O3",
    "Ni": "NiO",
    "Cu": "CuO",
    "Zn": "ZnO",
    "Sr": "SrO",
    "Ba": "BaO",
    "Zr": "ZrO2",
    "S": "SO3",
    "Pb": "PbO",
    "V": "V2O5",
    "Co": "CoO",
    "Ga": "Ga2O3",
    "As": "As2O5",
    "Rb": "Rb2O",
    "Y": "Y2O3",
    "Nb": "Nb2O5",
    "Mo": "MoO3",
    "Sn": "SnO2",
    "Sb": "Sb2O5",
    "Cs": "Cs2O",
    "La": "La2O3",
    "Ce": "CeO2",
    "Nd": "Nd2O3",
    "W": "WO3",
    "Th": "ThO2",
    "U": "UO2",
}

_CARBONATE_FORMULAS = {
    "Ca": "CaCO3",
    "Mg": "MgCO3",
    "Fe": "FeCO3",
    "Mn": "MnCO3",
    "Sr": "SrCO3",
    "Ba": "BaCO3",
    "Zn": "ZnCO3",
    "Pb": "PbCO3",
    "Na": "Na2CO3",
    "K": "K2CO3",
    "Cd": "CdCO3",
}

_HYDROXIDE_FORMULAS = {
    "Al": "Al(OH)3",
    "Fe": "FeOOH",
    "Mg": "Mg(OH)2",
    "Ca": "Ca(OH)2",
    "Na": "NaOH",
    "K": "KOH",
    "Mn": "MnOOH",
    "Ni": "Ni(OH)2",
    "Co": "Co(OH)2",
    "Cu": "Cu(OH)2",
    "Zn": "Zn(OH)2",
    "Cr": "Cr(OH)3",
    "Cd": "Cd(OH)2",
}

_FALLBACK_AW = {
    "H": 1.00794,
    "C": 12.0107,
    "O": 15.9994,
    "Na": 22.9897,
    "Mg": 24.305,
    "Al": 26.9815,
    "Si": 28.0855,
    "P": 30.9738,
    "S": 32.065,
    "K": 39.0983,
    "Ca": 40.078,
    "Ti": 47.867,
    "Cr": 51.9961,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Sr": 87.62,
    "Zr": 91.224,
    "Ba": 137.327,
    "Pb": 207.2,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|\d+)")
_LIGHT = frozenset({"H", "C", "O", "N", "F"})


@dataclass
class MatrixAssumptions:
    """User-controlled matrix model and optional light-element knobs (wt%)."""

    kind: MatrixKind = MatrixKind.MEASURED
    fe_as: str = "FeO"
    h2o_wt: float = 0.0
    oh_wt: float = 0.0
    co2_wt: float = 0.0

    def light_sum(self) -> float:
        return max(0.0, self.h2o_wt) + max(0.0, self.oh_wt) + max(0.0, self.co2_wt)

    def hint(self) -> str:
        if self.kind == MatrixKind.MEASURED:
            return (
                "Closes on measured elements only (galena PbS, metals, sulfides). "
                "Leave H2O/OH/CO2 at 0 unless the sample is hydrated or mixed."
            )
        if self.kind == MatrixKind.OXIDE:
            return (
                "Adds stoichiometric oxygen (SiO2, Al2O3, FeO/Fe2O3, …). "
                "Set H2O for hydrates (e.g. gypsum ~21% H2O)."
            )
        if self.kind == MatrixKind.CARBONATE:
            return (
                "Adds CO3 to Ca, Mg, Fe, Mn, …; other cations as oxides. "
                "Extra CO2/H2O knobs are in addition to that stoichiometry."
            )
        return (
            "Uses hydroxide formulas (FeOOH, Al(OH)3, …); Si stays SiO2. "
            "Tune extra OH and H2O to match a hydrated phase."
        )


def atomic_weight(symbol: str) -> float:
    """Atomic weight for an element symbol."""
    if _HAS_XRAYLIB:
        try:
            z = xrl.SymbolToAtomicNumber(symbol)
            return float(xrl.AtomicWeight(z))
        except Exception:
            pass
    if symbol in _FALLBACK_AW:
        return _FALLBACK_AW[symbol]
    raise KeyError(f"Unknown element: {symbol}")


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a simple chemical formula into element counts. Supports (OH)3."""
    tokens = _FORMULA_TOKEN.findall((formula or "").replace(" ", ""))
    if not tokens:
        return {}

    def _read(start: int) -> Tuple[Dict[str, int], int]:
        counts: Dict[str, int] = defaultdict(int)
        i = start
        while i < len(tokens):
            tok = tokens[i]
            if tok == "(":
                inner, i = _read(i + 1)
                n = 1
                if i < len(tokens) and tokens[i].isdigit():
                    n = int(tokens[i])
                    i += 1
                for el, c in inner.items():
                    counts[el] += c * n
            elif tok == ")":
                return dict(counts), i + 1
            elif tok[0].isalpha():
                i += 1
                n = 1
                if i < len(tokens) and tokens[i].isdigit():
                    n = int(tokens[i])
                    i += 1
                counts[tok] += n
            else:
                i += 1
        return dict(counts), i

    counts, _ = _read(0)
    return counts


_FE_AS_FORMULAS = {
    "feo": "FeO",
    "fe2o3": "Fe2O3",
    "fe₂o₃": "Fe2O3",
    "fe3o4": "Fe3O4",
    "fe₃o₄": "Fe3O4",
    "magnetite": "Fe3O4",
    "feooh": "FeOOH",
}

_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def coerce_matrix_kind(value) -> MatrixKind:
    """Accept MatrixKind, its .value string, or a combo label."""
    if isinstance(value, MatrixKind):
        return value
    if isinstance(value, str):
        key = value.strip()
        try:
            return MatrixKind(key)
        except ValueError:
            pass
        lower = key.lower()
        if "oxide" in lower or "silicate" in lower:
            return MatrixKind.OXIDE
        if "carbonate" in lower:
            return MatrixKind.CARBONATE
        if "hydroxide" in lower:
            return MatrixKind.HYDROXIDE
        if "measured" in lower or "metal" in lower or "sulfide" in lower:
            return MatrixKind.MEASURED
    return MatrixKind.MEASURED


def fe_formula(fe_as: str, *, default: str = "FeO") -> str:
    key = (fe_as or default).strip().lower().replace(" ", "")
    return _FE_AS_FORMULAS.get(key, default)


def formula_for(cation: str, assumptions: MatrixAssumptions) -> Optional[str]:
    """Stoichiometric compound for a measured cation, or None if elemental."""
    if cation in _LIGHT:
        return None
    kind = coerce_matrix_kind(assumptions.kind)
    if kind == MatrixKind.MEASURED:
        return None
    if kind == MatrixKind.CARBONATE and cation in _CARBONATE_FORMULAS:
        return _CARBONATE_FORMULAS[cation]
    if kind == MatrixKind.HYDROXIDE and cation in _HYDROXIDE_FORMULAS:
        return _HYDROXIDE_FORMULAS[cation]
    if kind in (MatrixKind.OXIDE, MatrixKind.CARBONATE, MatrixKind.HYDROXIDE):
        if cation == "Fe":
            return fe_formula(assumptions.fe_as, default="FeO")
        return _OXIDE_FORMULAS.get(cation)
    return None


def extras_per_cation_mass(cation: str, formula: str) -> Dict[str, float]:
    """Mass of each other element in the formula per unit mass of cation."""
    counts = parse_formula(formula)
    n_cat = counts.get(cation, 0)
    if n_cat <= 0:
        return {}
    aw_cat = atomic_weight(cation)
    extras: Dict[str, float] = {}
    for el, n in counts.items():
        if el == cation or n <= 0:
            continue
        extras[el] = (n / n_cat) * atomic_weight(el) / aw_cat
    return extras


def _knob_elements(h2o: float, oh: float, co2: float) -> Dict[str, float]:
    """Elemental wt from user H2O / OH / CO2 knobs (already in wt% of bulk)."""
    out: Dict[str, float] = defaultdict(float)
    aw_h = atomic_weight("H")
    aw_c = atomic_weight("C")
    aw_o = atomic_weight("O")
    if h2o > 0:
        mw = 2.0 * aw_h + aw_o
        out["H"] += h2o * (2.0 * aw_h / mw)
        out["O"] += h2o * (aw_o / mw)
    if oh > 0:
        mw = aw_h + aw_o
        out["H"] += oh * (aw_h / mw)
        out["O"] += oh * (aw_o / mw)
    if co2 > 0:
        mw = aw_c + 2.0 * aw_o
        out["C"] += co2 * (aw_c / mw)
        out["O"] += co2 * (2.0 * aw_o / mw)
    return dict(out)


def expand_composition(
    cation_masses: Dict[str, float],
    assumptions: MatrixAssumptions,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Build a closed elemental composition from measured cation masses.

    Returns:
        element_wt: element → wt% (includes H, C, O)
        formula_wt: compound/element label → wt% (SiO2, H2O, Pb, …)
    """
    h2o = max(0.0, float(assumptions.h2o_wt))
    oh = max(0.0, float(assumptions.oh_wt))
    co2 = max(0.0, float(assumptions.co2_wt))
    light = h2o + oh + co2
    if light >= 99.9:
        raise ValueError("H2O + OH + CO2 must be less than 100 wt%.")

    remaining = 100.0 - light
    elem_mass: Dict[str, float] = defaultdict(float)
    formula_mass: Dict[str, float] = defaultdict(float)

    for cation, mass in cation_masses.items():
        m = float(mass)
        if m <= 0 or not cation or cation in _LIGHT:
            continue
        elem_mass[cation] += m
        formula = formula_for(cation, assumptions)
        added = m
        if formula:
            for el, ratio in extras_per_cation_mass(cation, formula).items():
                extra = m * ratio
                elem_mass[el] += extra
                added += extra
            formula_mass[formula] += added
        else:
            formula_mass[cation] += m

    anhydrous = float(sum(elem_mass.values()))
    if anhydrous <= 0:
        raise ValueError("No measured cations to expand.")

    scale = remaining / anhydrous
    element_wt = {k: v * scale for k, v in elem_mass.items()}
    formula_wt = {k: v * scale for k, v in formula_mass.items()}

    for el, w in _knob_elements(h2o, oh, co2).items():
        element_wt[el] = element_wt.get(el, 0.0) + w
    if h2o > 0:
        formula_wt["H2O"] = formula_wt.get("H2O", 0.0) + h2o
    if oh > 0:
        formula_wt["OH"] = formula_wt.get("OH", 0.0) + oh
    if co2 > 0:
        formula_wt["CO2"] = formula_wt.get("CO2", 0.0) + co2

    return element_wt, formula_wt


def _nice_mole_multiplier(moles: Dict[str, float]) -> float:
    """Small integer that makes mole ratios closest to whole numbers."""
    vals = [v for v in moles.values() if v > 1e-8]
    if not vals:
        return 1.0
    best_m = 1
    best_score = float("inf")
    for m in range(1, 9):
        score = 0.0
        for v in vals:
            s = v * m
            score += abs(s - round(s))
        score /= len(vals)
        if score < best_score - 1e-6 or (
            abs(score - best_score) < 1e-6 and m < best_m
        ):
            best_score = score
            best_m = m
    return float(best_m)


def _fmt_stoich(n: float) -> Optional[str]:
    """Subscript count for a formula term. None means omit the element."""
    if n < 0.03:
        return None
    nearest = round(n)
    if abs(n - nearest) < 0.04:
        if nearest <= 0:
            return None
        if nearest == 1:
            return ""
        return str(int(nearest)).translate(_SUBSCRIPTS)
    return f"{n:.2f}".rstrip("0").rstrip(".")


def empirical_formula(element_wt: Dict[str, float]) -> str:
    """Stoichiometric formula from elemental wt% (e.g. CaCO₃, SiO₂, Fe₃O₄)."""
    moles: Dict[str, float] = {}
    for el, wt in (element_wt or {}).items():
        if float(wt) <= 0.02:
            continue
        try:
            moles[el] = float(wt) / atomic_weight(el)
        except KeyError:
            continue
    if not moles:
        return "—"

    cations = {k: v for k, v in moles.items() if k not in _LIGHT}
    if cations:
        base = max(cations.values())
    else:
        base = max(moles.values())
    if base <= 0:
        return "—"
    ratios = {k: v / base for k, v in moles.items()}
    mult = _nice_mole_multiplier(ratios)
    scaled = {k: v * mult for k, v in ratios.items()}

    cations_sorted = sorted(
        (e for e in scaled if e not in _LIGHT),
        key=lambda e: -scaled[e],
    )
    order = cations_sorted + [e for e in ("C", "H", "O", "N", "F") if e in scaled]
    parts = []
    for el in order:
        token = _fmt_stoich(scaled[el])
        if token is None:
            continue
        parts.append(f"{el}{token}")
    return "".join(parts) if parts else "—"


def format_formula_wt(formula_wt: Dict[str, float], max_terms: int = 10) -> str:
    """Compound wt% line: 'CaCO3  98.2 wt% · SiO2  1.8 wt%'."""
    items = sorted((formula_wt or {}).items(), key=lambda kv: -kv[1])
    parts = []
    for name, wt in items[:max_terms]:
        if wt <= 0.05:
            continue
        parts.append(f"{name}  {wt:.1f} wt%")
    if len(items) > max_terms:
        parts.append("…")
    return "   ·   ".join(parts) if parts else "—"


def measured_cation_wt(element_wt: Dict[str, float]) -> float:
    """Sum of wt% for elements that came from the spectrum (not H/C/O)."""
    return float(sum(v for k, v in element_wt.items() if k not in _LIGHT))
