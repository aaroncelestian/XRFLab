"""
Discover pure-element spectra for FWHM calibration.

Element identity comes from the filename (Fe.txt, copper.mca, cubic zirconia)
plus the X-ray line database — not from a sidecar CSV. Mixed / certified
standards in the same folder are listed but skipped unless the user assigns
an element in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.xray_data import get_element_lines, _get_element_name


SPECTRUM_SUFFIXES = {".txt", ".dat", ".mca", ".msa", ".emsa"}

FILENAME_ALIASES = {
    "cubic zirconia": "Zr",
    "cz": "Zr",
    "zirconia": "Zr",
    "zro2": "Zr",
    "aluminium": "Al",
    "aluminum": "Al",
}

AL_KA_ENERGY = 1.487  # keV, typical sample-holder line


_Z_BY_SYMBOL = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92,
}


def _maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    by_symbol = {s.lower(): s for s in _Z_BY_SYMBOL}
    by_name = {}
    for symbol, z in _Z_BY_SYMBOL.items():
        by_name[_get_element_name(z).lower()] = symbol
    by_name["aluminium"] = "Al"
    by_name["sulfur"] = "S"
    by_name["sulphur"] = "S"
    return by_symbol, by_name


_BY_SYMBOL, _BY_NAME = _maps()


def normalize_stem(stem: str) -> str:
    text = stem.strip().lower()
    text = re.sub(r"[_./\\]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_likely_mixed_standard(stem: str) -> bool:
    """True for certified / alloy / soil names that should not drive FWHM."""
    text = normalize_stem(stem)
    if text in FILENAME_ALIASES or text in _BY_SYMBOL or text in _BY_NAME:
        return False
    # "Fe standard" is still a foil; "STDS standard" / "NIST SRM" are mixed
    if text in ("stds standard",) or text.startswith("stds "):
        return True
    mixed_tokens = (
        "nist", "srm", "lksd", "pacs", "steel", "stainless", "brass",
        "mineral", "soil", "sediment", "certified",
    )
    return any(token in text.split() or token in text for token in mixed_tokens)


def guess_element_from_stem(stem: str) -> Optional[str]:
    """
    Infer a single element symbol from a spectrum filename stem.

    Matches exact symbols (Fe), names (iron), aliases (cubic zirconia → Zr),
    and numbered replicates (Fe_2, Fe 3, Cu-1).
    """
    if is_likely_mixed_standard(stem):
        return None

    text = normalize_stem(stem)
    if text in FILENAME_ALIASES:
        return FILENAME_ALIASES[text]
    if text in _BY_SYMBOL:
        return _BY_SYMBOL[text]
    if text in _BY_NAME:
        return _BY_NAME[text]

    # First token: "Fe foil", "Cu-1", "Fe_2" (underscores already spaces)
    token = re.split(r"[\s\-]+", text)[0]
    token = token.strip("0123456789")
    if token in FILENAME_ALIASES:
        return FILENAME_ALIASES[token]
    if token in _BY_SYMBOL:
        return _BY_SYMBOL[token]
    if token in _BY_NAME:
        return _BY_NAME[token]
    return None


def fwhm_lines_for_element(
    symbol: str,
    *,
    include_holder_al: bool = True,
) -> List[Tuple[str, float]]:
    """
    Emission lines to measure for detector FWHM.

    Uses Kα1 and Kβ1 from the line database. L-lines are skipped (they are
    multiplets and were a known outlier source, e.g. Zr L).
    """
    symbol = (symbol or "").strip()
    z = _Z_BY_SYMBOL.get(symbol, 0)
    if not z:
        return []

    lines_dict = get_element_lines(symbol, z)
    k_lines = {item["name"]: float(item["energy"]) for item in lines_dict.get("K", [])}

    out: List[Tuple[str, float]] = []
    for name in ("Kα1", "Kβ1"):
        energy = k_lines.get(name)
        if energy is None or energy <= 0 or energy > 20.0:
            continue
        out.append((f"{symbol} {name}", energy))

    if include_holder_al and symbol != "Al":
        out.append(("Al Kα", AL_KA_ENERGY))
    return out


def format_line_summary(lines: Sequence[Tuple[str, float]], limit: int = 4) -> str:
    if not lines:
        return "—"
    names = [name for name, _energy in lines[:limit]]
    extra = len(lines) - limit
    text = ", ".join(names)
    if extra > 0:
        text += f" +{extra}"
    return text


@dataclass
class FWHMStandardFile:
    """One spectrum in a FWHM standards folder."""
    path: Path
    filename: str
    stem: str
    element: Optional[str] = None
    included: bool = False
    reason: str = ""
    lines: List[Tuple[str, float]] = field(default_factory=list)

    def line_summary(self) -> str:
        return format_line_summary(self.lines)


def example_standards_dir() -> Optional[Path]:
    """Shipped foil folder next to this repo, if present."""
    root = Path(__file__).resolve().parents[1]
    candidate = root / "sample_data" / "data"
    return candidate if candidate.is_dir() else None


def _skip_subdir(name: str) -> bool:
    text = name.strip().lower()
    if text.startswith("."):
        return True
    if "spectrum value" in text:
        return True
    return is_likely_mixed_standard(name)


def _make_standard_file(
    path: Path,
    *,
    relative_name: str,
    element: Optional[str],
    include_holder_al: bool,
) -> FWHMStandardFile:
    mixed = element is None and is_likely_mixed_standard(path.stem)
    lines = fwhm_lines_for_element(element, include_holder_al=include_holder_al) if element else []
    if mixed:
        reason = "Looks like a mixed / certified standard — skipped for FWHM"
        included = False
    elif element and lines:
        reason = f"Using {element} K-lines from the line database"
        included = True
    elif element:
        reason = f"Recognized {element} but no usable K-lines"
        included = False
    else:
        reason = "Could not tell which element — assign one to include"
        included = False
    return FWHMStandardFile(
        path=path,
        filename=relative_name,
        stem=path.stem,
        element=element,
        included=included,
        reason=reason,
        lines=lines,
    )


def scan_fwhm_folder(
    data_dir: Path,
    *,
    include_holder_al: bool = True,
) -> List[FWHMStandardFile]:
    """
    List spectra in *data_dir* (and one level of element-named subfolders).

    Replicates are extra files for the same element (Fe.txt, Fe_2.txt, or
    Fe/Spectrum 1.txt). Mixed / certified folders are not recursed.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    results: List[FWHMStandardFile] = []
    for path in sorted(data_dir.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in SPECTRUM_SUFFIXES:
            results.append(
                _make_standard_file(
                    path,
                    relative_name=path.name,
                    element=guess_element_from_stem(path.stem),
                    include_holder_al=include_holder_al,
                )
            )
            continue
        if not path.is_dir() or _skip_subdir(path.name):
            continue
        folder_element = guess_element_from_stem(path.name)
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_file() or child.suffix.lower() not in SPECTRUM_SUFFIXES:
                continue
            element = guess_element_from_stem(child.stem) or folder_element
            results.append(
                _make_standard_file(
                    child,
                    relative_name=f"{path.name}/{child.name}",
                    element=element,
                    include_holder_al=include_holder_al,
                )
            )
    return results


def assignments_to_file_peaks(
    files: Iterable[FWHMStandardFile],
) -> Dict[str, Tuple[str, List[Tuple[str, float]]]]:
    """Map included filenames → (element, lines) for PeakShapeCalibrator."""
    out: Dict[str, Tuple[str, List[Tuple[str, float]]]] = {}
    for item in files:
        if not item.included or not item.element or not item.lines:
            continue
        out[item.filename] = (item.element, list(item.lines))
    return out


def line_group_key(measurement) -> str:
    """Stable key so Fe Kα1 / Fe Kα2 / replicate files group together."""
    line = (getattr(measurement, "line", None) or "").strip()
    element = (getattr(measurement, "element", None) or "").strip()
    if line:
        parts = line.split()
        series = re.sub(r"[12]$", "", parts[-1])
        head = parts[0] if len(parts) >= 2 else element
        return f"{head} {series}".strip()
    return element or "?"


@dataclass
class FWHMLineStats:
    """Replicate summary for one emission line."""
    label: str
    n: int
    energy_mean: float  # keV
    energy_std: float   # keV
    fwhm_mean: float    # keV
    fwhm_std: float     # keV
    fwhm_sem: float     # keV, std / sqrt(n)


def group_fwhm_replicates(measurements: Sequence) -> List[FWHMLineStats]:
    """Mean ± spread of FWHM for each unique line across replicate spectra."""
    buckets: Dict[str, List] = {}
    for measurement in measurements or []:
        key = line_group_key(measurement)
        buckets.setdefault(key, []).append(measurement)

    stats: List[FWHMLineStats] = []
    for label, group in buckets.items():
        energies = np.array([float(m.energy) for m in group], dtype=float)
        fwhms = np.array([float(m.fwhm) for m in group], dtype=float)
        n = len(group)
        std = float(np.std(fwhms, ddof=1)) if n >= 2 else 0.0
        stats.append(
            FWHMLineStats(
                label=label,
                n=n,
                energy_mean=float(np.mean(energies)),
                energy_std=float(np.std(energies, ddof=1)) if n >= 2 else 0.0,
                fwhm_mean=float(np.mean(fwhms)),
                fwhm_std=std,
                fwhm_sem=std / np.sqrt(n) if n >= 2 else 0.0,
            )
        )
    stats.sort(key=lambda s: s.energy_mean)
    return stats
