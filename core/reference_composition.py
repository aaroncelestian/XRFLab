"""
Load certified concentrations for intensity (standards) calibration.

A CSV with the same filename as the spectrum is one discovery hint, not the
UI. The user always confirms an editable table; we just try to pre-fill it.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


COMPOSITION_NAME_HINTS = ("elements", "concentration", "certified", "srm")
_SKIP_CSV_NAMES = {"xrf_lines_na_to_u.csv", "xrf_lines.csv"}


def _is_header_cell(value: str) -> bool:
    text = value.lower().strip()
    return any(
        token in text
        for token in ("element", "symbol", "conc", "wt", "mg/kg", "ppm", "notes")
    )


def _looks_like_symbol(value: str) -> bool:
    text = value.strip()
    if not text or len(text) > 2:
        return False
    return text.isalpha() and text[0].isupper()


def _to_wt_percent(value: float, *, units: str) -> float:
    if units in ("mg/kg", "ppm"):
        return value / 10000.0
    return value


def load_composition_csv(csv_path: str | Path) -> Dict[str, float]:
    """
    Load element concentrations as wt%.

    Accepts NIST-style headers (Symbol, Concentration_mg_kg) and simple
    Element, Concentration rows (wt% or mg/kg).
    """
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return {}

    header = [col.strip() for col in rows[0]]
    header_l = [col.lower() for col in header]
    has_header = any(_is_header_cell(col) for col in header)

    element_col = 0
    conc_col = 1 if len(header) > 1 else 0
    units = "wt%"

    if has_header:
        for i, col in enumerate(header_l):
            if "symbol" in col or col == "element":
                element_col = i
                break
        for i, col in enumerate(header_l):
            if "concentration" in col or col in ("conc", "wt%", "wt"):
                conc_col = i
                break
        conc_header = header_l[conc_col] if conc_col < len(header_l) else ""
        if "mg/kg" in conc_header or "ppm" in conc_header:
            units = "mg/kg"
        data_rows = rows[1:]
    else:
        data_rows = rows

    concentrations: Dict[str, float] = {}
    values: List[float] = []
    pending: List[Tuple[str, float]] = []

    for row in data_rows:
        if len(row) <= max(element_col, conc_col):
            continue
        element = row[element_col].strip()
        if not element or element.startswith("#"):
            continue
        # Prefer a 1–2 letter symbol if a name column was picked by mistake
        if not _looks_like_symbol(element) and len(row) > element_col + 1:
            maybe = row[element_col + 1].strip()
            if _looks_like_symbol(maybe):
                element = maybe
        try:
            raw = float(row[conc_col].strip())
        except (ValueError, IndexError):
            continue
        if raw <= 0:
            continue
        pending.append((element, raw))
        values.append(raw)

    if units == "wt%" and values and max(values) > 100:
        units = "mg/kg"

    for element, raw in pending:
        wt = _to_wt_percent(raw, units=units)
        if wt > 0:
            concentrations[element] = wt

    return concentrations


def _digit_keys(*parts: str) -> List[str]:
    keys = []
    for part in parts:
        keys.extend(re.findall(r"\d{4,}", part or ""))
    return keys


def _score_candidate(path: Path, stems: Sequence[str], name: str, digits: Sequence[str]) -> int:
    """Higher is a better match. 0 means skip."""
    filename = path.name.lower()
    if filename in _SKIP_CSV_NAMES:
        return 0
    stem_l = path.stem.lower()
    score = 0
    for stem in stems:
        s = stem.lower()
        if stem_l == s:
            score = max(score, 100)
        elif stem_l == f"{s}_elements" or stem_l == f"{s.replace(' ', '_')}_elements":
            score = max(score, 90)
        elif s.replace(" ", "") in stem_l.replace(" ", "").replace("_", ""):
            score = max(score, 40)
    name_l = (name or "").lower()
    if name_l and name_l.replace(" ", "") in stem_l.replace(" ", "").replace("_", ""):
        score = max(score, 50)
    for key in digits:
        if key in stem_l:
            score = max(score, 80 if "element" in stem_l else 55)
    if any(hint in stem_l for hint in COMPOSITION_NAME_HINTS):
        score = max(score, score + 5 if score else 10)
    return score


def find_composition_csv(
    spectrum_paths: Sequence[str | Path],
    *,
    standard_name: str = "",
    extra_dirs: Optional[Iterable[str | Path]] = None,
) -> Optional[Path]:
    """
    Find a likely composition CSV near the spectra.

    Search order (highest score wins):
    - same stem as a spectrum (NIST 2586.csv)
    - {stem}_elements.csv
    - *2586*elements*.csv when the name/file contains those digits
    """
    paths = [Path(p) for p in spectrum_paths if p]
    if not paths:
        return None

    stems = [p.stem for p in paths]
    folders = {p.parent for p in paths}
    if extra_dirs:
        folders.update(Path(d) for d in extra_dirs)

    digits = _digit_keys(standard_name, *stems)
    best: Optional[Path] = None
    best_score = 0

    for folder in folders:
        if not folder.is_dir():
            continue
        for csv_path in folder.glob("*.csv"):
            score = _score_candidate(csv_path, stems, standard_name, digits)
            if score > best_score:
                best_score = score
                best = csv_path

    # Same-stem next to the first spectrum is a strong default even if
    # scoring is conservative.
    if best is None:
        sibling = paths[0].with_suffix(".csv")
        if sibling.is_file() and sibling.name.lower() not in _SKIP_CSV_NAMES:
            return sibling
    return best if best_score >= 40 else None
