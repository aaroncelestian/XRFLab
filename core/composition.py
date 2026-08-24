"""
Bulk-rock composition tables: group replicate spectra, average samples,
and prepare correlate / ternary / ratio coordinates.

Intensities are whatever Batch produced (typically area-normalized
semi-quant). Oxide conversion and closing are optional display steps —
they do not turn relative intensities into fundamental-parameters wt%.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Last token looks like a replicate: _1, -rep2, _spot3, _r4, _s01
_REPLICATE_INDEX = re.compile(
    r"^(?P<sample>.+?)[_-](?:rep|r|spot|s)?\d+$",
    re.IGNORECASE,
)
# Trailing single letter: B01_a, Basalt01-d
_REPLICATE_LETTER = re.compile(
    r"^(?P<sample>.+)[_-][a-z]$",
    re.IGNORECASE,
)

DEFAULT_GROUP_REGEX = (
    r"^(?P<sample>.+?)(?:[_-](?:rep|r|spot)?\d+|[_\-][a-dA-D])$"
)

# oxide_wt = element_wt * factor ; factor = MW_oxide / (n_cations * AW_element)
_OXIDE_AS_FEO = {
    "Si": ("SiO2", 2.1393),
    "Al": ("Al2O3", 1.8895),
    "Fe": ("FeO", 1.2865),
    "Mg": ("MgO", 1.6583),
    "Ca": ("CaO", 1.3992),
    "Na": ("Na2O", 1.3480),
    "K": ("K2O", 1.2046),
    "Ti": ("TiO2", 1.6685),
    "Mn": ("MnO", 1.2912),
    "P": ("P2O5", 2.2916),
    "Cr": ("Cr2O3", 1.4615),
    "Ni": ("NiO", 1.2726),
    "Cu": ("CuO", 1.2518),
    "Zn": ("ZnO", 1.2447),
    "Sr": ("SrO", 1.1826),
    "Ba": ("BaO", 1.1165),
    "Zr": ("ZrO2", 1.3508),
    "S": ("SO3", 2.4972),
    "Pb": ("PbO", 1.0772),
    "V": ("V2O5", 1.7852),
    "Co": ("CoO", 1.2715),
}

_OXIDE_AS_FE2O3 = dict(_OXIDE_AS_FEO)
_OXIDE_AS_FE2O3["Fe"] = ("Fe2O3", 1.4297)
_OXIDE_AS_FE3O4 = dict(_OXIDE_AS_FEO)
_OXIDE_AS_FE3O4["Fe"] = ("Fe3O4", 1.3820)

SQRT3_OVER_2 = 0.8660254037844386


class GroupMode(str, Enum):
    AUTO = "auto"
    FOLDER = "folder"
    PREFIX = "prefix"
    REGEX = "regex"
    SEQUENTIAL = "sequential"
    NONE = "none"


@dataclass
class CompositionRow:
    """One fitted spectrum (or other composition point)."""

    name: str
    source_id: str
    sample: str
    values: Dict[str, float]
    success: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_id": self.source_id,
            "sample": self.sample,
            "values": {str(k): float(v) for k, v in (self.values or {}).items()},
            "success": bool(self.success),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompositionRow":
        return cls(
            name=str(data.get("name") or ""),
            source_id=str(data.get("source_id") or ""),
            sample=str(data.get("sample") or data.get("name") or ""),
            values={
                str(k): float(v) for k, v in (data.get("values") or {}).items()
            },
            success=bool(data.get("success", True)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SampleSummary:
    """Replicate-averaged composition for one sample."""

    sample: str
    n: int
    mean: Dict[str, float]
    std: Dict[str, float]
    rows: List[CompositionRow] = field(default_factory=list)

    def member_names(self) -> List[str]:
        return [r.name for r in self.rows]


def rows_from_batch_results(results: Iterable) -> List[CompositionRow]:
    """Convert BatchFitResult-like objects into composition rows."""
    rows: List[CompositionRow] = []
    for result in results:
        name = str(getattr(result, "spectrum_name", "") or "spectrum")
        path = str(getattr(result, "spectrum_path", "") or name)
        conc = getattr(result, "concentrations", None) or {}
        values = {str(k): float(v) for k, v in dict(conc).items()}
        rows.append(
            CompositionRow(
                name=name,
                source_id=path,
                sample=name,
                values=values,
                success=bool(getattr(result, "fit_success", True)),
            )
        )
    return rows


def strip_replicate_suffix(stem: str) -> str:
    """Drop a trailing replicate token from a file stem."""
    text = str(stem).strip()
    if not text:
        return text
    match = _REPLICATE_INDEX.match(text)
    if match:
        sample = match.group("sample").strip()
        if sample:
            return sample
    match = _REPLICATE_LETTER.match(text)
    if match:
        sample = match.group("sample").strip()
        if sample:
            return sample
    return text


def numbered_replicate_names(base: str, count: int, start: int = 1) -> List[str]:
    """Build ``Name`` (count=1) or ``Name_1`` … ``Name_n`` (padded when n >= 10).

    These names round-trip through ``strip_replicate_suffix``, so Composition
    can group the spots as one sample.
    """
    base = str(base).strip()
    count = int(count)
    if count <= 0 or not base:
        return []
    if count == 1:
        return [base]
    last = int(start) + count - 1
    width = len(str(last))
    return [f"{base}_{i:0{width}d}" for i in range(int(start), last + 1)]


def sample_from_folder(source_id: str, fallback: str) -> str:
    path = Path(source_id)
    parent = path.parent.name if path.parent.name else ""
    if parent and parent not in {".", ""}:
        return parent
    return fallback


def sample_from_regex(name: str, pattern: str) -> Optional[str]:
    try:
        compiled = re.compile(pattern)
    except re.error:
        return None
    match = compiled.search(name)
    if not match:
        return None
    if "sample" in match.groupdict() and match.group("sample"):
        return match.group("sample").strip()
    if match.lastindex and match.group(1):
        return match.group(1).strip()
    return None


_TRAILING_INDEX = re.compile(r"(\d+)\s*$")


def _trailing_index(text: str) -> Optional[int]:
    match = _TRAILING_INDEX.search(str(text).strip())
    return int(match.group(1)) if match else None


def _natural_key(text: str):
    parts = re.split(r"(\d+)", str(text).lower())
    return [int(p) if p.isdigit() else p for p in parts]


def sequential_sample_name(index: int, n_groups: int, prefix: str = "Sample") -> str:
    """1-based sample label, zero-padded (Sample 01 … Sample 10)."""
    width = max(2, len(str(max(int(n_groups), 1))))
    label = f"{int(index):0{width}d}"
    prefix = str(prefix or "").strip()
    return f"{prefix} {label}".strip() if prefix else label


def assign_sequential(
    rows: Sequence[CompositionRow],
    spots_per_sample: int,
    n_samples: Optional[int] = None,
    prefix: str = "Sample",
) -> None:
    """
    Chunk spectra in collection order into N samples × M spots.

    Order is trailing Spectrum N when every name has one (so Spectrum 10
    follows Spectrum 9, not Spectrum 1). Otherwise list order is kept.
    Extra spectra past N×M become additional samples.
    """
    spots = max(1, int(spots_per_sample or 1))
    n = len(rows)
    if n == 0:
        return

    keys: List[Optional[int]] = []
    for row in rows:
        idx = _trailing_index(row.name)
        if idx is None:
            idx = _trailing_index(Path(row.source_id).stem if row.source_id else "")
        keys.append(idx)
    if all(k is not None for k in keys):
        order = sorted(range(n), key=lambda i: (keys[i], i))
    else:
        order = list(range(n))

    n_groups = max(1, (n + spots - 1) // spots)
    if n_samples and int(n_samples) > 0:
        n_groups = max(n_groups, int(n_samples))

    for rank, row_i in enumerate(order):
        sample_i = rank // spots
        rows[row_i].sample = sequential_sample_name(
            sample_i + 1, n_groups, prefix
        )


def resolve_group_mode(rows: Sequence[CompositionRow], mode: GroupMode) -> GroupMode:
    """AUTO: folder if 2+ parent dirs and at least one has replicates, else prefix."""
    if mode != GroupMode.AUTO:
        return mode
    parents = []
    for row in rows:
        parent = str(Path(row.source_id).parent)
        parents.append(parent)
    unique = set(parents)
    if len(unique) >= 2:
        counts: Dict[str, int] = {}
        for parent in parents:
            counts[parent] = counts.get(parent, 0) + 1
        if max(counts.values()) >= 2:
            return GroupMode.FOLDER
    return GroupMode.PREFIX


def assign_samples(
    rows: Sequence[CompositionRow],
    mode: GroupMode = GroupMode.AUTO,
    regex: str = DEFAULT_GROUP_REGEX,
    *,
    spots_per_sample: int = 1,
    n_samples: Optional[int] = None,
    prefix: str = "Sample",
) -> GroupMode:
    """Set row.sample from path/name. Returns the mode actually used."""
    used = resolve_group_mode(rows, mode)
    if used == GroupMode.SEQUENTIAL:
        assign_sequential(
            rows,
            spots_per_sample=spots_per_sample,
            n_samples=n_samples,
            prefix=prefix,
        )
        return used
    for row in rows:
        stem = Path(row.source_id).stem if row.source_id else row.name
        if used == GroupMode.NONE:
            row.sample = row.name
        elif used == GroupMode.FOLDER:
            row.sample = sample_from_folder(row.source_id, strip_replicate_suffix(stem))
        elif used == GroupMode.REGEX:
            row.sample = (
                sample_from_regex(stem, regex)
                or sample_from_regex(row.name, regex)
                or strip_replicate_suffix(stem)
            )
        else:  # PREFIX
            row.sample = strip_replicate_suffix(stem)
    return used


def group_counts(rows: Sequence[CompositionRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.sample] = counts.get(row.sample, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: _natural_key(kv[0])))


def all_elements(rows: Sequence[CompositionRow]) -> List[str]:
    found = set()
    for row in rows:
        found.update(row.values.keys())
    return sorted(found)


def summarize_samples(
    rows: Sequence[CompositionRow],
    *,
    successful_only: bool = True,
) -> List[SampleSummary]:
    """Average compositions by sample. Missing elements count as 0."""
    buckets: Dict[str, List[CompositionRow]] = {}
    for row in rows:
        if successful_only and not row.success:
            continue
        buckets.setdefault(row.sample, []).append(row)

    summaries: List[SampleSummary] = []
    for sample in sorted(buckets.keys(), key=_natural_key):
        members = buckets[sample]
        elements = all_elements(members)
        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        n = len(members)
        for element in elements:
            vals = np.array(
                [float(m.values.get(element, 0.0)) for m in members],
                dtype=np.float64,
            )
            mean[element] = float(np.mean(vals)) if n else 0.0
            std[element] = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
        summaries.append(
            SampleSummary(
                sample=sample,
                n=n,
                mean=mean,
                std=std,
                rows=list(members),
            )
        )
    return summaries


def oxide_factor_table(fe_as: str = "FeO") -> Dict[str, Tuple[str, float]]:
    key = (fe_as or "FeO").strip().lower().replace(" ", "")
    if key in {"fe2o3", "fe₂o₃"}:
        return _OXIDE_AS_FE2O3
    if key in {"fe3o4", "fe₃o₄", "magnetite"}:
        return _OXIDE_AS_FE3O4
    return _OXIDE_AS_FEO


def oxide_label(element: str, fe_as: str = "FeO") -> str:
    table = oxide_factor_table(fe_as)
    if element in table:
        return table[element][0]
    return element


def convert_values(
    values: Dict[str, float],
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> Dict[str, float]:
    """Optionally convert elements to oxides and/or close to 100%."""
    if as_oxides:
        table = oxide_factor_table(fe_as)
        converted: Dict[str, float] = {}
        for element, value in values.items():
            if element in table:
                formula, factor = table[element]
                converted[formula] = converted.get(formula, 0.0) + float(value) * factor
            else:
                converted[element] = converted.get(element, 0.0) + float(value)
        values = converted
    if close:
        return close_to_100(values)
    return dict(values)


def close_to_100(values: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0:
        return {k: 0.0 for k in values}
    return {k: 100.0 * float(v) / total for k, v in values.items()}


def display_values(
    summary: SampleSummary,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> Dict[str, float]:
    return convert_values(
        summary.mean, as_oxides=as_oxides, fe_as=fe_as, close=close
    )


def display_row_values(
    row: CompositionRow,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> Dict[str, float]:
    return convert_values(
        row.values, as_oxides=as_oxides, fe_as=fe_as, close=close
    )


def row_composition_summary(
    row: CompositionRow,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
    max_terms: int = 6,
) -> str:
    """Compact formula-style summary for one spectrum row."""
    vals = display_row_values(
        row, as_oxides=as_oxides, fe_as=fe_as, close=close
    )
    if not vals:
        return "—"
    top = sorted(vals.items(), key=lambda item: -item[1])[: max(1, int(max_terms))]
    return ", ".join(f"{key} {value:.1f}" for key, value in top)


def component_keys(
    summaries: Sequence[SampleSummary],
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> List[str]:
    found = set()
    for summary in summaries:
        found.update(
            display_values(
                summary, as_oxides=as_oxides, fe_as=fe_as, close=close
            ).keys()
        )
    return sorted(found)


def ternary_xy(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Cartesian coordinates in an equilateral triangle.

    A (bottom left) = (0, 0), B (bottom right) = (1, 0),
    C (top) = (0.5, √3/2). a+b+c is renormalized if needed.
    """
    total = float(a) + float(b) + float(c)
    if total <= 0:
        return float("nan"), float("nan")
    aa, bb, cc = a / total, b / total, c / total
    x = bb + 0.5 * cc
    y = SQRT3_OVER_2 * cc
    return float(x), float(y)


def _pick(values: Dict[str, float], key: str) -> float:
    if key in values:
        return float(values[key])
    # Allow looking up Fe when the table is FeO, etc.
    return 0.0


def ternary_points(
    summaries: Sequence[SampleSummary],
    el_a: str,
    el_b: str,
    el_c: str,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = True,
) -> List[Tuple[str, float, float, float, float, float]]:
    """
    Returns (sample, x, y, a, b, c) for each summary with a+b+c > 0.

    a,b,c are the closed ternary fractions (0–100).
    """
    points = []
    for summary in summaries:
        vals = display_values(
            summary, as_oxides=as_oxides, fe_as=fe_as, close=close
        )
        a, b, c = _pick(vals, el_a), _pick(vals, el_b), _pick(vals, el_c)
        x, y = ternary_xy(a, b, c)
        if not np.isfinite(x):
            continue
        total = a + b + c
        points.append(
            (
                summary.sample,
                x,
                y,
                100.0 * a / total,
                100.0 * b / total,
                100.0 * c / total,
            )
        )
    return points


def replicate_ternary_points(
    summaries: Sequence[SampleSummary],
    el_a: str,
    el_b: str,
    el_c: str,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = True,
) -> List[Tuple[str, str, float, float]]:
    """(sample, spectrum_name, x, y) for each replicate."""
    points = []
    for summary in summaries:
        for row in summary.rows:
            vals = display_row_values(
                row, as_oxides=as_oxides, fe_as=fe_as, close=close
            )
            a, b, c = _pick(vals, el_a), _pick(vals, el_b), _pick(vals, el_c)
            x, y = ternary_xy(a, b, c)
            if not np.isfinite(x):
                continue
            points.append((summary.sample, row.name, x, y))
    return points


def scatter_points(
    summaries: Sequence[SampleSummary],
    el_x: str,
    el_y: str,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> List[Tuple[str, float, float, float, float]]:
    """(sample, x, y, x_std, y_std) using mean compositions."""
    points = []
    for summary in summaries:
        vals = display_values(
            summary, as_oxides=as_oxides, fe_as=fe_as, close=close
        )
        std_vals = convert_values(
            summary.std, as_oxides=as_oxides, fe_as=fe_as, close=False
        )
        x, y = _pick(vals, el_x), _pick(vals, el_y)
        points.append(
            (
                summary.sample,
                x,
                y,
                float(std_vals.get(el_x, 0.0)),
                float(std_vals.get(el_y, 0.0)),
            )
        )
    return points


def replicate_scatter_points(
    summaries: Sequence[SampleSummary],
    el_x: str,
    el_y: str,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> List[Tuple[str, str, float, float]]:
    points = []
    for summary in summaries:
        for row in summary.rows:
            vals = display_row_values(
                row, as_oxides=as_oxides, fe_as=fe_as, close=close
            )
            points.append(
                (summary.sample, row.name, _pick(vals, el_x), _pick(vals, el_y))
            )
    return points


def ratio_value(values: Dict[str, float], numer: str, denom: str) -> float:
    d = _pick(values, denom)
    if abs(d) < 1e-12:
        return float("nan")
    return _pick(values, numer) / d


def ratio_points(
    summaries: Sequence[SampleSummary],
    x_num: str,
    x_den: str,
    y_num: str,
    y_den: str,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> List[Tuple[str, float, float]]:
    points = []
    for summary in summaries:
        vals = display_values(
            summary, as_oxides=as_oxides, fe_as=fe_as, close=close
        )
        x = ratio_value(vals, x_num, x_den)
        y = ratio_value(vals, y_num, y_den)
        if np.isfinite(x) and np.isfinite(y):
            points.append((summary.sample, float(x), float(y)))
    return points


def replicate_ratio_points(
    summaries: Sequence[SampleSummary],
    x_num: str,
    x_den: str,
    y_num: str,
    y_den: str,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> List[Tuple[str, str, float, float]]:
    points = []
    for summary in summaries:
        for row in summary.rows:
            vals = display_row_values(
                row, as_oxides=as_oxides, fe_as=fe_as, close=close
            )
            x = ratio_value(vals, x_num, x_den)
            y = ratio_value(vals, y_num, y_den)
            if np.isfinite(x) and np.isfinite(y):
                points.append((summary.sample, row.name, float(x), float(y)))
    return points


def correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson r and Spearman ρ. NaN if fewer than 2 finite pairs."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 2:
        return float("nan"), float("nan")
    if np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(xa, ya)[0, 1])
    ra = xa.argsort().argsort().astype(np.float64)
    rb = ya.argsort().argsort().astype(np.float64)
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    return pearson, spearman


def correlation_matrix(
    summaries: Sequence[SampleSummary],
    keys: Sequence[str],
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
) -> np.ndarray:
    """Pearson r between sample-mean columns. Shape (n_keys, n_keys)."""
    n = len(keys)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    if len(summaries) < 2 or n == 0:
        return matrix
    cols = []
    for key in keys:
        cols.append(
            np.array(
                [
                    _pick(
                        display_values(
                            s, as_oxides=as_oxides, fe_as=fe_as, close=close
                        ),
                        key,
                    )
                    for s in summaries
                ],
                dtype=np.float64,
            )
        )
    data = np.column_stack(cols) if cols else np.empty((len(summaries), 0))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            r, _ = correlation(data[:, i], data[:, j])
            matrix[i, j] = r
            matrix[j, i] = r
    return matrix


def export_sample_means_csv(
    path: Path,
    summaries: Sequence[SampleSummary],
    keys: Optional[Sequence[str]] = None,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
    include_std: bool = True,
) -> None:
    if keys is None:
        keys = component_keys(
            summaries, as_oxides=as_oxides, fe_as=fe_as, close=close
        )
    fieldnames = ["Sample", "n"]
    for key in keys:
        fieldnames.append(key)
        if include_std:
            fieldnames.append(f"{key} std")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            vals = display_values(
                summary, as_oxides=as_oxides, fe_as=fe_as, close=close
            )
            std_vals = convert_values(
                summary.std, as_oxides=as_oxides, fe_as=fe_as, close=False
            )
            row = {"Sample": summary.sample, "n": summary.n}
            for key in keys:
                row[key] = f"{vals.get(key, 0.0):.4f}"
                if include_std:
                    row[f"{key} std"] = f"{std_vals.get(key, 0.0):.4f}"
            writer.writerow(row)


def export_sample_means_excel(
    path: Path,
    summaries: Sequence[SampleSummary],
    keys: Optional[Sequence[str]] = None,
    *,
    as_oxides: bool = False,
    fe_as: str = "FeO",
    close: bool = False,
    include_std: bool = True,
) -> None:
    import pandas as pd

    if keys is None:
        keys = component_keys(
            summaries, as_oxides=as_oxides, fe_as=fe_as, close=close
        )
    records = []
    for summary in summaries:
        vals = display_values(
            summary, as_oxides=as_oxides, fe_as=fe_as, close=close
        )
        std_vals = convert_values(
            summary.std, as_oxides=as_oxides, fe_as=fe_as, close=False
        )
        rec = {"Sample": summary.sample, "n": summary.n}
        for key in keys:
            rec[key] = vals.get(key, 0.0)
            if include_std:
                rec[f"{key} std"] = std_vals.get(key, 0.0)
        records.append(rec)
    pd.DataFrame(records).to_excel(path, index=False)
