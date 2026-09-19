"""
Self-contained HTML analysis report for an XRFLab session.

The engine is Qt-free: gather a ReportContext from the UI (or tests),
choose ReportOptions (sections + composition display mode), then write
one HTML file with inlined PNG figures.
"""

from __future__ import annotations

import base64
import html
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from core.composition import (
    VALUE_RELATIVE,
    VALUE_WT,
    CompositionRow,
    SampleSummary,
    apply_value_source,
    component_keys,
    convert_values,
    display_values,
    summarize_samples,
)

SECTION_METHODS = "methods"
SECTION_FWHM = "fwhm"
SECTION_TUBE = "tube"
SECTION_STANDARDS = "standards"
SECTION_FIT = "fit"
SECTION_COMPOSITION = "composition"
SECTION_ERRORS = "errors"

ALL_SECTIONS: Tuple[str, ...] = (
    SECTION_METHODS,
    SECTION_FWHM,
    SECTION_TUBE,
    SECTION_STANDARDS,
    SECTION_FIT,
    SECTION_COMPOSITION,
    SECTION_ERRORS,
)

SECTION_TITLES = {
    SECTION_METHODS: "Methods and instrument",
    SECTION_FWHM: "FWHM / peak-shape calibration",
    SECTION_TUBE: "Tube profiles",
    SECTION_STANDARDS: "Standards calibration",
    SECTION_FIT: "Spectrum fit example",
    SECTION_COMPOSITION: "Compositions",
    SECTION_ERRORS: "Error analysis and statistics",
}

METHOD_LABELS = {
    "semi_quant_area": "Semi-quant (relative intensity)",
    "fp_matrix": "Fundamental parameters (wt%)",
    "standards_curve": "Standards calibration (wt%)",
}

Z_FLAG = 3.0
CURVE_PLOT_CAP = 12
_SLOPE_EPS = 1e-15


# --------------------------------------------------------------------------- #
# Options / context
# --------------------------------------------------------------------------- #
@dataclass
class ReportOptions:
    """Section selection and composition display mode for one export."""

    sections: Set[str] = field(default_factory=lambda: set(ALL_SECTIONS))
    value_source: str = VALUE_RELATIVE
    as_oxides: bool = False
    fe_as: str = "FeO"
    close: bool = False
    include_replicates: bool = True
    app_version: str = ""
    title: str = "XRFLab Analysis Report"
    include_plots: bool = True

    def wants(self, section: str) -> bool:
        return section in self.sections


@dataclass
class ReportContext:
    """Snapshot of session data used to build a report."""

    spectrum: Any = None
    spectrum_path: Optional[str] = None
    project_path: Optional[str] = None
    fit_result: Any = None
    concentrations: Dict[str, Any] = field(default_factory=dict)
    quantification_method: str = "semi_quant_area"
    matrix: Any = None
    fp_result: Any = None
    fwhm_calibration: Any = None
    fwhm_measurements: Optional[Sequence[Any]] = None
    tube_library: Any = None
    standards_calibration: Any = None
    composition_rows: List[CompositionRow] = field(default_factory=list)
    composition_summaries: List[SampleSummary] = field(default_factory=list)
    detector: Any = None
    excitation_kv: Optional[float] = None
    tube_element: Optional[str] = None
    fit_settings: Dict[str, Any] = field(default_factory=dict)
    generated: Optional[datetime] = None


def app_version(default: str = "1.3.1") -> str:
    """Read the project version from pyproject.toml when available."""
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("version"):
                _, _, rest = stripped.partition("=")
                return rest.strip().strip('"').strip("'") or default
    except OSError:
        pass
    return default


def options_from_composition_state(
    state: Optional[dict],
    sections: Optional[Iterable[str]] = None,
) -> ReportOptions:
    """Prefill display-mode flags from CompositionPanel.capture_state()."""
    state = state or {}
    source = str(state.get("value_source") or VALUE_RELATIVE).lower()
    if source not in (VALUE_WT, VALUE_RELATIVE):
        source = VALUE_RELATIVE
    chosen = set(sections) if sections is not None else set(ALL_SECTIONS)
    return ReportOptions(
        sections=chosen,
        value_source=source,
        as_oxides=bool(state.get("oxides")),
        fe_as=str(state.get("fe_as") or "FeO"),
        close=bool(state.get("close")),
        app_version=app_version(),
    )


def display_kwargs(options: ReportOptions) -> Dict[str, Any]:
    """Kwargs for convert_values / display_values matching the Composition UI."""
    oxide_factors = bool(options.as_oxides) and options.value_source != VALUE_WT
    return {
        "as_oxides": oxide_factors,
        "fe_as": options.fe_as or "FeO",
        "close": bool(options.close),
    }


def composition_unit_label(options: ReportOptions) -> str:
    if options.close:
        return "% (closed)"
    if options.value_source == VALUE_WT:
        return "wt% (formula)" if options.as_oxides else "wt%"
    if options.as_oxides:
        return "oxide units"
    return "rel. %"


def has_standards_curves(obj: Any) -> bool:
    curves = getattr(obj, "curves", None)
    if not curves:
        return False
    return any(
        getattr(c, "fitted", False) and getattr(c, "enabled", True)
        for c in curves.values()
    )


def has_tube_profiles(library: Any) -> bool:
    profiles = getattr(library, "profiles", None)
    return bool(profiles)


def section_availability(context: ReportContext) -> Dict[str, Optional[str]]:
    """Map section id → None if available, else a short reason to disable."""
    reasons: Dict[str, Optional[str]] = {key: None for key in ALL_SECTIONS}
    if context.fwhm_calibration is None:
        reasons[SECTION_FWHM] = "No FWHM calibration applied"
    if not has_tube_profiles(context.tube_library):
        reasons[SECTION_TUBE] = "No tube profiles loaded"
    if not has_standards_curves(context.standards_calibration):
        reasons[SECTION_STANDARDS] = "No standards curves applied"
    if context.fit_result is None:
        reasons[SECTION_FIT] = "No spectrum fit in Analysis"
    if not _has_composition(context):
        reasons[SECTION_COMPOSITION] = "No composition results"
    if (
        context.fit_result is None
        and not has_standards_curves(context.standards_calibration)
        and context.fp_result is None
        and not context.composition_summaries
        and not context.composition_rows
    ):
        reasons[SECTION_ERRORS] = "No fit, calibration, or composition statistics"
    return reasons


def _has_composition(context: ReportContext) -> bool:
    if context.composition_rows or context.composition_summaries:
        return True
    if context.fp_result is not None:
        return True
    return bool(context.concentrations)


# --------------------------------------------------------------------------- #
# Analytical extras
# --------------------------------------------------------------------------- #
def curve_slope_at_reference(curve: Any) -> float:
    """dC/dI used for LOD: mean intensity for quadratic, else slope at I = 0."""
    coeffs = list(getattr(curve, "coefficients", None) or [0.0, 0.0, 0.0])
    while len(coeffs) < 3:
        coeffs.append(0.0)
    model = str(getattr(curve, "model", "linear") or "linear")
    if model == "quadratic":
        pts = list(getattr(curve, "included_points", lambda: [])())
        if pts:
            i_mean = float(np.mean([float(p.intensity) for p in pts]))
            return float(coeffs[1] + 2.0 * coeffs[2] * i_mean)
    return float(coeffs[1])


def lod_loq(curve: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    IUPAC-style LOD / LOQ in wt% from calibration residual scatter.

    I_LOD = 3·s / |dC/dI|, C_LOD = |C(I_LOD) − C(0)| (same for LOQ with 10).
    Undefined when the slope is ~0 or there is no residual degrees of freedom.
    """
    if curve is None or not getattr(curve, "fitted", False):
        return None, None
    s = math.sqrt(max(float(getattr(curve, "residual_variance", 0.0) or 0.0), 0.0))
    dcdi = abs(curve_slope_at_reference(curve))
    pts = list(getattr(curve, "included_points", lambda: [])())
    model = str(getattr(curve, "model", "linear") or "linear")
    n_par = {"through_origin": 1, "quadratic": 3}.get(model, 2)
    if s <= 0 or dcdi < _SLOPE_EPS or len(pts) - n_par <= 0:
        return None, None
    i_lod = 3.0 * s / dcdi
    i_loq = 10.0 * s / dcdi
    evaluate = getattr(curve, "evaluate", None)
    if callable(evaluate):
        lod = abs(float(evaluate(i_lod)) - float(evaluate(0.0)))
        loq = abs(float(evaluate(i_loq)) - float(evaluate(0.0)))
        return lod, loq
    return 3.0 * s, 10.0 * s


def point_recovery(point: Any) -> Optional[float]:
    """Predicted / certified × 100 for one CRM point."""
    certified = getattr(point, "concentration", None)
    predicted = getattr(point, "predicted", None)
    if certified is None or predicted is None:
        return None
    try:
        certified_f = float(certified)
        predicted_f = float(predicted)
    except (TypeError, ValueError):
        return None
    if certified_f == 0 or not math.isfinite(certified_f) or not math.isfinite(predicted_f):
        return None
    return 100.0 * predicted_f / certified_f


def curve_recovery_stats(curve: Any) -> Dict[str, Optional[float]]:
    values = []
    for point in getattr(curve, "included_points", lambda: [])():
        rec = point_recovery(point)
        if rec is not None:
            values.append(rec)
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def standardized_residual(point: Any, sigma: float) -> Optional[float]:
    residual = getattr(point, "residual", None)
    if residual is None or sigma <= 0:
        return None
    try:
        return float(residual) / sigma
    except (TypeError, ValueError):
        return None


def flagged_residuals(curve: Any, z_thresh: float = Z_FLAG) -> List[Tuple[Any, float]]:
    sigma = math.sqrt(max(float(getattr(curve, "residual_variance", 0.0) or 0.0), 0.0))
    flagged = []
    for point in getattr(curve, "included_points", lambda: [])():
        z = standardized_residual(point, sigma)
        if z is not None and abs(z) > z_thresh:
            flagged.append((point, z))
    return flagged


def precision_stats(mean: float, std: float, n: int) -> Dict[str, Optional[float]]:
    """SEM and RSD% from a sample mean and SD."""
    if n < 2:
        return {"sem": None, "rsd": None}
    sem = float(std) / math.sqrt(n) if math.isfinite(std) else None
    rsd = None
    if math.isfinite(mean) and abs(mean) > 0 and math.isfinite(std):
        rsd = 100.0 * float(std) / abs(float(mean))
    return {"sem": sem, "rsd": rsd}


def fit_residual_summary(energy, residuals) -> Dict[str, Any]:
    """RMS residual, max |residual|, and a residual-vs-energy trend note."""
    empty = {"rms": None, "max_abs": None, "trend": "", "n": 0}
    if energy is None or residuals is None:
        return empty
    e = np.asarray(energy, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if e.size == 0 or r.size == 0 or e.size != r.size:
        return empty
    mask = np.isfinite(e) & np.isfinite(r)
    e, r = e[mask], r[mask]
    if r.size == 0:
        return empty
    rms = float(np.sqrt(np.mean(r ** 2)))
    max_abs = float(np.max(np.abs(r)))
    trend = ""
    if r.size >= 8 and float(np.std(e)) > 0 and float(np.std(r)) > 0:
        corr = float(np.corrcoef(e, r)[0, 1])
        if math.isfinite(corr):
            if abs(corr) >= 0.3:
                trend = (
                    f"Residuals correlate with energy (r = {corr:.2f}); "
                    "possible systematic mismatch."
                )
            else:
                trend = f"No strong residual-vs-energy trend (r = {corr:.2f})."
    return {"rms": rms, "max_abs": max_abs, "trend": trend, "n": int(r.size)}


# --------------------------------------------------------------------------- #
# Composition preparation
# --------------------------------------------------------------------------- #
def single_spectrum_row(context: ReportContext) -> Optional[CompositionRow]:
    """Build a CompositionRow from the current Analysis quantification."""
    relative: Dict[str, float] = {}
    wt: Dict[str, float] = {}
    formula_wt: Dict[str, float] = {}
    fp = context.fp_result
    if fp is not None:
        wt = {str(k): float(v) for k, v in dict(getattr(fp, "element_wt", None) or {}).items()}
        formula_wt = {
            str(k): float(v) for k, v in dict(getattr(fp, "formula_wt", None) or {}).items()
        }
    raw: Dict[str, float] = {}
    for key, val in (context.concentrations or {}).items():
        try:
            if isinstance(val, dict):
                raw[str(key)] = float(val.get("concentration", 0.0))
            else:
                raw[str(key)] = float(val)
        except (TypeError, ValueError):
            continue
    method = str(context.quantification_method or "")
    if method == "fp_matrix":
        if not wt:
            wt = dict(raw)
    elif method == "standards_curve":
        if not wt:
            wt = dict(raw)
    else:
        relative = dict(raw)
    if not relative and not wt and not formula_wt:
        return None
    name = Path(context.spectrum_path or "current spectrum").name
    values = dict(wt or relative or formula_wt)
    return CompositionRow(
        name=name,
        source_id=str(context.spectrum_path or name),
        sample=name,
        values=values,
        relative=relative,
        wt=wt,
        formula_wt=formula_wt,
    )


def prepared_rows(context: ReportContext, options: ReportOptions) -> List[CompositionRow]:
    """Copy composition rows and apply the selected value source."""
    if context.composition_rows:
        rows = [CompositionRow.from_dict(r.to_dict()) for r in context.composition_rows]
    else:
        row = single_spectrum_row(context)
        rows = [row] if row is not None else []
    if not rows:
        return []
    apply_value_source(
        rows,
        options.value_source,
        as_oxides=options.as_oxides,
        fe_as=options.fe_as,
    )
    return rows


def prepared_summaries(
    context: ReportContext, options: ReportOptions
) -> List[SampleSummary]:
    rows = prepared_rows(context, options)
    if not rows:
        return []
    return summarize_samples(rows)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def _esc(value: Any) -> str:
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _fmt_opt(value: Any, digits: int = 4, suffix: str = "") -> str:
    text = _fmt(value, digits)
    if text == "—":
        return text
    return f"{text}{suffix}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str = "") -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return (
        f"<table>{cap}<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _png_img(png: Optional[bytes], alt: str) -> str:
    if not png:
        return ""
    b64 = base64.b64encode(png).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{b64}" alt="{_esc(alt)}"/>'
        f"<figcaption>{_esc(alt)}</figcaption></figure>"
    )


def _kv_table(pairs: Sequence[Tuple[str, Any]]) -> str:
    safe = []
    for key, val in pairs:
        text = "—" if val is None or val == "" else _esc(val)
        safe.append((_esc(key), text))
    return _table(["Item", "Value"], safe)


def _maybe_plot(name: str, *args, **kwargs) -> Optional[bytes]:
    try:
        from core import report_plots
    except Exception:
        return None
    fn = getattr(report_plots, name, None)
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _spectrum_energy(context: ReportContext):
    spec = context.spectrum
    if spec is None:
        return None
    return getattr(spec, "energy", None)


def _fwhm_equation(cal: Any) -> str:
    model = str(getattr(cal, "model_type", "") or "")
    if model == "detector":
        return "FWHM(E) = √(FWHM₀² + 2.355² · ε · E)"
    if model == "linear":
        return "FWHM(E) = a + b·E"
    if model == "quadratic":
        return "FWHM(E) = a + b·E + c·E²"
    if model == "exponential":
        return "FWHM(E) = A · exp(b·E)"
    if model == "power":
        return "FWHM(E) = A · Eᵇ"
    return model or "—"


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def _header_html(context: ReportContext, options: ReportOptions) -> str:
    when = context.generated or datetime.now()
    method = METHOD_LABELS.get(
        context.quantification_method, context.quantification_method or "—"
    )
    matrix = context.matrix
    if matrix is not None and hasattr(matrix, "to_dict"):
        md = matrix.to_dict()
        matrix_txt = (
            f"{md.get('kind', '—')} (Fe as {md.get('fe_as', '—')}; "
            f"H₂O {md.get('h2o_wt', 0):.2f}, OH {md.get('oh_wt', 0):.2f}, "
            f"CO₂ {md.get('co2_wt', 0):.2f} wt%)"
        )
    else:
        matrix_txt = "—"
    version = options.app_version or app_version()
    pairs = [
        ("Software", f"XRFLab {version}"),
        ("Generated", when.strftime("%Y-%m-%d %H:%M")),
        ("Spectrum", context.spectrum_path or "—"),
        ("Project", context.project_path or "—"),
        ("Quantification", method),
        ("Matrix model", matrix_txt),
        ("Composition display", composition_unit_label(options)),
    ]
    return (
        f"<header><h1>{_esc(options.title)}</h1>"
        f"{_kv_table(pairs)}</header>"
    )


def _energy_axis_lines(context: ReportContext) -> List[str]:
    lines: List[str] = []
    spec = context.spectrum
    if spec is not None and hasattr(spec, "get_energy_calibration"):
        try:
            offset, gain = spec.get_energy_calibration()
            lines.append(f"E = {float(offset):.6g} + {float(gain):.6g}·channel keV")
        except Exception:
            pass
        md = getattr(spec, "metadata", None) or {}
        if isinstance(md, dict):
            if md.get("energy_offset_ev") is not None:
                lines.append(f"import offset {md['energy_offset_ev']} eV")
            if md.get("energy_calibration"):
                lines.append(str(md["energy_calibration"]))
    stats = getattr(context.fit_result, "statistics", None) or {}
    if stats.get("energy_offset_kev") is not None:
        try:
            ev = float(stats["energy_offset_kev"]) * 1000.0
            lines.append(f"grouped-fit energy offset {ev:.2f} eV")
        except (TypeError, ValueError):
            pass
    if stats.get("energy_gain") is not None:
        lines.append(f"grouped-fit gain × {stats['energy_gain']}")
    return lines


def _section_methods(context: ReportContext, options: ReportOptions) -> str:
    det = context.detector
    fwhm = context.fwhm_calibration or getattr(det, "fwhm_calibration", None)
    det_bits = []
    if det is not None:
        det_bits.append(f"FWHM₀ = {_fmt(getattr(det, 'fwhm_0', None), 4)} keV")
        det_bits.append(f"ε = {_fmt(getattr(det, 'epsilon', None), 5)}")
        if getattr(det, "use_calibrated_shapes", False):
            det_bits.append("calibrated peak shapes on")
    if fwhm is not None:
        det_bits.append(f"applied model: {getattr(fwhm, 'model_type', '—')}")
    spec = context.spectrum
    live = getattr(spec, "live_time", None) if spec is not None else None
    settings = context.fit_settings or {}
    std = context.standards_calibration
    if std is not None and getattr(std, "fit_settings", None):
        settings = {**settings, **dict(std.fit_settings)}
    energy_lines = _energy_axis_lines(context)
    pairs = [
        ("Detector", "; ".join(det_bits) if det_bits else "—"),
        ("Tube anode", context.tube_element or settings.get("tube_element") or "—"),
        ("Tube voltage", _fmt_opt(context.excitation_kv, 1, " kV")),
        ("Live time", _fmt_opt(live, 2, " s")),
        ("Energy axis", "; ".join(energy_lines) if energy_lines else "—"),
        ("Peak shape", settings.get("peak_shape") or "—"),
        ("Grouped lines", settings.get("grouped_lines", "—")),
        ("Background", settings.get("background_method") or "—"),
    ]
    return (
        f'<section id="section-methods"><h2>{SECTION_TITLES[SECTION_METHODS]}</h2>'
        f"{_kv_table(pairs)}</section>"
    )


def _section_fwhm(context: ReportContext, options: ReportOptions) -> str:
    cal = context.fwhm_calibration
    if cal is None:
        return ""
    params = dict(getattr(cal, "parameters", None) or {})
    errors = dict(getattr(cal, "parameter_errors", None) or {})
    param_rows = []
    for name, value in params.items():
        err = errors.get(name)
        err_txt = f" ± {_fmt(err, 5)}" if err not in (None, 0, 0.0) else ""
        param_rows.append((_esc(name), f"{_fmt(value, 6)}{err_txt}"))
    er = getattr(cal, "energy_range", None) or ("—", "—")
    pairs = [
        ("Model", getattr(cal, "model_type", "—")),
        ("Equation", _fwhm_equation(cal)),
        ("R²", _fmt(getattr(cal, "r_squared", None), 4)),
        ("RMSE", f"{_fmt(getattr(cal, 'rmse', None), 5)} keV"),
        ("AIC", _fmt(getattr(cal, "aic", None), 2)),
        ("BIC", _fmt(getattr(cal, "bic", None), 2)),
        ("n peaks", getattr(cal, "n_peaks", "—")),
        ("Energy range", f"{er[0]}–{er[1]} keV" if er else "—"),
        ("Date", getattr(cal, "calibration_date", None) or "—"),
    ]
    parts = [
        f'<section id="section-fwhm"><h2>{SECTION_TITLES[SECTION_FWHM]}</h2>',
        _kv_table(pairs),
    ]
    if param_rows:
        parts.append(_table(["Parameter", "Value"], param_rows, "Model parameters"))
    measurements = list(context.fwhm_measurements or [])
    if measurements:
        mrows = []
        for item in measurements:
            try:
                element = getattr(item, "element", None) or item.get("element")
                line = getattr(item, "line", None) or item.get("line")
                energy = getattr(item, "energy", None)
                if energy is None:
                    energy = item.get("energy")
                fwhm = getattr(item, "fwhm", None)
                if fwhm is None:
                    fwhm = item.get("fwhm")
                quality = getattr(item, "fit_quality", None)
                if quality is None:
                    quality = item.get("fit_quality")
            except AttributeError:
                continue
            mrows.append((
                _esc(element),
                _esc(line),
                _fmt(energy, 4),
                _fmt(float(fwhm) * 1000.0 if fwhm is not None else None, 2),
                _fmt(quality, 4),
            ))
        if mrows:
            parts.append(
                _table(
                    ["Element", "Line", "Energy (keV)", "FWHM (eV)", "Peak R²"],
                    mrows,
                    "This-session peak measurements",
                )
            )
    if options.include_plots:
        parts.append(
            _png_img(
                _maybe_plot("plot_fwhm_calibration", cal, measurements),
                "FWHM versus energy",
            )
        )
    parts.append("</section>")
    return "".join(parts)


def _section_tube(context: ReportContext, options: ReportOptions) -> str:
    library = context.tube_library
    if not has_tube_profiles(library):
        return ""
    profiles = getattr(library, "profiles", {}) or {}
    rows = []
    for key in sorted(profiles, key=lambda k: float(k) if str(k).replace(".", "", 1).isdigit() else 0):
        prof = profiles[key]
        ratios = getattr(prof, "line_ratios", None) or {}
        ratio_txt = ", ".join(
            f"{name}={float(val):.3f}" for name, val in sorted(ratios.items())
        )
        rows.append((
            _fmt(getattr(prof, "tube_kv", key), 1),
            _esc(getattr(prof, "tube_element", "")),
            _esc(getattr(prof, "source", "")),
            _esc(getattr(prof, "scatterer", "") or "—"),
            _esc(ratio_txt or "—"),
        ))
    return (
        f'<section id="section-tube"><h2>{SECTION_TITLES[SECTION_TUBE]}</h2>'
        f"<p>Anode: {_esc(getattr(library, 'tube_element', '—'))}.</p>"
        f"{_table(['kV', 'Anode', 'Source', 'Scatterer', 'Line ratios'], rows)}"
        "</section>"
    )


def _enabled_curves(std: Any) -> Dict[str, Any]:
    out = {}
    for name, curve in dict(getattr(std, "curves", None) or {}).items():
        if getattr(curve, "fitted", False) and getattr(curve, "enabled", True):
            out[name] = curve
    return out


def _section_standards(context: ReportContext, options: ReportOptions) -> str:
    std = context.standards_calibration
    if not has_standards_curves(std):
        return ""
    curves = _enabled_curves(std)
    summary_rows = []
    point_blocks = []
    for name, curve in sorted(curves.items()):
        lod, loq = lod_loq(curve)
        rec = curve_recovery_stats(curve)
        coeffs = list(getattr(curve, "coefficients", None) or [])
        errs = list(getattr(curve, "coefficient_errors", None) or [])
        coef_txt = ", ".join(
            f"{_fmt(c, 5)}" + (f" ± {_fmt(errs[i], 5)}" if i < len(errs) else "")
            for i, c in enumerate(coeffs)
        )
        summary_rows.append((
            _esc(curve.element),
            _esc(getattr(curve, "line_group", "")),
            _esc(getattr(curve, "model", "")),
            _esc(coef_txt),
            _fmt(getattr(curve, "r_squared", None), 4),
            _fmt(getattr(curve, "rmse", None), 4),
            _esc(getattr(curve, "n_standards", "")),
            _fmt(getattr(curve, "mean_rsd_percent", None), 2),
            _fmt(lod, 4),
            _fmt(loq, 4),
            _fmt(rec["mean"], 1),
        ))
        sigma = math.sqrt(max(float(getattr(curve, "residual_variance", 0.0) or 0.0), 0.0))
        prow = []
        for point in getattr(curve, "included_points", lambda: [])():
            z = standardized_residual(point, sigma)
            flag = " yes" if z is not None and abs(z) > Z_FLAG else ""
            prow.append((
                _esc(getattr(point, "standard", "")),
                _fmt(getattr(point, "concentration", None), 4),
                _fmt(getattr(point, "predicted", None), 4),
                _fmt(getattr(point, "residual", None), 4),
                _fmt(point_recovery(point), 1),
                f"{_fmt(z, 2)}{flag}",
            ))
        if prow:
            point_blocks.append(
                f"<h3>{_esc(curve.element)} calibration points</h3>"
                + _table(
                    [
                        "Standard",
                        "Certified (wt%)",
                        "Predicted (wt%)",
                        "Residual (wt%)",
                        "Recovery (%)",
                        "z-residual",
                    ],
                    prow,
                )
            )
    parts = [
        f'<section id="section-standards"><h2>{SECTION_TITLES[SECTION_STANDARDS]}</h2>',
        "<p>LOD / LOQ are 3·s and 10·s mapped through |dC/dI| from the "
        "curve residual scatter (wt%). Recovery is predicted / certified × 100. "
        "z-residual uses s = √residual variance; |z| &gt; 3 is flagged.</p>",
        _table(
            [
                "Element",
                "Lines",
                "Model",
                "Coefficients ± σ",
                "R²",
                "RMSE (wt%)",
                "n",
                "Mean RSD %",
                "LOD (wt%)",
                "LOQ (wt%)",
                "Mean recovery %",
            ],
            summary_rows,
            "Calibration curves",
        ),
        "".join(point_blocks),
    ]
    if options.include_plots:
        parts.append(
            _png_img(
                _maybe_plot("plot_predicted_vs_certified", curves),
                "Predicted versus certified wt%",
            )
        )
        n_extra = max(0, len(curves) - CURVE_PLOT_CAP)
        note = (
            f"<p>{n_extra} additional enabled curve(s) are table-only.</p>"
            if n_extra
            else ""
        )
        parts.append(
            _png_img(
                _maybe_plot("plot_calibration_curves", curves, max_n=CURVE_PLOT_CAP),
                "Intensity versus certified wt%",
            )
            + note
        )
    parts.append("</section>")
    return "".join(parts)


def _section_fit(context: ReportContext, options: ReportOptions) -> str:
    fit = context.fit_result
    if fit is None:
        return ""
    stats = dict(getattr(fit, "statistics", None) or {})
    resid = fit_residual_summary(_spectrum_energy(context), getattr(fit, "residuals", None))
    pairs = [
        ("χ²", _fmt(stats.get("chi_squared"), 3)),
        ("χ²ᵣ", _fmt(stats.get("reduced_chi_squared"), 3)),
        ("R²", _fmt(stats.get("r_squared"), 4)),
        ("Degrees of freedom", stats.get("dof", "—")),
        ("Fit mode", stats.get("fit_mode", "—")),
        ("Energy offset (keV)", _fmt(stats.get("energy_offset_kev"), 5)),
        ("Energy gain", _fmt(stats.get("energy_gain"), 6)),
        ("RMS residual (counts)", _fmt(resid["rms"], 2)),
        ("Max |residual| (counts)", _fmt(resid["max_abs"], 2)),
        ("Residual trend", resid["trend"] or "—"),
    ]
    peaks = list(getattr(fit, "peaks", None) or [])
    peak_rows = []
    for peak in peaks:
        peak_rows.append((
            _esc(getattr(peak, "element", None) or ""),
            _esc(getattr(peak, "line", None) or ""),
            _fmt(getattr(peak, "energy", None), 4),
            _fmt(getattr(peak, "area", None), 1),
            _fmt(getattr(peak, "fwhm", None), 4),
            "tube" if getattr(peak, "is_tube_line", False) else "",
        ))
    flags = list(getattr(fit, "tube_overlap_flags", None) or [])
    flag_html = ""
    if flags:
        items = []
        for flag in flags:
            if isinstance(flag, dict):
                items.append(_esc(flag.get("message") or flag.get("line") or flag))
            else:
                items.append(_esc(flag))
        flag_html = "<h3>Tube-overlap warnings</h3><ul>" + "".join(
            f"<li>{item}</li>" for item in items
        ) + "</ul>"
    parts = [
        f'<section id="section-fit"><h2>{SECTION_TITLES[SECTION_FIT]}</h2>',
        _kv_table(pairs),
    ]
    if peak_rows:
        parts.append(
            _table(
                ["Element", "Line", "Energy (keV)", "Area", "FWHM (keV)", "Note"],
                peak_rows,
                "Fitted peaks",
            )
        )
    parts.append(flag_html)
    if options.include_plots:
        spec = context.spectrum
        energy = getattr(spec, "energy", None) if spec is not None else None
        counts = getattr(spec, "counts", None) if spec is not None else None
        parts.append(
            _png_img(
                _maybe_plot(
                    "plot_spectrum_fit",
                    energy,
                    counts,
                    getattr(fit, "fitted_spectrum", None),
                    getattr(fit, "background", None),
                ),
                "Measured spectrum and fit",
            )
        )
        parts.append(
            _png_img(
                _maybe_plot("plot_fit_residuals", energy, getattr(fit, "residuals", None)),
                "Fit residuals versus energy",
            )
        )
    parts.append("</section>")
    return "".join(parts)


def _section_composition(context: ReportContext, options: ReportOptions) -> str:
    if not _has_composition(context):
        return ""
    kw = display_kwargs(options)
    unit = composition_unit_label(options)
    summaries = prepared_summaries(context, options)
    rows = prepared_rows(context, options)
    parts = [
        f'<section id="section-composition"><h2>{SECTION_TITLES[SECTION_COMPOSITION]}</h2>',
        f"<p>Values are shown as <strong>{_esc(unit)}</strong> "
        f"(source: {'FP wt%' if options.value_source == VALUE_WT else 'relative intensity'}"
        f"{'; oxides' if options.as_oxides else ''}"
        f"{'; closed to 100%' if options.close else ''}).</p>",
    ]
    keys: List[str] = []
    if summaries:
        keys = component_keys(summaries, **kw)
        header = ["Sample", "n"]
        for key in keys:
            header.extend([key, f"{key} SD", f"{key} SEM", f"{key} RSD%"])
        body = []
        for summary in summaries:
            vals = display_values(summary, **kw)
            std_vals = convert_values(summary.std, as_oxides=kw["as_oxides"], fe_as=kw["fe_as"], close=False)
            cells: List[str] = [_esc(summary.sample), _esc(summary.n)]
            for key in keys:
                mean = float(vals.get(key, 0.0) or 0.0)
                std = float(std_vals.get(key, 0.0) or 0.0)
                prec = precision_stats(mean, std, summary.n)
                cells.extend([
                    _fmt(mean, 4),
                    _fmt(std, 4) if summary.n >= 2 else "—",
                    _fmt(prec["sem"], 4) if summary.n >= 2 else "—",
                    _fmt(prec["rsd"], 2) if summary.n >= 2 else "—",
                ])
            body.append(cells)
        parts.append(_table(header, body, f"Sample means ({unit})"))
        if options.include_replicates and rows and any(s.n > 1 for s in summaries):
            rhead = ["Spectrum", "Sample"] + keys
            rbody = []
            for row in rows:
                vals = convert_values(row.values, **kw)
                rbody.append(
                    [_esc(row.name), _esc(row.sample)]
                    + [_fmt(vals.get(key, 0.0), 4) for key in keys]
                )
            parts.append(_table(rhead, rbody, f"Replicate spectra ({unit})"))
        if options.include_plots and keys:
            series = [
                {"name": s.sample, "values": display_values(s, **kw)}
                for s in summaries
            ]
            plot_keys = keys[:12]
            parts.append(
                _png_img(
                    _maybe_plot(
                        "plot_composition_bars",
                        series,
                        plot_keys,
                        ylabel=unit,
                        title="Sample means",
                    ),
                    f"Composition ({unit})",
                )
            )
    parts.append("</section>")
    return "".join(parts)


def _section_errors(context: ReportContext, options: ReportOptions) -> str:
    bullets: List[str] = []
    fit = context.fit_result
    if fit is not None:
        stats = dict(getattr(fit, "statistics", None) or {})
        resid = fit_residual_summary(_spectrum_energy(context), getattr(fit, "residuals", None))
        chi_r = stats.get("reduced_chi_squared")
        r2 = stats.get("r_squared")
        note = f"Spectrum fit: χ²ᵣ = {_fmt(chi_r, 3)}, R² = {_fmt(r2, 4)}"
        if resid["rms"] is not None:
            note += f", RMS residual = {_fmt(resid['rms'], 1)} counts"
        if resid["trend"]:
            note += f". {resid['trend']}"
        bullets.append(note)
        flags = list(getattr(fit, "tube_overlap_flags", None) or [])
        if flags:
            bullets.append(f"{len(flags)} tube-overlap flag(s) on the current fit.")

    std = context.standards_calibration
    if has_standards_curves(std):
        flagged_all = []
        recs = []
        lods = []
        for curve in _enabled_curves(std).values():
            rec = curve_recovery_stats(curve)
            if rec["mean"] is not None:
                recs.append((curve.element, rec["mean"], rec["min"], rec["max"]))
            lod, loq = lod_loq(curve)
            if lod is not None:
                lods.append((curve.element, lod, loq))
            for point, z in flagged_residuals(curve):
                flagged_all.append((curve.element, getattr(point, "standard", ""), z))
        if recs:
            bits = ", ".join(
                f"{el} {_fmt(mean, 1)}% [{_fmt(lo, 1)}–{_fmt(hi, 1)}]"
                for el, mean, lo, hi in recs
            )
            bullets.append(f"Standards mean recovery: {bits}.")
        if lods:
            bits = ", ".join(
                f"{el} LOD {_fmt(lod, 4)} / LOQ {_fmt(loq, 4)} wt%"
                for el, lod, loq in lods
            )
            bullets.append(f"Calibration detection limits: {bits}.")
        if flagged_all:
            bits = ", ".join(
                f"{el}/{stdn} z={_fmt(z, 2)}" for el, stdn, z in flagged_all
            )
            bullets.append(f"Flagged calibration residuals (|z| &gt; {Z_FLAG:g}): {bits}.")
        else:
            bullets.append("No calibration points with |z-residual| &gt; 3.")

    fp = context.fp_result
    if fp is not None:
        residual = getattr(fp, "residual", None)
        iters = getattr(fp, "iterations", None)
        bullets.append(
            f"FP iteration residual = {_fmt(residual, 4)} after {iters or '—'} iteration(s)."
        )
        warnings = list(getattr(fp, "line_warnings", None) or [])
        if warnings:
            bullets.append(f"{len(warnings)} FP line-consistency warning(s).")
            for warn in warnings[:12]:
                bullets.append(_esc(warn))
        outliers = []
        conc = getattr(fp, "concentrations", None) or {}
        for el, data in conc.items():
            if not isinstance(data, dict):
                continue
            for check in data.get("line_checks") or []:
                if check.get("flag"):
                    outliers.append(
                        f"{el} {check.get('line', '')} "
                        f"obs/pred = {_fmt(check.get('ratio'), 2)}"
                    )
        if outliers:
            bullets.append("FP line-check outliers: " + "; ".join(outliers) + ".")

    summaries = prepared_summaries(context, options) if _has_composition(context) else []
    high_rsd = []
    for summary in summaries:
        if summary.n < 2:
            continue
        kw = display_kwargs(options)
        vals = display_values(summary, **kw)
        std_vals = convert_values(
            summary.std, as_oxides=kw["as_oxides"], fe_as=kw["fe_as"], close=False
        )
        for key, mean in vals.items():
            prec = precision_stats(float(mean), float(std_vals.get(key, 0.0) or 0.0), summary.n)
            if prec["rsd"] is not None and prec["rsd"] > 10.0:
                high_rsd.append(f"{summary.sample} {key} RSD {_fmt(prec['rsd'], 1)}%")
    if high_rsd:
        bullets.append("Replicate RSD &gt; 10%: " + "; ".join(high_rsd) + ".")
    elif any(s.n >= 2 for s in summaries):
        bullets.append("All replicate RSDs are ≤ 10% for displayed components.")

    if not bullets:
        return ""
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return (
        f'<section id="section-errors"><h2>{SECTION_TITLES[SECTION_ERRORS]}</h2>'
        f"<ul>{items}</ul></section>"
    )


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light; }
html { font-size: 11pt; }
body {
  font-family: "Source Sans 3", "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: #1a1a1a;
  max-width: 960px;
  margin: 1.5rem auto;
  padding: 0 1.2rem 3rem;
  line-height: 1.45;
}
h1 { font-size: 1.7rem; margin-bottom: 0.4rem; }
h2 { font-size: 1.25rem; margin-top: 0; }
h3 { font-size: 1.05rem; }
header { margin-bottom: 1.5rem; }
section { margin: 1.6rem 0; }
table { border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.2rem; font-size: 0.92rem; }
th, td { border: 1px solid #ccc; padding: 0.28rem 0.45rem; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
caption { caption-side: top; font-weight: 600; text-align: left; margin-bottom: 0.3rem; }
figure { margin: 1rem 0; }
img { max-width: 100%; height: auto; }
figcaption { font-size: 0.9rem; color: #444; margin-top: 0.3rem; }
@media print {
  @page { size: A4; margin: 14mm; }
  body { max-width: none; margin: 0; padding: 0; }
  section + section { break-before: page; page-break-before: always; }
  h1, h2, h3 { break-after: avoid; }
  figure, table { break-inside: avoid; }
  tr { break-inside: avoid; }
}
"""


_SECTION_BUILDERS = {
    SECTION_METHODS: _section_methods,
    SECTION_FWHM: _section_fwhm,
    SECTION_TUBE: _section_tube,
    SECTION_STANDARDS: _section_standards,
    SECTION_FIT: _section_fit,
    SECTION_COMPOSITION: _section_composition,
    SECTION_ERRORS: _section_errors,
}


def build_report_html(context: ReportContext, options: ReportOptions) -> str:
    """Render a full HTML document for the selected sections."""
    if not options.app_version:
        options.app_version = app_version()
    if context.generated is None:
        context.generated = datetime.now()
    chunks = [_header_html(context, options)]
    for section in ALL_SECTIONS:
        if not options.wants(section):
            continue
        builder = _SECTION_BUILDERS[section]
        chunks.append(builder(context, options))
    body = "\n".join(c for c in chunks if c)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        f"<title>{_esc(options.title)}</title>"
        f"<style>{_CSS}</style></head><body>\n{body}\n</body></html>\n"
    )


def write_report(path, context: ReportContext, options: ReportOptions) -> Path:
    """Write a self-contained HTML report and return the path."""
    dest = Path(path)
    dest.write_text(build_report_html(context, options), encoding="utf-8")
    return dest
