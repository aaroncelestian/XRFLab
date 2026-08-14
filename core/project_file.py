"""
XRFLab project file (.xrfp): a self-contained HDF5 snapshot of the workspace.

The file embeds spectra, fits, maps, cubes, images, batch results, composition
rows, calibrations, and UI state so a project can be reopened without the
original .ipj / spectrum files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FORMAT_ID = "XRFLAB_PROJECT"
FORMAT_VERSION = 1
FILE_FILTER = "XRFLab Project (*.xrfp);;All Files (*)"

_GZIP = dict(compression="gzip", compression_opts=4, shuffle=True)


class ProjectFileError(ValueError):
    """Invalid or unsupported .xrfp file."""


# ------------------------------------------------------------------ JSON
def jsonable(obj: Any) -> Any:
    """Convert nested objects into JSON-serializable Python types."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and not np.isfinite(val):
            return None
        return val
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    if hasattr(obj, "value") and not callable(obj.value):
        try:
            return jsonable(obj.value)
        except Exception:
            pass
    return str(obj)


def _write_json(group, name: str, obj: Any) -> None:
    import h5py

    payload = json.dumps(jsonable(obj), ensure_ascii=False)
    dt = h5py.string_dtype(encoding="utf-8")
    if name in group:
        del group[name]
    group.create_dataset(name, data=payload, dtype=dt)


def _read_json(group, name: str, default=None):
    if name not in group:
        return default
    raw = group[name][()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    elif isinstance(raw, np.bytes_):
        text = raw.decode("utf-8")
    elif isinstance(raw, str):
        text = raw
    else:
        text = bytes(raw).decode("utf-8")
    if not text:
        return default
    return json.loads(text)


def _write_array(group, name: str, arr, *, optional: bool = True) -> None:
    if arr is None:
        if not optional:
            raise ValueError(f"required array {name} is None")
        return
    data = np.asarray(arr)
    if name in group:
        del group[name]
    kwargs = dict(_GZIP) if data.size else {}
    if data.ndim >= 2 and data.size:
        chunks = tuple(max(1, min(int(s), 64)) for s in data.shape)
        kwargs["chunks"] = chunks
    group.create_dataset(name, data=data, **kwargs)


def _read_array(group, name: str, default=None):
    if name not in group:
        return default
    return np.array(group[name])


def _set_attr(obj, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, dict)):
        return
    try:
        obj.attrs[key] = value
    except (TypeError, ValueError):
        obj.attrs[key] = str(value)


# -------------------------------------------------------------- spectrum
def _write_spectrum(group, spectrum) -> None:
    _write_array(group, "energy", spectrum.energy)
    _write_array(group, "counts", spectrum.counts)
    group.attrs["live_time"] = float(spectrum.live_time)
    group.attrs["real_time"] = float(spectrum.real_time)
    _write_json(group, "metadata", spectrum.metadata or {})


def _read_spectrum(group):
    from core.spectrum import Spectrum

    return Spectrum(
        energy=np.asarray(group["energy"], dtype=np.float64),
        counts=np.asarray(group["counts"], dtype=np.float64),
        live_time=float(group.attrs.get("live_time", 100.0)),
        real_time=float(group.attrs.get("real_time", 100.0)),
        metadata=_read_json(group, "metadata", {}) or {},
    )


# ---------------------------------------------------------------- mapping
def _write_element_map(group, em) -> None:
    group.attrs["name"] = em.name
    group.attrs["line"] = em.line or ""
    group.attrs["element"] = em.element or ""
    _write_array(group, "data", em.data)
    _write_json(group, "metadata", em.metadata or {})


def _read_element_map(group):
    from core.mapping.models import ElementMap

    return ElementMap(
        name=str(group.attrs.get("name", "")),
        data=np.array(group["data"]),
        line=str(group.attrs.get("line", "")),
        element=str(group.attrs.get("element", "")),
        metadata=_read_json(group, "metadata", {}) or {},
    )


def _write_overview(group, image) -> None:
    if image is None:
        return
    group.attrs["name"] = image.name
    _write_array(group, "data", image.data)
    _write_json(group, "metadata", image.metadata or {})


def _read_overview(group):
    from core.mapping.models import OverviewImage

    if group is None or "data" not in group:
        return None
    return OverviewImage(
        name=str(group.attrs.get("name", "")),
        data=np.array(group["data"]),
        metadata=_read_json(group, "metadata", {}) or {},
    )


def _write_map_spectrum(group, ms) -> None:
    group.attrs["name"] = ms.name or ""
    group.attrs["kind"] = ms.kind or "spot"
    if ms.x is not None:
        group.attrs["x"] = float(ms.x)
    if ms.y is not None:
        group.attrs["y"] = float(ms.y)
    if ms.index is not None:
        group.attrs["index"] = int(ms.index)
    _write_spectrum(group.create_group("spectrum"), ms.spectrum)
    _write_json(group, "peak_labels", ms.peak_labels or [])
    _write_json(group, "metadata", ms.metadata or {})


def _read_map_spectrum(group):
    from core.mapping.models import MapSpectrum

    x = group.attrs.get("x")
    y = group.attrs.get("y")
    index = group.attrs.get("index")
    return MapSpectrum(
        spectrum=_read_spectrum(group["spectrum"]),
        name=str(group.attrs.get("name", "")),
        x=None if x is None else float(x),
        y=None if y is None else float(y),
        index=None if index is None else int(index),
        kind=str(group.attrs.get("kind", "spot")),
        peak_labels=list(_read_json(group, "peak_labels", []) or []),
        metadata=_read_json(group, "metadata", {}) or {},
    )


def _write_line_scan(group, ls) -> None:
    group.attrs["name"] = ls.name or ""
    group.attrs["source"] = ls.source or "ipj"
    group.attrs["kind"] = ls.kind or "line_scan"
    if ls.start_xy is not None:
        group.attrs["start_xy"] = np.asarray(ls.start_xy, dtype=np.float64)
    if ls.end_xy is not None:
        group.attrs["end_xy"] = np.asarray(ls.end_xy, dtype=np.float64)
    _write_json(group, "metadata", ls.metadata or {})
    pts = group.create_group("points")
    for i, pt in enumerate(ls.points or []):
        _write_map_spectrum(pts.create_group(f"{i:04d}"), pt)


def _read_xy_attr(attrs, key) -> Optional[Tuple[float, float]]:
    if key not in attrs:
        return None
    val = np.asarray(attrs[key], dtype=np.float64).reshape(-1)
    if val.size < 2:
        return None
    return (float(val[0]), float(val[1]))


def _read_line_scan(group):
    from core.mapping.models import LineScan

    pts_grp = group["points"] if "points" in group else None
    points = []
    if pts_grp is not None:
        for key in sorted(pts_grp.keys()):
            points.append(_read_map_spectrum(pts_grp[key]))
    return LineScan(
        name=str(group.attrs.get("name", "")),
        points=points,
        start_xy=_read_xy_attr(group.attrs, "start_xy"),
        end_xy=_read_xy_attr(group.attrs, "end_xy"),
        source=str(group.attrs.get("source", "ipj")),
        kind=str(group.attrs.get("kind", "line_scan")),
        metadata=_read_json(group, "metadata", {}) or {},
    )


def _write_cube(group, cube) -> None:
    if cube is None:
        return
    _write_array(group, "data", cube.data)
    group.attrs["ev_per_channel"] = float(cube.ev_per_channel)
    group.attrs["energy_offset_ev"] = float(cube.energy_offset_ev)


def _read_cube(group):
    from core.mapping.cube import SpectrumCube

    if group is None or "data" not in group:
        return None
    return SpectrumCube(
        data=np.array(group["data"]),
        ev_per_channel=float(group.attrs.get("ev_per_channel", 10.0)),
        energy_offset_ev=float(group.attrs.get("energy_offset_ev", 0.0)),
    )


def _write_fov(group, fov) -> None:
    group.attrs["id"] = fov.id
    group.attrs["name"] = fov.name or ""
    group.attrs["width"] = int(fov.width or 0)
    group.attrs["height"] = int(fov.height or 0)
    if fov.pixel_size_mm is not None:
        group.attrs["pixel_size_mm"] = float(fov.pixel_size_mm)
    if fov.stage_center_mm is not None:
        group.attrs["stage_center_mm"] = np.asarray(
            fov.stage_center_mm, dtype=np.float64
        )
    _write_json(group, "metadata", fov.metadata or {})
    maps = group.create_group("element_maps")
    for i, em in enumerate(fov.element_maps or []):
        _write_element_map(maps.create_group(f"{i:04d}"), em)
    if fov.overview is not None:
        _write_overview(group.create_group("overview"), fov.overview)
    if fov.optical is not None:
        _write_overview(group.create_group("optical"), fov.optical)
    specs = group.create_group("spectra")
    for i, ms in enumerate(fov.spectra or []):
        _write_map_spectrum(specs.create_group(f"{i:04d}"), ms)
    scans = group.create_group("line_scans")
    for i, ls in enumerate(fov.line_scans or []):
        _write_line_scan(scans.create_group(f"{i:04d}"), ls)
    if fov.cube is not None:
        _write_cube(group.create_group("cube"), fov.cube)


def _read_fov(group):
    from core.mapping.models import MappingFOV

    maps = []
    if "element_maps" in group:
        mg = group["element_maps"]
        for key in sorted(mg.keys()):
            maps.append(_read_element_map(mg[key]))
    spectra = []
    if "spectra" in group:
        sg = group["spectra"]
        for key in sorted(sg.keys()):
            spectra.append(_read_map_spectrum(sg[key]))
    line_scans = []
    if "line_scans" in group:
        lg = group["line_scans"]
        for key in sorted(lg.keys()):
            line_scans.append(_read_line_scan(lg[key]))
    center = _read_xy_attr(group.attrs, "stage_center_mm")
    px = group.attrs.get("pixel_size_mm")
    return MappingFOV(
        id=str(group.attrs.get("id", "")),
        name=str(group.attrs.get("name", "")),
        width=int(group.attrs.get("width", 0)),
        height=int(group.attrs.get("height", 0)),
        element_maps=maps,
        overview=_read_overview(group["overview"]) if "overview" in group else None,
        optical=_read_overview(group["optical"]) if "optical" in group else None,
        spectra=spectra,
        line_scans=line_scans,
        cube=_read_cube(group["cube"]) if "cube" in group else None,
        pixel_size_mm=None if px is None else float(px),
        stage_center_mm=center,
        metadata=_read_json(group, "metadata", {}) or {},
    )


def _write_sample(group, sample) -> None:
    group.attrs["id"] = sample.id
    group.attrs["name"] = sample.name or ""
    _write_json(group, "metadata", sample.metadata or {})
    if sample.whole_image is not None:
        _write_overview(group.create_group("whole_image"), sample.whole_image)
    sites = group.create_group("sites")
    for i, site in enumerate(sample.sites or []):
        _write_fov(sites.create_group(f"{i:04d}"), site)


def _read_sample(group):
    from core.mapping.models import MappingSample

    sites = []
    if "sites" in group:
        sg = group["sites"]
        for key in sorted(sg.keys()):
            sites.append(_read_fov(sg[key]))
    return MappingSample(
        id=str(group.attrs.get("id", "")),
        name=str(group.attrs.get("name", "")),
        sites=sites,
        whole_image=(
            _read_overview(group["whole_image"]) if "whole_image" in group else None
        ),
        metadata=_read_json(group, "metadata", {}) or {},
    )


def write_mapping_project(group, project) -> None:
    group.attrs["path"] = project.path or ""
    group.attrs["name"] = project.name or ""
    _write_json(group, "metadata", project.metadata or {})
    samples = group.create_group("samples")
    for i, sample in enumerate(project.samples or []):
        _write_sample(samples.create_group(f"{i:04d}"), sample)


def read_mapping_project(group):
    from core.mapping.models import MappingProject

    samples = []
    if "samples" in group:
        sg = group["samples"]
        for key in sorted(sg.keys()):
            samples.append(_read_sample(sg[key]))
    return MappingProject(
        path=str(group.attrs.get("path", "")),
        name=str(group.attrs.get("name", "")),
        samples=samples,
        metadata=_read_json(group, "metadata", {}) or {},
    )


# ------------------------------------------------------------------ fit
def _write_fit_result(group, fit) -> None:
    _write_array(group, "background", fit.background)
    _write_array(group, "fitted_spectrum", fit.fitted_spectrum)
    _write_array(group, "residuals", fit.residuals)
    _write_json(group, "meta", fit.to_dict(include_arrays=False))


def _read_fit_result(group):
    from core.fitting import FitResult

    meta = _read_json(group, "meta", {}) or {}
    return FitResult.from_dict(
        meta,
        background=_read_array(group, "background"),
        fitted_spectrum=_read_array(group, "fitted_spectrum"),
        residuals=_read_array(group, "residuals"),
    )


# ----------------------------------------------------------------- batch
def _write_batch_result(group, result) -> None:
    group.attrs["spectrum_name"] = result.spectrum_name
    group.attrs["spectrum_path"] = result.spectrum_path
    group.attrs["fit_success"] = bool(result.fit_success)
    group.attrs["chi_squared"] = float(result.chi_squared)
    group.attrs["r_squared"] = float(result.r_squared)
    group.attrs["fit_time"] = float(result.fit_time)
    group.attrs["error_message"] = result.error_message or ""
    group.attrs["quantification_method"] = result.quantification_method or ""
    _write_json(group, "elements_found", list(result.elements_found or []))
    _write_json(group, "concentrations", result.concentrations or {})
    _write_json(group, "concentration_errors", result.concentration_errors or {})
    _write_json(group, "peak_areas", result.peak_areas or {})
    _write_array(group, "fitted_spectrum", result.fitted_spectrum)
    _write_array(group, "residuals", result.residuals)
    _write_array(group, "energy", result.energy)
    _write_array(group, "measured_counts", result.measured_counts)
    contrib = result.element_contributions or {}
    if contrib:
        cg = group.create_group("element_contributions")
        for name, arr in contrib.items():
            _write_array(cg, str(name), arr)


def _read_batch_result(group):
    from core.batch_processing import BatchFitResult

    contrib = None
    if "element_contributions" in group:
        contrib = {}
        cg = group["element_contributions"]
        for name in cg.keys():
            contrib[name] = np.array(cg[name])
    return BatchFitResult(
        spectrum_name=str(group.attrs.get("spectrum_name", "")),
        spectrum_path=str(group.attrs.get("spectrum_path", "")),
        fit_success=bool(group.attrs.get("fit_success", False)),
        chi_squared=float(group.attrs.get("chi_squared", 0.0)),
        r_squared=float(group.attrs.get("r_squared", 0.0)),
        elements_found=list(_read_json(group, "elements_found", []) or []),
        concentrations=dict(_read_json(group, "concentrations", {}) or {}),
        concentration_errors=dict(
            _read_json(group, "concentration_errors", {}) or {}
        ),
        peak_areas=dict(_read_json(group, "peak_areas", {}) or {}),
        fitted_spectrum=_read_array(group, "fitted_spectrum"),
        residuals=_read_array(group, "residuals"),
        energy=_read_array(group, "energy"),
        measured_counts=_read_array(group, "measured_counts"),
        element_contributions=contrib,
        fit_time=float(group.attrs.get("fit_time", 0.0)),
        error_message=str(group.attrs.get("error_message", "")),
        quantification_method=str(
            group.attrs.get("quantification_method", "semi_quant_area")
        ),
    )


def _batch_config_dict(config) -> dict:
    if config is None:
        return {}
    return {
        "elements": jsonable(getattr(config, "elements", []) or []),
        "excitation_energy": float(getattr(config, "excitation_energy", 20.0)),
        "tube_current": float(getattr(config, "tube_current", 1.0)),
        "live_time": float(getattr(config, "live_time", 30.0)),
        "incident_angle": float(getattr(config, "incident_angle", 45.0)),
        "takeoff_angle": float(getattr(config, "takeoff_angle", 45.0)),
        "background_method": str(getattr(config, "background_method", "snip")),
        "peak_shape": str(getattr(config, "peak_shape", "voigt")),
        "include_escape_peaks": bool(getattr(config, "include_escape_peaks", True)),
        "include_pileup": bool(getattr(config, "include_pileup", False)),
        "tube_element": str(getattr(config, "tube_element", "Rh")),
        "include_tube_lines": bool(getattr(config, "include_tube_lines", True)),
        "include_compton": bool(getattr(config, "include_compton", True)),
        "auto_find_peaks": bool(getattr(config, "auto_find_peaks", True)),
        "excitation_kv": float(getattr(config, "excitation_kv", 50.0)),
        "use_calibration": bool(getattr(config, "use_calibration", False)),
        "save_individual_fits": bool(getattr(config, "save_individual_fits", True)),
        "save_plots": bool(getattr(config, "save_plots", False)),
        "output_directory": (
            str(config.output_directory) if getattr(config, "output_directory", None) else None
        ),
    }


def _batch_config_from_dict(data: dict):
    from core.batch_processing import BatchProcessingConfig

    data = data or {}
    out_dir = data.get("output_directory")
    return BatchProcessingConfig(
        elements=list(data.get("elements") or []),
        excitation_energy=float(data.get("excitation_energy", 20.0)),
        tube_current=float(data.get("tube_current", 1.0)),
        live_time=float(data.get("live_time", 30.0)),
        incident_angle=float(data.get("incident_angle", 45.0)),
        takeoff_angle=float(data.get("takeoff_angle", 45.0)),
        background_method=str(data.get("background_method", "snip")),
        peak_shape=str(data.get("peak_shape", "voigt")),
        include_escape_peaks=bool(data.get("include_escape_peaks", True)),
        include_pileup=bool(data.get("include_pileup", False)),
        tube_element=str(data.get("tube_element", "Rh")),
        include_tube_lines=bool(data.get("include_tube_lines", True)),
        include_compton=bool(data.get("include_compton", True)),
        auto_find_peaks=bool(data.get("auto_find_peaks", True)),
        excitation_kv=float(data.get("excitation_kv", 50.0)),
        use_calibration=bool(data.get("use_calibration", False)),
        save_individual_fits=bool(data.get("save_individual_fits", True)),
        save_plots=bool(data.get("save_plots", False)),
        output_directory=Path(out_dir) if out_dir else None,
    )


# -------------------------------------------------------------- document
@dataclass
class ProjectDocument:
    """In-memory snapshot written to / read from a .xrfp file."""

    analysis: Dict[str, Any] = field(default_factory=dict)
    mapping_project: Any = None
    mapping_ui: Dict[str, Any] = field(default_factory=dict)
    drawn_line_scan: Any = None
    batch: Dict[str, Any] = field(default_factory=dict)
    composition: Dict[str, Any] = field(default_factory=dict)
    calibrations: Dict[str, Any] = field(default_factory=dict)
    window: Dict[str, Any] = field(default_factory=dict)


def save_project(path: str, document: ProjectDocument) -> None:
    """Write ``document`` to ``path`` atomically."""
    import h5py

    path = Path(path)
    if path.suffix.lower() != ".xrfp":
        path = path.with_suffix(".xrfp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format_id"] = FORMAT_ID
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()
            f.attrs["app"] = "XRFLab"

            _write_analysis(f.create_group("analysis"), document.analysis or {})

            mapping = f.create_group("mapping")
            mapping.attrs["present"] = document.mapping_project is not None
            if document.mapping_project is not None:
                write_mapping_project(
                    mapping.create_group("project"), document.mapping_project
                )
            _write_json(mapping, "ui", document.mapping_ui or {})
            if document.drawn_line_scan is not None:
                _write_line_scan(
                    mapping.create_group("drawn_line_scan"), document.drawn_line_scan
                )

            _write_batch(f.create_group("batch"), document.batch or {})
            _write_json(f.create_group("composition"), "state", document.composition or {})
            _write_json(
                f.create_group("calibrations"), "state", document.calibrations or {}
            )
            _write_json(f.create_group("window"), "state", document.window or {})

        tmp.replace(path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def load_project(path: str) -> ProjectDocument:
    """Load a .xrfp file into a ProjectDocument."""
    import h5py

    path = Path(path)
    if not path.is_file():
        raise ProjectFileError(f"Project file not found: {path}")

    with h5py.File(path, "r") as f:
        fmt = str(f.attrs.get("format_id", ""))
        if fmt != FORMAT_ID:
            raise ProjectFileError(
                f"Not an XRFLab project file (format_id={fmt!r})."
            )
        version = int(f.attrs.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise ProjectFileError(
                f"Project file version {version} is newer than this XRFLab "
                f"({FORMAT_VERSION}). Update XRFLab to open it."
            )

        analysis = _read_analysis(f["analysis"]) if "analysis" in f else {}
        mapping_project = None
        mapping_ui: Dict[str, Any] = {}
        drawn = None
        if "mapping" in f:
            mg = f["mapping"]
            if "project" in mg:
                mapping_project = read_mapping_project(mg["project"])
            mapping_ui = _read_json(mg, "ui", {}) or {}
            if "drawn_line_scan" in mg:
                drawn = _read_line_scan(mg["drawn_line_scan"])

        batch = _read_batch(f["batch"]) if "batch" in f else {}
        composition = {}
        if "composition" in f:
            composition = _read_json(f["composition"], "state", {}) or {}
        calibrations = {}
        if "calibrations" in f:
            calibrations = _read_json(f["calibrations"], "state", {}) or {}
        window = {}
        if "window" in f:
            window = _read_json(f["window"], "state", {}) or {}

    return ProjectDocument(
        analysis=analysis,
        mapping_project=mapping_project,
        mapping_ui=mapping_ui,
        drawn_line_scan=drawn,
        batch=batch,
        composition=composition,
        calibrations=calibrations,
        window=window,
    )


def _write_analysis(group, analysis: dict) -> None:
    spectrum = analysis.get("spectrum")
    if spectrum is not None:
        _write_spectrum(group.create_group("spectrum"), spectrum)
    if analysis.get("spectrum_path"):
        group.attrs["spectrum_path"] = str(analysis["spectrum_path"])
    group.attrs["quantification_method"] = str(
        analysis.get("quantification_method") or "semi_quant_area"
    )
    _write_json(group, "elements", analysis.get("elements") or [])
    _write_json(group, "concentrations", analysis.get("concentrations") or {})
    matrix = analysis.get("matrix")
    if matrix is not None and hasattr(matrix, "to_dict"):
        matrix = matrix.to_dict()
    _write_json(group, "matrix", matrix or {})
    fp = analysis.get("fp_result")
    if fp is not None and hasattr(fp, "to_dict"):
        fp = fp.to_dict()
    _write_json(group, "fp_result", fp)
    fit = analysis.get("fit_result")
    if fit is not None:
        _write_fit_result(group.create_group("fit_result"), fit)
    _write_json(group, "ui", analysis.get("ui") or {})


def _read_analysis(group) -> dict:
    from core.fp_quantification import FPQuantResult
    from core.matrix_model import MatrixAssumptions

    analysis: Dict[str, Any] = {
        "spectrum": _read_spectrum(group["spectrum"]) if "spectrum" in group else None,
        "spectrum_path": str(group.attrs.get("spectrum_path", "") or "") or None,
        "quantification_method": str(
            group.attrs.get("quantification_method", "semi_quant_area")
        ),
        "elements": list(_read_json(group, "elements", []) or []),
        "concentrations": dict(_read_json(group, "concentrations", {}) or {}),
        "matrix": MatrixAssumptions.from_dict(_read_json(group, "matrix", {}) or {}),
        "fp_result": FPQuantResult.from_dict(_read_json(group, "fp_result")),
        "fit_result": (
            _read_fit_result(group["fit_result"]) if "fit_result" in group else None
        ),
        "ui": _read_json(group, "ui", {}) or {},
    }
    return analysis


def _write_batch(group, batch: dict) -> None:
    _write_json(group, "file_paths", batch.get("file_paths") or [])
    _write_json(group, "config", batch.get("config") or {})
    _write_json(group, "ui", batch.get("ui") or {})
    mem = batch.get("memory_spectra") or {}
    if mem:
        mg = group.create_group("memory_spectra")
        for i, (name, spec) in enumerate(mem.items()):
            sg = mg.create_group(f"{i:04d}")
            sg.attrs["name"] = str(name)
            _write_spectrum(sg, spec)
    results = batch.get("results") or []
    if results:
        rg = group.create_group("results")
        for i, result in enumerate(results):
            _write_batch_result(rg.create_group(f"{i:04d}"), result)


def _read_batch(group) -> dict:
    memory = {}
    if "memory_spectra" in group:
        mg = group["memory_spectra"]
        for key in sorted(mg.keys()):
            sg = mg[key]
            name = str(sg.attrs.get("name", key))
            memory[name] = _read_spectrum(sg)
    results = []
    if "results" in group:
        rg = group["results"]
        for key in sorted(rg.keys()):
            results.append(_read_batch_result(rg[key]))
    return {
        "file_paths": list(_read_json(group, "file_paths", []) or []),
        "config": _read_json(group, "config", {}) or {},
        "ui": _read_json(group, "ui", {}) or {},
        "memory_spectra": memory,
        "results": results,
    }
