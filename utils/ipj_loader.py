"""
Loader for Oxford INCA / Horiba XGT .ipj project files (OLE compound documents).

Extracts spot/sum spectra, element intensity maps, SEI transmission overviews,
XGT optical camera BMPs (sample whole-image and map-area crop), ordered
point series classified as line scans (equal step) or multipoint (irregular
steps), and SmartMap ListData hyperspectral cubes.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.mapping.cube import decode_listdata
from core.mapping.models import (
    ElementMap,
    LineScan,
    MappingFOV,
    MappingProject,
    MappingSample,
    MapSpectrum,
    OverviewImage,
)
from core.spectrum import Spectrum

try:
    import olefile
except ImportError as exc:  # pragma: no cover
    olefile = None
    _OLE_IMPORT_ERROR = exc
else:
    _OLE_IMPORT_ERROR = None

# Micromap binary layout (INCA GreyImage/micromap)
_MICROMAP_HEADER = 32
_MICROMAP_TRAILER = 216
_SPECTRUM_HEADER = 10
# INCA/XGT Spectrum streams are 4096 × uint32 (10 eV/ch is common after fix;
# inference may select 20 eV from peak labels)
_DEFAULT_EV_PER_CHANNEL = 10.0
_DEFAULT_N_CHANNELS = 4096
# XGT2Data stage X/Y (float32) — observed across numbered spectra
_XGT2_STAGE_X_OFF = 154
_XGT2_STAGE_Y_OFF = 162
# Max relative deviation from median step for "equal spacing" line scans
_LINE_SCAN_STEP_REL_TOL = 0.15


def load_ipj(file_path: str | Path) -> MappingProject:
    """Load an .ipj file into a MappingProject."""
    if olefile is None:
        raise ImportError(
            "olefile is required to read .ipj files. "
            "Install with: pip install olefile"
        ) from _OLE_IMPORT_ERROR

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".ipj":
        raise ValueError(f"Expected .ipj file, got {path.suffix}")

    ole = olefile.OleFileIO(str(path))
    try:
        return _parse_ole(ole, path)
    finally:
        ole.close()


def _parse_ole(ole: "olefile.OleFileIO", path: Path) -> MappingProject:
    sample_ids = _child_storages(ole, ["Samples"])
    samples: List[MappingSample] = []
    all_fovs: List[MappingFOV] = []

    for sample_idx, sample_id in enumerate(sample_ids, start=1):
        sample_name = _read_display_name(
            ole,
            ["Samples", sample_id, "ISampleInfo"],
            preferred_prefixes=("Sample",),
            default=f"Sample {sample_idx}",
        )
        whole_image = _parse_xgt_bmp(
            ole,
            under=["Samples", sample_id],
            stream_name="XGT2 WholeImage",
            name="Sample camera",
            kind="whole_image",
        )
        sites: List[MappingFOV] = []
        fov_ids = _child_storages(ole, ["Samples", sample_id, "FOVs"])
        for site_idx, fov_id in enumerate(fov_ids, start=1):
            fov = _parse_fov(ole, sample_id, fov_id, site_index=site_idx)
            sites.append(fov)
            all_fovs.append(fov)

        samples.append(
            MappingSample(
                id=sample_id,
                name=sample_name,
                sites=sites,
                whole_image=whole_image,
                metadata={"path": f"Samples/{sample_id}"},
            )
        )

    # Promote numbered spectrum series into LineScan when appropriate
    for fov in all_fovs:
        _maybe_attach_line_scan(fov)

    n_cubes = sum(1 for f in all_fovs if f.cube is not None)
    project = MappingProject(
        path=str(path),
        name=path.stem,
        samples=samples,
        fovs=all_fovs,
        metadata={
            "format": "inca_ipj",
            "n_samples": len(samples),
            "n_fovs": len(all_fovs),
            "n_cubes": n_cubes,
            "smartmap_cube": "decoded" if n_cubes else "absent",
        },
    )
    return project


def _parse_fov(
    ole: "olefile.OleFileIO",
    sample_id: str,
    fov_id: str,
    site_index: int = 1,
) -> MappingFOV:
    base = ["Samples", sample_id, "FOVs", fov_id]
    site_name = _read_display_name(
        ole,
        base + ["IFOVInfo"],
        preferred_prefixes=("Site of Interest", "Site"),
        default=f"Site of Interest {site_index}",
    )
    element_maps = _parse_element_maps(ole, base)
    overview = _parse_overview(ole, base)
    optical = _parse_xgt_bmp(
        ole,
        under=base,
        stream_name="XGT2 MapAreaImage",
        name="Map area photo",
        kind="map_area",
    )
    spectra = _parse_spectra(ole, base)
    cube = _parse_smartmap_cube(ole, base, spectra)

    width = height = 0
    if cube is not None:
        width, height = cube.width, cube.height
    else:
        list_path = base + ["SmartMap", "ListData"]
        if ole.exists(list_path):
            raw = ole.openstream(list_path).read()
            if len(raw) >= 24:
                width, height = struct.unpack_from("<II", raw, 16)
    if (not width or not height) and element_maps:
        height, width = element_maps[0].shape
    if (not width or not height) and overview is not None:
        height, width = overview.data.shape[:2]

    # When vendor MAP* images are absent, expose a total-counts map from the cube
    if cube is not None and not element_maps:
        element_maps.append(
            ElementMap(
                name="Total counts (cube)",
                data=cube.data.sum(axis=0).astype(np.float64),
                line="Total",
                element="",
                metadata={"source": "cube_total"},
            )
        )

    has_smartmap = ole.exists(base + ["SmartMap"])
    return MappingFOV(
        id=fov_id,
        name=site_name,
        width=int(width),
        height=int(height),
        element_maps=element_maps,
        overview=overview,
        optical=optical,
        spectra=spectra,
        cube=cube,
        metadata={
            "sample_id": sample_id,
            "site_name": site_name,
            "has_smartmap": has_smartmap,
            "has_cube": cube is not None,
            "has_optical": optical is not None,
            "cube_shape": tuple(cube.shape) if cube is not None else None,
            "path": "/".join(base),
        },
    )


def _read_display_name(
    ole: "olefile.OleFileIO",
    path: Sequence[str],
    *,
    preferred_prefixes: Tuple[str, ...] = (),
    default: str = "",
) -> str:
    """Extract a human-readable name from ISampleInfo / IFOVInfo streams."""
    if not ole.exists(list(path)):
        return default
    raw = ole.openstream(list(path)).read()
    strings = [
        s.decode("ascii", errors="ignore").strip()
        for s in re.findall(rb"[\x20-\x7e]{3,60}", raw)
    ]
    for prefix in preferred_prefixes:
        for s in strings:
            if s.lower().startswith(prefix.lower()):
                return s
    # Prefer "Sample N" / "Site of Interest N" style tokens
    for s in strings:
        if re.match(r"^(Sample|Site of Interest)\s+\d+", s, flags=re.I):
            return s
    for s in strings:
        if s.lower() not in ("comment:", "default", "empty") and not s.startswith(
            ("Hor", "cls", "Whole")
        ):
            return s
    return default


def _parse_smartmap_cube(
    ole: "olefile.OleFileIO",
    fov_base: List[str],
    spectra: List[MapSpectrum],
):
    """Decode SmartMap/ListData when present; return SpectrumCube or None."""
    list_path = fov_base + ["SmartMap", "ListData"]
    if not ole.exists(list_path):
        return None
    raw = ole.openstream(list_path).read()
    if len(raw) < 64:
        return None

    expected = None
    for ms in spectra:
        if ms.kind == "sum" or "sum" in ms.name.lower():
            expected = ms.spectrum.counts
            break
    if expected is None and spectra:
        expected = spectra[0].spectrum.counts

    try:
        cube = decode_listdata(raw, expected_sum=expected)
    except Exception:
        return None

    # Match energy scale to sum spectrum when channel counts agree
    if expected is not None:
        sum_ms = next(
            (s for s in spectra if s.kind == "sum" or "sum" in s.name.lower()),
            spectra[0] if spectra else None,
        )
        if sum_ms is not None:
            ev = float(
                sum_ms.spectrum.metadata.get(
                    "ev_per_channel", _DEFAULT_EV_PER_CHANNEL
                )
            )
            if len(expected) == cube.n_channels:
                cube.ev_per_channel = ev
            elif len(expected) == cube.n_channels * 2:
                cube.ev_per_channel = ev * 2.0
    return cube


def _parse_element_maps(
    ole: "olefile.OleFileIO",
    fov_base: List[str],
) -> List[ElementMap]:
    images_base = fov_base + ["Images"]
    maps: List[ElementMap] = []
    for img_id in _child_storages(ole, images_base):
        if not img_id.startswith("MAP"):
            continue
        label = _read_label(ole, images_base + [img_id, "GreyImage", "label"])
        line_name = label.get("title") or img_id
        mm_path = images_base + [img_id, "GreyImage", "micromap"]
        if not ole.exists(mm_path):
            continue
        data, meta = _read_micromap(ole.openstream(mm_path).read())
        maps.append(
            ElementMap(
                name=line_name,
                data=data,
                line=line_name,
                metadata={"image_id": img_id, **meta, **label},
            )
        )
    maps.sort(key=lambda m: m.name)
    return maps


def _parse_xgt_bmp(
    ole: "olefile.OleFileIO",
    *,
    under: Sequence[str],
    stream_name: str,
    name: str,
    kind: str,
) -> Optional[OverviewImage]:
    """Decode an XGT-wrapped 24/32-bit BMP (WholeImage or MapAreaImage)."""
    path = _find_stream(ole, stream_name, under=under)
    if path is None:
        return None
    try:
        raw = ole.openstream(path).read()
        data, bmp_meta = decode_embedded_bmp(raw)
    except Exception:
        return None
    if data.size == 0:
        return None
    return OverviewImage(
        name=name,
        data=data,
        metadata={
            "source": "xgt_bmp",
            "kind": kind,
            "path": "/".join(path),
            **bmp_meta,
        },
    )


def decode_embedded_bmp(raw: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decode a Windows BMP embedded in an XGT ExtraData stream.

    The stream has a short vendor header; the BMP starts at the first 'BM'.
    Returns RGB uint8 array (H, W, 3), origin at top-left.
    """
    bm = raw.find(b"BM")
    if bm < 0 or bm + 54 > len(raw):
        raise ValueError("no BMP signature in stream")
    bmp = raw[bm:]
    pixel_off = struct.unpack_from("<I", bmp, 10)[0]
    dib = struct.unpack_from("<I", bmp, 14)[0]
    if dib < 16 or pixel_off < 14 or pixel_off > len(bmp):
        raise ValueError(f"invalid BMP header dib={dib} pixel_off={pixel_off}")
    width = struct.unpack_from("<i", bmp, 18)[0]
    height_signed = struct.unpack_from("<i", bmp, 22)[0]
    planes, bpp = struct.unpack_from("<HH", bmp, 26)
    compression = (
        struct.unpack_from("<I", bmp, 30)[0]
        if dib >= 20 and len(bmp) >= 34
        else 0
    )
    bottom_up = height_signed > 0
    height = abs(int(height_signed))
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        raise ValueError(f"invalid BMP size {width}x{height}")
    if planes != 1 or bpp not in (24, 32) or compression not in (0, 3):
        raise ValueError(f"unsupported BMP bpp={bpp} compression={compression}")

    nbytes = bpp // 8
    row_stride = ((width * nbytes + 3) // 4) * 4
    needed = pixel_off + row_stride * height
    if len(bmp) < needed:
        raise ValueError(
            f"truncated BMP: have {len(bmp)} need {needed} for {width}x{height}"
        )

    body = np.frombuffer(
        bmp, dtype=np.uint8, offset=pixel_off, count=row_stride * height
    )
    rows = body.reshape((height, row_stride))
    pixels = rows[:, : width * nbytes].reshape((height, width, nbytes))
    rgb = np.ascontiguousarray(pixels[:, :, 2::-1])  # BGR(A) → RGB
    if bottom_up:
        rgb = np.ascontiguousarray(rgb[::-1])
    meta = {
        "width": int(width),
        "height": int(height),
        "bpp": int(bpp),
        "bmp_offset": int(bm),
    }
    return rgb, meta


def _find_stream(
    ole: "olefile.OleFileIO",
    stream_name: str,
    under: Sequence[str] = (),
) -> Optional[List[str]]:
    """Return the first stream path ending with stream_name under `under`."""
    prefix = tuple(under)
    for entry in ole.listdir(streams=True, storages=True):
        if entry[-1] != stream_name:
            continue
        if prefix and tuple(entry[: len(prefix)]) != prefix:
            continue
        try:
            if ole.get_type(entry) == olefile.STGTY_STREAM:
                return list(entry)
        except Exception:
            continue
    return None


def _parse_overview(
    ole: "olefile.OleFileIO",
    fov_base: List[str],
) -> Optional[OverviewImage]:
    images_base = fov_base + ["Images"]
    for img_id in _child_storages(ole, images_base):
        if not img_id.startswith("SEI"):
            continue
        label = _read_label(ole, images_base + [img_id, "GreyImage", "label"])
        mm_path = images_base + [img_id, "GreyImage", "micromap"]
        if not ole.exists(mm_path):
            continue
        data, meta = _read_micromap(ole.openstream(mm_path).read())
        return OverviewImage(
            name=label.get("title") or "Overview",
            data=data,
            metadata={"image_id": img_id, **meta, **label},
        )
    return None


def _parse_spectra(
    ole: "olefile.OleFileIO",
    fov_base: List[str],
) -> List[MapSpectrum]:
    spectra_base = fov_base + ["Spectra"]
    out: List[MapSpectrum] = []
    for spe_id in _child_storages(ole, spectra_base):
        spe_base = spectra_base + [spe_id]
        spec_path = spe_base + ["XrayData", "Spectrum"]
        if not ole.exists(spec_path):
            continue
        counts = _read_spectrum_counts(ole.openstream(spec_path).read())
        info = _read_information(ole, spe_base + ["Information"])
        peaks = _read_peak_labels(ole, spe_base + ["PeakLabels", "PeakLabels"])
        live_time = float(info.get("live_time", 100.0))
        real_time = float(info.get("real_time", live_time))
        name = str(info.get("name") or spe_id)
        ev_per_ch = _infer_ev_per_channel(counts, peaks)
        n = len(counts)
        energy = (np.arange(n, dtype=np.float64) * ev_per_ch) / 1000.0  # keV

        kind = "sum" if "sum" in name.lower() else "spot"
        # Numbered spectra belong to an ordered point series (line or multipoint)
        if re.fullmatch(r"Spectrum\s+\d+", name, flags=re.I):
            kind = "line_point"

        spectrum = Spectrum(
            energy=energy,
            counts=counts.astype(np.float64),
            live_time=live_time,
            real_time=real_time,
            metadata={
                "name": name,
                "spe_id": spe_id,
                "ev_per_channel": ev_per_ch,
                "energy_calibration": "ipj_default_or_peak_inferred",
                "source": "ipj",
            },
        )
        index = None
        m = re.search(r"(\d+)\s*$", name)
        if m and kind == "line_point":
            index = int(m.group(1))

        stage_x, stage_y = _read_xgt2_stage_xy(ole, spe_base)

        out.append(
            MapSpectrum(
                spectrum=spectrum,
                name=name,
                x=stage_x,
                y=stage_y,
                index=index,
                kind=kind,
                peak_labels=peaks,
                metadata={"spe_id": spe_id},
            )
        )

    # Stable order: sum first, then by index/name
    def _sort_key(ms: MapSpectrum):
        if ms.kind == "sum":
            return (0, 0, ms.name)
        if ms.index is not None:
            return (1, ms.index, ms.name)
        return (2, 0, ms.name)

    out.sort(key=_sort_key)
    return out


def _read_xgt2_stage_xy(
    ole: "olefile.OleFileIO",
    spe_base: List[str],
) -> Tuple[Optional[float], Optional[float]]:
    """Read stage X/Y floats from the spectrum's XGT2Data stream, if present."""
    path = spe_base + ["XGT2Data", "XGT2Data"]
    if not ole.exists(path):
        return None, None
    raw = ole.openstream(path).read()
    need = max(_XGT2_STAGE_X_OFF, _XGT2_STAGE_Y_OFF) + 4
    if len(raw) < need:
        return None, None
    x = struct.unpack_from("<f", raw, _XGT2_STAGE_X_OFF)[0]
    y = struct.unpack_from("<f", raw, _XGT2_STAGE_Y_OFF)[0]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None, None
    if abs(x) > 1e5 or abs(y) > 1e5:
        return None, None
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return None, None
    return float(x), float(y)


def classify_point_series_kind(
    points: Sequence[MapSpectrum],
    rel_tol: float = _LINE_SCAN_STEP_REL_TOL,
) -> str:
    """
    Classify ordered points as line_scan (equal step size) or multipoint.

    Uses consecutive Euclidean distances in stage X/Y. Without usable
    coordinates we cannot verify equal spacing, so the series is multipoint.
    """
    coords = [
        (float(p.x), float(p.y))
        for p in points
        if p.x is not None and p.y is not None
        and np.isfinite(p.x) and np.isfinite(p.y)
    ]
    if len(coords) < 3:
        return "multipoint"
    arr = np.asarray(coords, dtype=np.float64)
    steps = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    if not np.all(np.isfinite(steps)):
        return "multipoint"
    med = float(np.median(steps))
    if med <= 0:
        return "multipoint"
    if float(np.max(np.abs(steps - med))) <= rel_tol * med:
        return "line_scan"
    return "multipoint"


def _maybe_attach_line_scan(fov: MappingFOV) -> None:
    """Group numbered spectra as a line scan or multipoint series."""
    points = [s for s in fov.spectra if s.kind == "line_point"]
    if len(points) < 3:
        return
    points = sorted(points, key=lambda s: (s.index is None, s.index or 0, s.name))
    kind = classify_point_series_kind(points)
    n = len(points)
    if kind == "line_scan":
        name = f"Line scan ({n} points)"
    else:
        name = f"Multipoint ({n} points)"
    steps = None
    if all(p.x is not None and p.y is not None for p in points):
        xs = np.array([p.x for p in points], dtype=np.float64)
        ys = np.array([p.y for p in points], dtype=np.float64)
        step_arr = np.hypot(np.diff(xs), np.diff(ys))
        steps = {
            "median": float(np.median(step_arr)),
            "mean": float(np.mean(step_arr)),
            "cv": float(np.std(step_arr) / (np.mean(step_arr) or 1.0)),
            "rel_tol": _LINE_SCAN_STEP_REL_TOL,
        }
    fov.line_scans.append(
        LineScan(
            name=name,
            points=points,
            source="ipj",
            kind=kind,
            metadata={"fov_id": fov.id, "step_stats": steps},
        )
    )


def _read_micromap(raw: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode INCA GreyImage/micromap binary."""
    if len(raw) < _MICROMAP_HEADER + _MICROMAP_TRAILER:
        raise ValueError(f"micromap too small: {len(raw)} bytes")

    # Header: u32[0]=8?, … width @16, height @20 (observed)
    width, height = struct.unpack_from("<II", raw, 16)
    if width == 0 or height == 0 or width > 8192 or height > 8192:
        raise ValueError(f"invalid micromap dims {width}x{height}")

    payload = len(raw) - _MICROMAP_HEADER - _MICROMAP_TRAILER
    n_pix = width * height
    if n_pix == 0:
        raise ValueError("empty micromap")

    bpp = payload // n_pix
    if bpp not in (1, 2, 4) or payload != n_pix * bpp:
        # Try without assuming trailer if exact fit after header
        payload2 = len(raw) - _MICROMAP_HEADER
        bpp2 = payload2 // n_pix if n_pix else 0
        if bpp2 in (1, 2, 4) and payload2 == n_pix * bpp2:
            bpp = bpp2
            body = raw[_MICROMAP_HEADER : _MICROMAP_HEADER + n_pix * bpp]
        else:
            raise ValueError(
                f"cannot parse micromap {width}x{height}: "
                f"payload={payload} bpp candidate={bpp}"
            )
    else:
        body = raw[_MICROMAP_HEADER : _MICROMAP_HEADER + n_pix * bpp]

    dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[bpp]
    data = np.frombuffer(body, dtype=dtype).reshape((height, width))
    # Copy so array owns memory after OLE stream closes
    data = np.array(data, copy=True)
    return data, {"width": width, "height": height, "bpp": bpp}


def _read_spectrum_counts(raw: bytes) -> np.ndarray:
    """
    Decode an INCA/XGT Spectrum stream.

    Layout: 10-byte header + 4096 little-endian uint32 counts.
    (Older mistaken u16 decode produced an 8192-bin comb with zeros on odd channels.)
    """
    if len(raw) < _SPECTRUM_HEADER + 4:
        raise ValueError("spectrum stream too small")
    body = raw[_SPECTRUM_HEADER:]

    # Prefer explicit channel count from header when present (u16 @ offset 6 = 0x0FFF)
    n_hint = 0
    if len(raw) >= 8:
        n_hint = struct.unpack_from("<H", raw, 6)[0]
        if n_hint == 0x0FFF:
            n_hint = 4096
        elif not (256 <= n_hint <= 8192):
            n_hint = 0

    # uint32 body (normal XGT/INCA case)
    if len(body) >= 4 and len(body) % 4 == 0:
        counts32 = np.frombuffer(body, dtype="<u4")
        if n_hint and len(counts32) >= n_hint:
            return np.array(counts32[:n_hint], dtype=np.float64, copy=True)
        if len(counts32) in (2048, 4096, 8192):
            return np.array(counts32, dtype=np.float64, copy=True)
        # 16384 bytes → 4096 u32
        if len(counts32) == _DEFAULT_N_CHANNELS:
            return np.array(counts32, dtype=np.float64, copy=True)

    # Fallback: uint16 (rare)
    if len(body) % 2 != 0:
        body = body[: len(body) - 1]
    counts16 = np.frombuffer(body, dtype="<u2")
    # Detect mistaken interleaved high-word zeros → collapse to u32 view
    if (
        len(counts16) >= 4
        and len(counts16) % 2 == 0
        and np.mean(counts16[1::2] == 0) > 0.9
        and np.mean(counts16[0::2] > 0) > 0.05
    ):
        return np.array(counts16[0::2], dtype=np.float64, copy=True)
    return np.array(counts16, dtype=np.float64, copy=True)


def _read_information(
    ole: "olefile.OleFileIO",
    path: Sequence[str],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if not ole.exists(list(path)):
        return info
    raw = ole.openstream(list(path)).read()
    # Live time often stored as double near offset 38–40 (observed ~300.0)
    for off in (38, 40, 32, 48):
        if off + 8 <= len(raw):
            val = struct.unpack_from("<d", raw, off)[0]
            if 0.1 < val < 1e7:
                info.setdefault("live_time", float(val))
                break
    # Real time sometimes later
    for off in (50, 58, 56, 64):
        if off + 8 <= len(raw):
            val = struct.unpack_from("<d", raw, off)[0]
            if 0.1 < val < 1e7 and abs(val - info.get("live_time", -1)) > 0.01:
                info["real_time"] = float(val)
                break
    # ASCII name (e.g. "Sum Spectrum", "Spectrum 12")
    names = re.findall(rb"[\x20-\x7e]{4,40}", raw)
    preferred = None
    for n in names:
        s = n.decode("ascii", errors="ignore")
        if "spectrum" in s.lower() or s.lower().startswith("sum"):
            preferred = s.strip("\x00 ")
            break
    if preferred is None and names:
        preferred = names[-1].decode("ascii", errors="ignore").strip()
    if preferred:
        info["name"] = preferred
    return info


def _read_peak_labels(
    ole: "olefile.OleFileIO",
    path: Sequence[str],
) -> List[Dict[str, Any]]:
    if not ole.exists(list(path)):
        return []
    raw = ole.openstream(list(path)).read()
    if len(raw) < 12:
        return []
    # Header: u32 version?, u32 count
    count = struct.unpack_from("<I", raw, 4)[0]
    if count <= 0 or count > 5000:
        return []
    records: List[Dict[str, Any]] = []
    off = 8
    # Record: u16 Z, u16 unk, 2-char element, float32 energy_eV  (10 bytes)
    for _ in range(count):
        if off + 10 > len(raw):
            break
        z, unk = struct.unpack_from("<HH", raw, off)
        el = raw[off + 4 : off + 6].decode("ascii", errors="replace").strip()
        energy_ev = struct.unpack_from("<f", raw, off + 6)[0]
        if 1 <= z <= 118 and 100 < energy_ev < 120000:
            records.append(
                {
                    "z": int(z),
                    "element": el,
                    "energy_ev": float(energy_ev),
                    "energy_kev": float(energy_ev) / 1000.0,
                    "flags": int(unk),
                }
            )
        off += 10
    return records


def _read_label(
    ole: "olefile.OleFileIO",
    path: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not ole.exists(list(path)):
        return out
    raw = ole.openstream(list(path)).read()
    # Pascal-ish: u32 len, bytes, u32 len, bytes ("Map")
    off = 0
    parts: List[str] = []
    while off + 4 <= len(raw):
        n = struct.unpack_from("<I", raw, off)[0]
        off += 4
        if n <= 0 or off + n > len(raw) or n > 200:
            break
        parts.append(raw[off : off + n].decode("latin1", errors="replace"))
        off += n
    if parts:
        out["title"] = parts[0]
    if len(parts) > 1:
        out["kind"] = parts[1]
    return out


def _infer_ev_per_channel(
    counts: np.ndarray,
    peaks: List[Dict[str, Any]],
) -> float:
    """
    Prefer 10 eV/ch (common XGT). If peak labels exist, refine using the
    strongest labeled Ka-like line near a local maximum.
    """
    candidates = [5.0, 10.0, 20.0]
    if not peaks:
        return _DEFAULT_EV_PER_CHANNEL

    # Use first few unique element Ka energies (~primary)
    by_el: Dict[str, float] = {}
    for p in peaks:
        el = p["element"]
        e = p["energy_ev"]
        # Prefer ~K-alpha region: for Z, rough Ka energy
        if el not in by_el or abs(e - by_el[el]) > 500:
            # keep lowest energy per element as Ka-ish
            if el not in by_el or e < by_el[el]:
                by_el[el] = e

    best = _DEFAULT_EV_PER_CHANNEL
    best_score = -1.0
    for ev in candidates:
        score = 0.0
        for e_ev in list(by_el.values())[:6]:
            ch = int(round(e_ev / ev))
            if 5 <= ch < len(counts) - 5:
                window = counts[ch - 5 : ch + 6]
                score += float(window.max())
        if score > best_score:
            best_score = score
            best = ev
    return best


def _child_storages(ole: "olefile.OleFileIO", base: Sequence[str]) -> List[str]:
    """Return immediate child storage names under base."""
    base_t = tuple(base)
    children: List[str] = []
    seen = set()
    for entry in ole.listdir(streams=True, storages=True):
        if len(entry) == len(base_t) + 1 and tuple(entry[: len(base_t)]) == base_t:
            name = entry[-1]
            if name not in seen:
                # Only storages (directories) — skip if it's only a stream at this level
                # listdir returns both; prefer unique names that have further children
                # or exist as storage
                seen.add(name)
                children.append(name)
    # Filter to storages: those with deeper paths OR type storage
    storages: List[str] = []
    for name in children:
        path = list(base) + [name]
        # Has children?
        has_child = any(
            len(e) > len(path) and list(e[: len(path)]) == path
            for e in ole.listdir(streams=True, storages=True)
        )
        if has_child:
            storages.append(name)
    return storages
