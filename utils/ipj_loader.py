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
from datetime import datetime, timedelta
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
# INCA/XGT Spectrum streams are 4096 × uint32 at 10 eV/ch.
# Channel 0 is ~-0.40 keV so the electronic zero/noise peak sits near 0 keV.
_DEFAULT_EV_PER_CHANNEL = 10.0
_DEFAULT_N_CHANNELS = 4096
_XGT_ZERO_CHANNEL = 40.0
# XGT2Data stage X/Y: float32 at these offsets (4-byte zero pad after each).
# Empirically stage millimetres for spot / line / multipoint spectra and for
# the map Sum Spectrum (map FOV centre). Nearby doubles at 150/158 are a
# different field (often large encoder-like values on map FOVs) — do not use.
_XGT2_STAGE_X_OFF = 154
_XGT2_STAGE_Y_OFF = 162
# Max relative deviation from median step for "equal spacing" line scans
_LINE_SCAN_STEP_REL_TOL = 0.15
# XGT-7200 stage travel is 100 mm; reject absurd stage readings
_STAGE_XY_ABS_MAX_MM = 150.0
# OLE Automation date epoch (same as Excel)
_OLE_EPOCH = datetime(1899, 12, 30)


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
        sample_info = _read_inca_info(ole, ["Samples", sample_id, "ISampleInfo"])
        sample_name = (
            sample_info.get("name")
            or _read_display_name(
                ole,
                ["Samples", sample_id, "ISampleInfo"],
                preferred_prefixes=("Sample",),
                default=f"Sample {sample_idx}",
            )
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

        sample_meta = {
            "path": f"Samples/{sample_id}",
            "comment": sample_info.get("comment", ""),
            "sample_type": sample_info.get("sample_type", ""),
        }
        samples.append(
            MappingSample(
                id=sample_id,
                name=sample_name,
                sites=sites,
                whole_image=whole_image,
                metadata=sample_meta,
            )
        )

    # Promote numbered spectrum series into LineScan when appropriate
    for fov in all_fovs:
        _maybe_attach_line_scan(fov)

    n_cubes = sum(1 for f in all_fovs if f.cube is not None)
    proj_info = _read_inca_info(ole, ["IIncaProjectInfo"])
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
            "project_title": proj_info.get("name") or path.stem,
            "instrument": proj_info.get("instrument", ""),
            "comment": proj_info.get("comment", ""),
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
    site_info = _read_inca_info(ole, base + ["IFOVInfo"])
    site_name = (
        site_info.get("name")
        or _read_display_name(
            ole,
            base + ["IFOVInfo"],
            preferred_prefixes=("Site of Interest", "Site"),
            default=f"Site of Interest {site_index}",
        )
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
    acq = _acquisition_metadata(
        ole,
        base,
        spectra,
        width=int(width),
        height=int(height),
    )
    map_geom = _parse_map_extra_geometry(
        ole, base, map_width=int(width), map_height=int(height)
    )
    pixel_size_mm = map_geom.get("pixel_size_mm")
    stage_center = _stage_center_from_spectra(spectra)
    # Prefer explicit µm size from MapExtra when present
    size_mm = map_geom.get("size_mm")
    if size_mm is not None and width > 0 and height > 0:
        # Keep pixel_size consistent with C,D even if mean pitch already set
        pixel_size_mm = float(size_mm[0]) / float(width)
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
        pixel_size_mm=pixel_size_mm,
        stage_center_mm=stage_center,
        metadata={
            "sample_id": sample_id,
            "site_name": site_name,
            "has_smartmap": has_smartmap,
            "has_cube": cube is not None,
            "has_optical": optical is not None,
            "cube_shape": tuple(cube.shape) if cube is not None else None,
            "path": "/".join(base),
            "comment": site_info.get("comment", ""),
            "map_extra": map_geom,
            **acq,
        },
    )


_INFO_PLACEHOLDERS = {
    "",
    "comment:",
    "comment",
    "default",
    "empty",
}


def _read_counted_strings(raw: bytes, start: int = 5) -> List[str]:
    """Parse INCA I*Info streams: [u32 ver][u8 tag] then (u32 n, n bytes)*."""
    out: List[str] = []
    off = start if len(raw) >= start else 0
    while off + 4 <= len(raw):
        n = struct.unpack_from("<I", raw, off)[0]
        off += 4
        if n == 0:
            out.append("")
            continue
        if n > 400 or off + n > len(raw):
            break
        out.append(raw[off : off + n].decode("latin1", errors="replace"))
        off += n
    return out


def _clean_info_text(value: str) -> str:
    text = (value or "").strip()
    if text.lower() in _INFO_PLACEHOLDERS:
        return ""
    return text


def _read_inca_info(
    ole: "olefile.OleFileIO",
    path: Sequence[str],
) -> Dict[str, Any]:
    """Read name / comment / type / instrument from an I*Info stream."""
    out: Dict[str, Any] = {}
    if not ole.exists(list(path)):
        return out
    raw = ole.openstream(list(path)).read()
    strings = _read_counted_strings(raw)
    out["strings"] = strings
    kind = path[-1] if path else ""

    if kind == "IIncaProjectInfo":
        if strings:
            out["name"] = _clean_info_text(strings[0])
        if len(strings) > 1:
            out["instrument"] = _clean_info_text(strings[1])
        if len(strings) > 2:
            out["comment"] = _clean_info_text(strings[2])
        return out

    if kind == "ISampleInfo":
        if strings:
            out["name"] = _clean_info_text(strings[0]) or strings[0].strip()
        if len(strings) > 1:
            out["comment"] = _clean_info_text(strings[1])
        if len(strings) > 2:
            typ = strings[2].strip()
            out["sample_type"] = "" if typ.lower() in ("0",) else typ
        return out

    if kind == "IFOVInfo":
        # Typical: [comment, site name]. Sometimes only the site name.
        name = ""
        comment = ""
        for s in strings:
            cleaned = _clean_info_text(s)
            if not cleaned:
                continue
            if re.match(r"^(Site of Interest|Site)\b", cleaned, flags=re.I):
                name = cleaned
            elif not comment:
                comment = cleaned
            elif not name:
                name = cleaned
        if not name:
            for s in reversed(strings):
                cleaned = _clean_info_text(s) or s.strip()
                if cleaned:
                    name = cleaned
                    break
        out["name"] = name
        out["comment"] = comment
        return out

    if strings:
        out["name"] = _clean_info_text(strings[0]) or strings[0].strip()
    return out


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
                cube.energy_offset_ev = float(
                    sum_ms.spectrum.metadata.get("energy_offset_ev", 0.0)
                )
            elif len(expected) == cube.n_channels * 2:
                cube.ev_per_channel = ev * 2.0
                cube.energy_offset_ev = float(
                    sum_ms.spectrum.metadata.get("energy_offset_ev", 0.0)
                )
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
        ev_per_ch, offset_ev = _infer_energy_calibration(counts, peaks)
        n = len(counts)
        energy = (offset_ev + np.arange(n, dtype=np.float64) * ev_per_ch) / 1000.0

        kind = "sum" if "sum" in name.lower() else "spot"
        # Numbered spectra belong to an ordered point series (line or multipoint)
        if re.fullmatch(r"Spectrum\s+\d+", name, flags=re.I):
            kind = "line_point"

        em = _read_em_conditions(
            ole, spe_base + ["EMConditions", "EMConditions"]
        )
        spec_meta = {
            "name": name,
            "spe_id": spe_id,
            "ev_per_channel": ev_per_ch,
            "energy_offset_ev": offset_ev,
            "energy_calibration": "xgt_10eV_ch_offset",
            "source": "ipj",
            "live_time": live_time,
            "real_time": real_time,
        }
        if info.get("acquired_at"):
            spec_meta["acquired_at"] = info["acquired_at"]
        # Analysis Experimental Parameters (mA, not EMSA nanoamps)
        if "kv" in em:
            spec_meta["excitation_energy"] = float(em["kv"])
            spec_meta["kv"] = float(em["kv"])
        if "ma" in em:
            spec_meta["tube_current_ma"] = float(em["ma"])
            spec_meta["ma"] = float(em["ma"])
        spectrum = Spectrum(
            energy=energy,
            counts=counts.astype(np.float64),
            live_time=live_time,
            real_time=real_time,
            metadata=spec_meta,
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
    """
    Read stage X/Y (mm) from the spectrum's ``XGT2Data/XGT2Data`` stream.

    Layout (little-endian), observed on XGT-7200 IPJs::

        offset 154: float32 X (mm)
        offset 158: 4 zero pad bytes
        offset 162: float32 Y (mm)
        offset 166: 4 zero pad bytes

    The doubles that start at 150/158 on some map FOVs are a different
    quantity (often thousands) and must not be used as stage millimetres.
    """
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
    if abs(x) > _STAGE_XY_ABS_MAX_MM or abs(y) > _STAGE_XY_ABS_MAX_MM:
        return None, None
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return None, None
    return float(x), float(y)


def _stage_center_from_spectra(
    spectra: Sequence[MapSpectrum],
) -> Optional[Tuple[float, float]]:
    """Prefer the sum spectrum's stage XY as the map FOV centre."""
    for s in spectra:
        if s.kind == "sum" and s.x is not None and s.y is not None:
            return (float(s.x), float(s.y))
    for s in spectra:
        if "sum" in (s.name or "").lower() and s.x is not None and s.y is not None:
            return (float(s.x), float(s.y))
    return None


def _parse_map_extra_geometry(
    ole: "olefile.OleFileIO",
    fov_base: List[str],
    *,
    map_width: int = 0,
    map_height: int = 0,
) -> Dict[str, Any]:
    """
    Parse ``XGT2 MapExtraData`` for map stage size / pixel pitch.

    Doubles are stored as ``05 00`` + float64. After a ~0.14 value (probe /
    related, *not* the map step) and ``5.89`` / ``1.0``, the stream carries:

        A, B, 0.0, C, D

    where ``C`` and ``D`` are the map width and height in **micrometres**
    (so ``C/width`` and ``D/height`` are an integer µm/pixel). Physical size
    is therefore ``(C/1000, D/1000)`` mm — not ``dims × 0.14``.
    """
    path = fov_base + ["ExtraData", "MAPExDataKeyName", "XGT2 MapExtraData"]
    if not ole.exists(path):
        return {}
    raw = ole.openstream(path).read()
    doubles: List[float] = []
    i = 0
    while i < len(raw) - 9:
        if raw[i] == 5 and raw[i + 1] == 0:
            val = struct.unpack_from("<d", raw, i + 2)[0]
            if np.isfinite(val):
                doubles.append(float(val))
            i += 10
        else:
            i += 1
    out: Dict[str, Any] = {"doubles": doubles}

    # Locate the A,B,0,C,D block: after the unique ~(0.05,0.5) value and 5.89, 1.0
    probe = next((d for d in doubles if 0.05 < d < 0.5), None)
    size_w_um = size_h_um = None
    if probe is not None and probe in doubles:
        idx = doubles.index(probe)
        block = doubles[idx : idx + 8]
        out["extra_block"] = block
        out["probe_or_spot_mm"] = probe
        if len(block) >= 8 and abs(block[5]) < 1e-9:
            size_w_um, size_h_um = block[6], block[7]
            out["size_w_um"] = size_w_um
            out["size_h_um"] = size_h_um
            out["stage_ab_raw"] = (block[3], block[4])

    if size_w_um is not None and size_h_um is not None and size_w_um > 0 and size_h_um > 0:
        out["size_mm"] = (size_w_um / 1000.0, size_h_um / 1000.0)
        pitches = []
        if map_width > 0:
            pitches.append(size_w_um / map_width / 1000.0)
        if map_height > 0:
            pitches.append(size_h_um / map_height / 1000.0)
        if pitches:
            out["pixel_size_mm"] = float(np.mean(pitches))
    return out


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
    # OLE date at offset 58 (observed ~44025 → 2020-07)
    if len(raw) >= 66:
        ole_days = struct.unpack_from("<d", raw, 58)[0]
        iso = _ole_date_to_iso(ole_days)
        if iso:
            info["acquired_at"] = iso
            info["acquired_ole"] = float(ole_days)
    # Real time sometimes later; skip OLE-date-sized values
    live = float(info.get("live_time", -1))
    for off in (46, 50, 56, 64):
        if off + 8 <= len(raw):
            val = struct.unpack_from("<d", raw, off)[0]
            if not (0.1 < val < 1e6):
                continue
            if _ole_date_to_iso(val):
                continue
            if abs(val - live) > 0.01:
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


def _ole_date_to_iso(ole_days: float) -> Optional[str]:
    """Convert an OLE Automation date to ISO-8601, or None if implausible."""
    try:
        val = float(ole_days)
    except (TypeError, ValueError):
        return None
    # ~1982–2064; XGT/INCA files in this project are 2016–2020
    if not (30000.0 < val < 60000.0) or not np.isfinite(val):
        return None
    try:
        dt = _OLE_EPOCH + timedelta(days=val)
    except (OverflowError, ValueError):
        return None
    if dt.year < 1990 or dt.year > 2100:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _read_em_conditions(
    ole: "olefile.OleFileIO",
    path: Sequence[str],
) -> Dict[str, Any]:
    """Tube kV / mA from EMConditions (float32 @ 44 / 52).

    Present under SmartMap, per-spectrum, and GreyImage paths on XGT IPJs.
    """
    out: Dict[str, Any] = {}
    if not ole.exists(list(path)):
        return out
    raw = ole.openstream(list(path)).read()
    if len(raw) < 56:
        return out
    kv = struct.unpack_from("<f", raw, 44)[0]
    ma = struct.unpack_from("<f", raw, 52)[0]
    if np.isfinite(kv) and 5.0 <= kv <= 100.0:
        out["kv"] = float(kv)
    if np.isfinite(ma) and 0.1 <= ma <= 1000.0:
        out["ma"] = float(ma)
    return out


def _acquisition_metadata(
    ole: "olefile.OleFileIO",
    fov_base: Sequence[str],
    spectra: Sequence[MapSpectrum],
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Map live time, estimated dwell, tube settings, and acquisition stamp."""
    meta: Dict[str, Any] = {}
    sum_ms = next(
        (s for s in spectra if s.kind == "sum" or "sum" in s.name.lower()),
        None,
    )
    ref_ms = sum_ms or (spectra[0] if spectra else None)
    if ref_ms is not None:
        live = float(ref_ms.spectrum.live_time)
        if live > 0:
            meta["map_live_time_s"] = live
        real = float(ref_ms.spectrum.real_time)
        if real > 0:
            meta["map_real_time_s"] = real
        acquired = ref_ms.spectrum.metadata.get("acquired_at")
        if acquired:
            meta["acquired_at"] = acquired
    em = _read_em_conditions(
        ole, list(fov_base) + ["SmartMap", "EMConditions", "EMConditions"]
    )
    # Multipoint / spectra-only sites often have no SmartMap stream — use
    # the first spectrum (or sum) EMConditions instead.
    if "kv" not in em or "ma" not in em:
        for ms in spectra:
            sm = ms.spectrum.metadata or {}
            if "kv" not in em and sm.get("kv") is not None:
                em["kv"] = float(sm["kv"])
            if "ma" not in em and sm.get("ma") is not None:
                em["ma"] = float(sm["ma"])
            if "kv" in em and "ma" in em:
                break
    if "kv" not in em or "ma" not in em:
        for ms in spectra:
            spe_id = (ms.metadata or {}).get("spe_id") or ms.name
            path = list(fov_base) + [
                "Spectra",
                str(spe_id),
                "EMConditions",
                "EMConditions",
            ]
            em_spe = _read_em_conditions(ole, path)
            em.setdefault("kv", em_spe.get("kv"))
            em.setdefault("ma", em_spe.get("ma"))
            if em.get("kv") is not None and em.get("ma") is not None:
                break
    # Drop None placeholders from setdefault
    em = {k: v for k, v in em.items() if v is not None}
    meta.update(em)
    n = int(width) * int(height) if width and height else 0
    if n > 0:
        meta["n_pixels"] = n
    live = meta.get("map_live_time_s")
    if live and n > 0:
        dwell = float(live) / n
        meta["dwell_s"] = dwell
        meta["dwell_ms"] = dwell * 1000.0
        meta["dwell_source"] = "map_live_time / n_pixels"
    return meta


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


def _xgt_energy_offset_ev(ev_per_ch: float) -> float:
    """XGT-7200 MCA: channel 40 ≈ 0 keV (noise peak visible just above zero)."""
    return -_XGT_ZERO_CHANNEL * float(ev_per_ch)


def _infer_energy_calibration(
    counts: np.ndarray,
    peaks: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """
    Return (eV/channel, offset_eV) for E = offset + channel * gain.

    XGT/INCA 4096-bin spectra are 10 eV/ch with a -400 eV intercept. Without
    that offset, Ca Kα at channel 410 is plotted at 4.10 keV instead of 3.69.
    """
    ev = _infer_ev_per_channel(counts, peaks)
    offset = _xgt_energy_offset_ev(ev)
    if peaks:
        # Re-score gain using the XGT intercept so labeled Ka lines land on
        # the observed maxima (channel = (E - offset) / gain).
        by_el: Dict[str, float] = {}
        for p in peaks:
            el = p.get("element")
            e = p.get("energy_ev")
            if not el or e is None:
                continue
            if el not in by_el or e < by_el[el]:
                by_el[el] = float(e)
        best = ev
        best_score = -1.0
        for cand in (5.0, 10.0, 20.0):
            off = _xgt_energy_offset_ev(cand)
            score = 0.0
            for e_ev in list(by_el.values())[:6]:
                ch = int(round((e_ev - off) / cand))
                if 5 <= ch < len(counts) - 5:
                    window = counts[ch - 5 : ch + 6]
                    score += float(window.max())
            if score > best_score:
                best_score = score
                best = cand
        ev = best
        offset = _xgt_energy_offset_ev(ev)
    return float(ev), float(offset)


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
