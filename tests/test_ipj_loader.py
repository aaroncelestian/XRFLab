"""Tests for INCA/XGT .ipj mapping loader."""

from pathlib import Path

import numpy as np
import pytest

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data" / "data"
IPJ_FILES = {
    "barstow": SAMPLE_DIR / "barstow1.ipj",
    "dylan": SAMPLE_DIR / "dylan_corsetti_slide_1_STROM.ipj",
    "emerald": SAMPLE_DIR / "ca emerald with citrine.ipj",
}


def _have_olefile():
    try:
        import olefile  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _have_olefile(),
    reason="olefile not installed",
)


@pytest.fixture(scope="module")
def barstow():
    from utils.ipj_loader import load_ipj

    if not IPJ_FILES["barstow"].exists():
        pytest.skip("barstow1.ipj not present")
    return load_ipj(IPJ_FILES["barstow"])


@pytest.fixture(scope="module")
def dylan():
    from utils.ipj_loader import load_ipj

    if not IPJ_FILES["dylan"].exists():
        pytest.skip("dylan ipj not present")
    return load_ipj(IPJ_FILES["dylan"])


@pytest.fixture(scope="module")
def emerald():
    from utils.ipj_loader import load_ipj

    if not IPJ_FILES["emerald"].exists():
        pytest.skip("emerald ipj not present")
    return load_ipj(IPJ_FILES["emerald"])


def test_barstow_spectra_and_dims(barstow):
    assert barstow.name == "barstow1"
    assert len(barstow.samples) >= 1
    assert barstow.samples[0].name == "Sample 1"
    assert len(barstow.samples[0].sites) >= 2
    site_names = {s.name for s in barstow.samples[0].sites}
    assert "Site of Interest 1" in site_names
    assert "Site of Interest 2" in site_names
    assert len(barstow.fovs) >= 1
    all_spec = barstow.all_spectra()
    assert len(all_spec) >= 10
    # At least one sum spectrum
    assert any(s.kind == "sum" or "sum" in s.name.lower() for s in all_spec)
    # Spectrum shape: 4096 × uint32 (not comb-aliased 8192 u16)
    s0 = all_spec[0].spectrum
    assert s0.num_channels == 4096
    assert s0.counts.max() > 0
    assert np.mean(s0.counts[1::2] == 0) < 0.9  # not the old zero-odd comb
    assert s0.energy[-1] > s0.energy[0]


def test_dylan_point_series_and_maps(dylan):
    map_fov = dylan.primary_fov
    assert map_fov is not None
    assert map_fov.width == 128
    assert map_fov.height == 109
    assert len(map_fov.element_maps) >= 3
    names = {m.name for m in map_fov.element_maps}
    assert any("Fe" in n for n in names)
    assert any("Si" in n for n in names)

    # Numbered Spectrum 1…N series (step spacing decides line vs multipoint)
    series_fovs = [f for f in dylan.fovs if f.line_scans]
    assert series_fovs, "expected a numbered point-series FOV"
    ls = series_fovs[0].line_scans[0]
    assert ls.n_points >= 30
    assert ls.points[0].spectrum.num_channels == 4096
    assert ls.points[0].x is not None and ls.points[0].y is not None
    # Dylan Site 2 has irregular stage steps → multipoint, not line scan
    assert ls.kind == "multipoint"
    assert ls.is_multipoint
    assert "Multipoint" in ls.name


def test_barstow_point_series_is_multipoint(barstow):
    series_fovs = [f for f in barstow.fovs if f.line_scans]
    assert series_fovs, "expected numbered spectra FOV"
    ls = series_fovs[0].line_scans[0]
    assert ls.n_points >= 10
    assert ls.kind == "multipoint"
    assert all(p.x is not None and p.y is not None for p in ls.points)


def test_classify_equal_vs_irregular_steps():
    from core.mapping.models import MapSpectrum
    from core.spectrum import Spectrum
    from utils.ipj_loader import classify_point_series_kind

    def _pt(i, x, y):
        sp = Spectrum(
            energy=np.arange(4, dtype=np.float64),
            counts=np.ones(4, dtype=np.float64),
            live_time=1.0,
            real_time=1.0,
            metadata={"name": f"Spectrum {i}"},
        )
        return MapSpectrum(spectrum=sp, name=f"Spectrum {i}", x=x, y=y, index=i, kind="line_point")

    line = [_pt(i, float(i), 0.0) for i in range(1, 8)]
    assert classify_point_series_kind(line) == "line_scan"

    multi = [
        _pt(1, 0.0, 0.0),
        _pt(2, 1.0, 0.0),
        _pt(3, 1.2, 0.0),
        _pt(4, 5.0, 3.0),
        _pt(5, 5.1, 3.2),
    ]
    assert classify_point_series_kind(multi) == "multipoint"


def test_emerald_element_maps(emerald):
    fov = emerald.primary_fov
    assert fov is not None
    assert len(fov.element_maps) >= 10
    for m in fov.element_maps:
        assert m.data.ndim == 2
        assert m.data.size > 0
        assert m.element


def test_line_profile_and_correlation(dylan):
    from core.mapping.profiles import extract_line_profile
    from core.mapping.correlations import map_correlation, rgb_composite

    fov = dylan.primary_fov
    m0 = fov.element_maps[0]
    m1 = fov.element_maps[1]
    dist, vals = extract_line_profile(
        m0, (0, m0.height // 2), (m0.width - 1, m0.height // 2)
    )
    assert len(dist) == len(vals)
    assert len(dist) > 10

    x, y, r, rho = map_correlation(m0, m1)
    assert len(x) == len(y)
    assert np.isfinite(r) or np.isnan(r)

    rgb = rgb_composite(m0, m1, None)
    assert rgb.shape == (m0.height, m0.width, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_ipj_sample_info_text_fields(dylan, barstow, emerald):
    from utils.ipj_loader import _read_counted_strings

    raw = (
        b"\x01\x00\x00\x00\x34"
        b"\x08\x00\x00\x00Sample 1"
        b"\x00\x00\x00\x00"
        b"\x07\x00\x00\x00Default"
    )
    assert _read_counted_strings(raw)[:3] == ["Sample 1", "", "Default"]

    assert dylan.metadata.get("instrument") == "XGT7200"
    assert dylan.metadata.get("project_title")
    sample = dylan.samples[0]
    assert sample.name == "Sample 1"
    assert sample.metadata.get("sample_type") == "Default"
    site = dylan.primary_fov
    assert site is not None
    assert site.name.startswith("Site of Interest")

    assert emerald.metadata.get("project_title") == "ca emerald with citrine"
    assert barstow.metadata.get("instrument") == "XGT7200"


def test_dylan_map_acquisition_metadata(dylan):
    fov = dylan.primary_fov
    assert fov is not None
    assert fov.metadata.get("map_live_time_s") == 300.0
    assert fov.width == 128 and fov.height == 109
    assert fov.metadata.get("n_pixels") == 128 * 109
    dwell_ms = fov.metadata.get("dwell_ms")
    assert dwell_ms is not None
    assert abs(dwell_ms - 300_000 / (128 * 109)) < 0.05
    assert fov.metadata.get("kv") == 30.0
    assert fov.metadata.get("ma") == 15.0
    acquired = fov.metadata.get("acquired_at") or ""
    assert acquired.startswith("2020-07-13")
    summary = fov.acquisition_summary()
    assert "ms/pixel" in summary
    assert "30 kV" in summary


def test_hyperspectral_cube_dylan(dylan):
    fov = dylan.primary_fov
    assert fov is not None
    assert fov.cube is not None
    assert fov.cube.shape == (4096, 109, 128)
    # Sum spectrum matches cube totals (same 4096 bins)
    sum_ms = fov.sum_spectrum()
    assert sum_ms is not None
    assert sum_ms.spectrum.num_channels == 4096
    assert np.allclose(fov.cube.sum_spectrum(), sum_ms.spectrum.counts)

    # Pixel extract stays on cube channel axis
    ms = fov.spectrum_at_pixel(64, 54)
    assert ms is not None
    assert ms.spectrum.num_channels == 4096
    assert ms.spectrum.counts.sum() > 0

    # ROI map from Fe Ka region
    em = fov.add_roi_map_from_cube(6.30, 6.50, name="Fe Ka ROI")
    assert em is not None
    assert em.data.shape == (109, 128)
    assert em.data.sum() > 0


def test_hyperspectral_cube_barstow(barstow):
    """barstow has no vendor MAP images — cube provides total-counts map."""
    fov = next((f for f in barstow.fovs if f.cube is not None), None)
    assert fov is not None
    assert fov.cube.shape[0] == 4096
    assert any(m.metadata.get("source") == "cube_total" for m in fov.element_maps)
    sum_ms = fov.sum_spectrum()
    assert sum_ms is not None
    assert np.allclose(fov.cube.sum_spectrum(), sum_ms.spectrum.counts)


def _make_test_bmp(rgb: np.ndarray) -> bytes:
    """Minimal 24-bit bottom-up BMP with a 24-byte vendor prefix."""
    import struct

    h, w = rgb.shape[:2]
    stride = ((w * 3 + 3) // 4) * 4
    pixel_off = 54
    filesize = pixel_off + stride * h
    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, filesize)
    struct.pack_into("<I", header, 10, pixel_off)
    struct.pack_into("<I", header, 14, 40)
    struct.pack_into("<i", header, 18, w)
    struct.pack_into("<i", header, 22, h)
    struct.pack_into("<HH", header, 26, 1, 24)
    body = bytearray(stride * h)
    flipped = rgb[::-1]
    for y in range(h):
        for x in range(w):
            r, g, b = (int(v) for v in flipped[y, x])
            i = y * stride + x * 3
            body[i : i + 3] = bytes((b, g, r))
    prefix = b"\x02\x00\x01\x00\x03\x00" + b"\x00" * 18
    return prefix + bytes(header) + bytes(body)


def test_decode_embedded_bmp_roundtrip():
    from utils.ipj_loader import decode_embedded_bmp

    src = np.zeros((3, 5, 3), dtype=np.uint8)
    src[0, 0] = (255, 0, 0)
    src[1, 2] = (0, 255, 0)
    src[2, 4] = (0, 0, 255)
    rgb, meta = decode_embedded_bmp(_make_test_bmp(src))
    assert rgb.shape == (3, 5, 3)
    assert rgb.dtype == np.uint8
    assert meta["width"] == 5 and meta["height"] == 3
    np.testing.assert_array_equal(rgb, src)


def test_optical_camera_bmps(barstow, dylan, emerald):
    for proj in (barstow, dylan, emerald):
        assert proj.samples, proj.name
        sample = proj.samples[0]
        assert sample.whole_image is not None, proj.name
        img = sample.whole_image.data
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.shape == (1944, 2592, 3)
        assert img.dtype == np.uint8
        assert int(img.max()) > 0

    barstow_opt = next((f for f in barstow.fovs if f.optical is not None), None)
    assert barstow_opt is not None
    assert barstow_opt.optical.data.shape[2] == 3
    assert barstow_opt.optical.data.shape[0] > 100
    assert barstow_opt.optical.metadata.get("kind") == "map_area"

    dylan_opt = next((f for f in dylan.fovs if f.optical is not None), None)
    assert dylan_opt is not None
    assert dylan_opt.optical.data.ndim == 3
