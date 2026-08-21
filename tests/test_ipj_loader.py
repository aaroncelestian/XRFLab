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


def test_barstow_sum_spectrum_ca_ka_energy(barstow):
    """XGT intercept: Ca Kα in the tufa sum spectrum must sit near 3.69 keV."""
    fov = next(f for f in barstow.fovs if f.cube is not None)
    ms = fov.sum_spectrum()
    assert ms is not None
    sp = ms.spectrum
    i = int(np.argmax(sp.counts))
    e_max = float(sp.energy[i])
    assert 3.55 < e_max < 3.85, f"Ca Kα at {e_max:.3f} keV (expected ~3.69)"
    assert float(sp.metadata.get("energy_offset_ev", 0.0)) == -400.0
    # Cube ROI around Ca Kα must include the intense channels
    axis = fov.cube.energy_axis_kev()
    mask = (axis >= 3.59) & (axis <= 3.79)
    assert int(fov.cube.data[mask].sum()) > int(fov.cube.data.sum()) * 0.05


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
    # Stage XY from XGT2Data float64@150/158 (µm → mm)
    assert abs(ls.points[0].x) < 150 and abs(ls.points[0].y) < 150
    # Correct µm→mm positions: early dylan points are an equal-step transect
    from utils.ipj_loader import classify_point_series_kind

    early = ls.points[:12]
    assert classify_point_series_kind(early) == "line_scan"
    # Full series then wanders → multipoint overall
    assert ls.kind == "multipoint"
    assert ls.is_multipoint
    assert "Multipoint" in ls.name
    # Correct µm positions: early steps ~0.4 mm; full path travels tens of mm
    path = ls.path_distances()
    assert float(path[-1]) > 40.0
    assert float(ls.projected_positions().max()) > 10.0
    # Multipoint profile uses Spectrum index, collection order
    np.testing.assert_allclose(ls.distances(), np.arange(1, ls.n_points + 1))
    np.testing.assert_array_equal(ls.plot_order(), np.arange(ls.n_points))


def test_map_sum_stage_matches_map_extra_um(dylan, barstow):
    """Sum-spectrum stage centre matches MapExtra micrometre centre / 1000."""
    dylan_map = dylan.primary_fov
    assert dylan_map.stage_center_mm is not None
    np.testing.assert_allclose(dylan_map.stage_center_mm, (18.598, 3.033), atol=1e-3)

    barstow_map = next(f for f in barstow.fovs if f.element_maps or f.cube)
    assert barstow_map.stage_center_mm is not None
    np.testing.assert_allclose(barstow_map.stage_center_mm, (0.512, -12.57), atol=1e-3)


def test_barstow_point_series_is_multipoint(barstow):
    series_fovs = [f for f in barstow.fovs if f.line_scans]
    assert series_fovs, "expected numbered spectra FOV"
    ls = series_fovs[0].line_scans[0]
    assert ls.n_points >= 10
    assert ls.kind == "multipoint"
    assert all(p.x is not None and p.y is not None for p in ls.points)


def test_map_fov_has_stage_bounds(dylan, barstow):
    """Area maps expose stage centre + µm-derived size for camera registration."""
    dylan_map = dylan.primary_fov
    assert dylan_map is not None
    assert dylan_map.pixel_size_mm is not None
    # Real map pitch is ~14–18 µm, not the ~0.14 mm probe-related field
    assert 0.010 < dylan_map.pixel_size_mm < 0.030
    assert dylan_map.stage_center_mm is not None
    bounds = dylan_map.stage_bounds_mm()
    assert bounds is not None
    x0, y0, x1, y1 = bounds
    assert x1 > x0 and y1 > y0
    w_mm, h_mm = dylan_map.stage_size_mm
    np.testing.assert_allclose(x1 - x0, w_mm)
    np.testing.assert_allclose(y1 - y0, h_mm)
    # dylan map is a few mm across, not tens of mm
    assert w_mm < 5.0 and h_mm < 5.0

    barstow_map = next(f for f in barstow.fovs if f.element_maps or f.cube)
    assert barstow_map.pixel_size_mm is not None
    assert 0.010 < barstow_map.pixel_size_mm < 0.030
    assert barstow_map.stage_center_mm is not None
    bw, bh = barstow_map.stage_size_mm
    # barstow SmartMap is still a small patch (~7×5 mm), not half the stage
    assert bw < 12.0 and bh < 10.0
    assert barstow_map.stage_bounds_mm() is not None


def test_overlay_respects_dest_rect():
    from core.mapping.display import overlay_on_photo

    photo = np.zeros((100, 200, 3), dtype=np.float64)
    photo[:] = 0.2
    overlay = np.ones((10, 10, 3), dtype=np.float64)
    out = overlay_on_photo(
        photo, overlay, opacity=1.0, dest_rect=(50, 20, 80, 40)
    )
    np.testing.assert_allclose(out[0, 0], 0.2)
    assert out[25, 60].mean() > 0.9


def test_embed_map_on_photo_keeps_photo_shape():
    from core.mapping.display import embed_map_on_photo

    mmap = np.arange(20, dtype=np.float64).reshape(4, 5)
    full = embed_map_on_photo(mmap, (10, 12), dest_rect=None)
    assert full.shape == (10, 12)
    placed = embed_map_on_photo(mmap, (10, 12), dest_rect=(2, 1, 7, 5))
    assert placed.shape == (10, 12)
    assert placed[0, 0] == 0.0
    assert placed[1:5, 2:7].shape == (4, 5)
    assert placed[1:5, 2:7].max() > 0


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


def test_multipoint_distances_use_spectrum_index_not_path():
    """Multipoint profile stays in Spectrum order; path/projection stay available."""
    from core.mapping.models import LineScan, MapSpectrum
    from core.spectrum import Spectrum

    def _pt(i, x, y):
        sp = Spectrum(
            energy=np.arange(4, dtype=np.float64),
            counts=np.ones(4, dtype=np.float64),
            live_time=1.0,
            real_time=1.0,
            metadata={"name": f"Spectrum {i}"},
        )
        return MapSpectrum(
            spectrum=sp, name=f"Spectrum {i}", x=x, y=y, index=i, kind="line_point"
        )

    # Collect 0 → 1 → 2 → 1 (backtrack). Path travel = 3; spatial span = 2.
    pts = [_pt(1, 0.0, 0.0), _pt(2, 1.0, 0.0), _pt(3, 2.0, 0.0), _pt(4, 1.0, 0.0)]
    ls = LineScan(name="multi", points=pts, source="ipj", kind="multipoint")
    path = ls.path_distances()
    proj = ls.projected_positions()
    np.testing.assert_allclose(path, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(proj, [0.0, 1.0, 2.0, 1.0])
    np.testing.assert_allclose(ls.distances(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(ls.plot_order(), [0, 1, 2, 3])

    line_pts = [_pt(i, float(i), 0.0) for i in range(1, 6)]
    line = LineScan(
        name="line", points=line_pts, source="ipj", kind="line_scan"
    )
    np.testing.assert_allclose(line.distances(), line.path_distances())
    np.testing.assert_array_equal(line.plot_order(), np.arange(5))


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


def test_multipoint_site_gets_tube_settings_from_spectrum(dylan, barstow):
    """Sites without SmartMap still expose kV/mA via spectrum EMConditions."""
    # dylan Site 2 is multipoint (no SmartMap)
    multi = next(
        (f for f in dylan.fovs if not f.metadata.get("has_smartmap")),
        None,
    )
    assert multi is not None
    assert multi.metadata.get("kv") == 30.0
    assert multi.metadata.get("ma") == 15.0
    assert multi.metadata.get("map_live_time_s") is not None
    assert multi.spectra
    sm = multi.spectra[0].spectrum.metadata
    assert sm.get("excitation_energy") == 30.0
    assert sm.get("tube_current_ma") == 15.0
    assert sm.get("live_time")

    multi_b = next(
        (f for f in barstow.fovs if not f.metadata.get("has_smartmap")),
        None,
    )
    assert multi_b is not None
    assert multi_b.metadata.get("kv") == 50.0
    assert multi_b.metadata.get("ma") == 15.0


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
        assert sample.whole_image.metadata.get("orientation") == "rot180_mirror_lr"

    barstow_opt = next((f for f in barstow.fovs if f.optical is not None), None)
    assert barstow_opt is not None
    assert barstow_opt.optical.data.shape[2] == 3
    assert barstow_opt.optical.data.shape[0] > 100
    assert barstow_opt.optical.metadata.get("kind") == "map_area"
    assert barstow_opt.optical.metadata.get("orientation") == "rot180_mirror_lr"

    dylan_opt = next((f for f in dylan.fovs if f.optical is not None), None)
    assert dylan_opt is not None
    assert dylan_opt.optical.data.ndim == 3
    assert dylan_opt.optical.metadata.get("orientation") == "rot180_mirror_lr"


def test_map_area_thumbnail_is_crop_of_sample_camera(dylan, emerald, barstow):
    """Small MapAreaImage BMPs are exact crops of the sample-camera photo."""
    from core.mapping.camera import locate_image_crop

    for proj, expect_crop in ((dylan, True), (emerald, True), (barstow, False)):
        sample = proj.samples[0]
        fov = next(f for f in proj.fovs if f.optical is not None)
        photo = sample.whole_image.data
        opt = fov.optical.data
        rect = locate_image_crop(photo, opt)
        if not expect_crop:
            assert rect is None, proj.name
            continue
        assert rect is not None, proj.name
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        np.testing.assert_array_equal(photo[y0:y1, x0:x1], opt)


def test_point_spectra_skip_sum_and_spectra_only_flag(barstow, dylan):
    assert not barstow.is_spectra_only()
    assert barstow.has_spatial_data()
    points = barstow.point_spectra()
    assert points
    assert all("sum" not in p.name.lower() for p in points)
    assert dylan.has_spatial_data() or dylan.has_line_scans()
    assert not dylan.is_spectra_only()
    ls_fov = next(f for f in dylan.fovs if f.line_scans)
    ls_points = ls_fov.point_spectra()
    assert len(ls_points) == ls_fov.line_scans[0].n_points or len(ls_points) >= ls_fov.line_scans[0].n_points
