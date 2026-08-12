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
    # Spectrum shape
    s0 = all_spec[0].spectrum
    assert s0.num_channels == 8192
    assert s0.counts.max() > 0
    assert s0.energy[-1] > s0.energy[0]


def test_dylan_line_scan_and_maps(dylan):
    map_fov = dylan.primary_fov
    assert map_fov is not None
    assert map_fov.width == 128
    assert map_fov.height == 109
    assert len(map_fov.element_maps) >= 3
    names = {m.name for m in map_fov.element_maps}
    assert any("Fe" in n for n in names)
    assert any("Si" in n for n in names)

    # Line scan FOV with ~34 points
    line_fovs = [f for f in dylan.fovs if f.line_scans]
    assert line_fovs, "expected a line-scan FOV"
    ls = line_fovs[0].line_scans[0]
    assert ls.n_points >= 30
    assert ls.points[0].spectrum.num_channels == 8192


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


def test_hyperspectral_cube_dylan(dylan):
    fov = dylan.primary_fov
    assert fov is not None
    assert fov.cube is not None
    assert fov.cube.shape == (4096, 109, 128)
    # Sum spectrum matches cube totals (4096 paired bins)
    sum_ms = fov.sum_spectrum()
    assert sum_ms is not None
    pair = sum_ms.spectrum.counts.reshape(4096, 2).sum(axis=1)
    assert np.allclose(fov.cube.sum_spectrum(), pair)

    # Pixel extract expands to 8192 for Analysis
    ms = fov.spectrum_at_pixel(64, 54)
    assert ms is not None
    assert ms.spectrum.num_channels == 8192
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
    pair = sum_ms.spectrum.counts.reshape(4096, 2).sum(axis=1)
    assert np.allclose(fov.cube.sum_spectrum(), pair)
