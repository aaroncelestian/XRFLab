"""Tests for map enhancement and neighborhood spectrum sums."""

import numpy as np

from core.mapping.cube import SpectrumCube
from core.mapping.display import (
    apply_intensity_scale,
    block_bin,
    enhance_map,
    format_acquisition,
    upsample_map,
)
from core.mapping.models import MappingFOV, MapSpectrum
from core.spectrum import Spectrum


def test_block_bin_averages_and_keeps_shape():
    data = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = block_bin(data, 2)
    assert out.shape == (4, 4)
    # top-left 2×2 mean is (0+1+4+5)/4 = 2.5
    np.testing.assert_allclose(out[:2, :2], 2.5)


def test_enhance_mean_blurs_impulse():
    data = np.zeros((7, 7), dtype=np.float64)
    data[3, 3] = 9.0
    raw = enhance_map(data, smooth="none", neighborhood=3)
    np.testing.assert_allclose(raw, data)
    mean = enhance_map(data, smooth="mean", neighborhood=3)
    assert mean[3, 3] < 9.0
    assert mean[3, 2] > 0
    np.testing.assert_allclose(mean.sum(), data.sum(), rtol=1e-6)


def test_upsample_cubic_is_smoother_than_nearest():
    data = np.zeros((6, 6), dtype=np.float64)
    data[2, 2] = 1.0
    near = upsample_map(data, factor=4, method="nearest")
    cubic = upsample_map(data, factor=4, method="cubic")
    assert near.shape == (24, 24)
    assert cubic.shape == (24, 24)
    # Nearest is piecewise-constant; cubic has intermediate values
    assert np.unique(near).size <= 2
    assert np.unique(np.round(cubic, 6)).size > 2
    rgb = np.zeros((5, 5, 3), dtype=np.float64)
    rgb[2, 2, 0] = 1.0
    out = upsample_map(rgb, factor=2, method="bilinear")
    assert out.shape == (10, 10, 3)


def test_intensity_scales_are_monotonic():
    data = np.array([[0.0, 1.0, 100.0]])
    sqrt = apply_intensity_scale(data, "sqrt")
    log = apply_intensity_scale(data, "log")
    assert sqrt[0, 1] < sqrt[0, 2]
    assert log[0, 1] < log[0, 2]
    assert sqrt[0, 2] < 100.0
    assert log[0, 2] < 100.0


def test_format_acquisition():
    text = format_acquisition(
        {
            "map_live_time_s": 300.0,
            "dwell_ms": 21.5,
            "n_pixels": 13952,
            "kv": 30.0,
            "ma": 15.0,
            "acquired_at": "2020-07-13 09:43:28",
        }
    )
    assert "300" in text
    assert "ms/pixel" in text
    assert "30 kV" in text
    assert "2020-07-13" in text


def test_spectrum_neighborhood_sums_window():
    cube = np.zeros((4, 5, 5), dtype=np.uint16)
    cube[:, 2, 2] = 1
    cube[:, 2, 3] = 2
    sc = SpectrumCube(data=cube, ev_per_channel=10.0)
    single, n1 = sc.spectrum_neighborhood(2, 2, size=1)
    np.testing.assert_allclose(single, 1.0)
    assert n1 == 1
    wide, n9 = sc.spectrum_neighborhood(2, 2, size=3)
    # center 1 + right 2; other 7 neighbors are 0
    np.testing.assert_allclose(wide, 3.0)
    assert n9 == 9


def test_spectrum_at_pixel_scales_live_time():
    nch, h, w = 8, 4, 5
    data = np.ones((nch, h, w), dtype=np.uint16)
    fov = MappingFOV(id="t", name="test", width=w, height=h)
    fov.cube = SpectrumCube(data=data, ev_per_channel=10.0)
    fov.metadata["dwell_s"] = 0.02
    energy = np.arange(nch, dtype=np.float64) * 0.01
    fov.spectra.append(
        MapSpectrum(
            spectrum=Spectrum(
                energy=energy,
                counts=np.ones(nch),
                live_time=0.4,
                real_time=0.4,
                metadata={"name": "Sum Spectrum"},
            ),
            name="Sum Spectrum",
            kind="sum",
        )
    )
    one = fov.spectrum_at_pixel(2, 2, neighborhood=1)
    three = fov.spectrum_at_pixel(2, 2, neighborhood=3)
    assert one is not None and three is not None
    assert one.spectrum.total_counts == nch
    assert three.spectrum.total_counts == nch * 9
    assert np.isclose(one.spectrum.live_time, 0.02)
    assert np.isclose(three.spectrum.live_time, 0.18)
    assert "3×3" in three.name
