"""Tests for area-ROI masks and cube region sums."""

import numpy as np

from core.mapping.cube import SpectrumCube
from core.mapping.models import MappingFOV, MapSpectrum
from core.mapping.regions import circle_mask, polygon_mask, rect_mask, region_label
from core.spectrum import Spectrum


def test_rect_mask_covers_inclusive_box():
    mask = rect_mask(6, 8, 1.0, 1.0, 3.0, 3.0)
    # pixel centers at 1.5 and 2.5 are inside [1, 3]
    assert mask[1, 1] and mask[2, 2]
    assert not mask[0, 0]
    assert not mask[4, 4]


def test_circle_mask_radius():
    mask = circle_mask(11, 11, 5.5, 5.5, 2.0)
    assert mask[5, 5]
    assert mask[5, 7]
    assert not mask[5, 9]


def test_polygon_triangle_mask():
    verts = [(1.0, 1.0), (8.0, 1.0), (4.5, 7.0)]
    mask = polygon_mask(8, 10, verts)
    assert mask.sum() > 5
    assert mask[2, 4]
    assert not mask[7, 0]


def test_cube_spectrum_in_mask_sums_only_selected():
    nch, h, w = 5, 4, 6
    data = np.zeros((nch, h, w), dtype=np.uint16)
    data[:, 1, 2] = 3
    data[:, 1, 3] = 5
    data[:, 0, 0] = 100
    cube = SpectrumCube(data=data, ev_per_channel=10.0)
    mask = np.zeros((h, w), dtype=bool)
    mask[1, 2] = True
    mask[1, 3] = True
    counts, n = cube.spectrum_in_mask(mask)
    assert n == 2
    np.testing.assert_allclose(counts, 8.0)


def test_fov_region_sum_and_live_time():
    nch, h, w = 8, 6, 8
    data = np.ones((nch, h, w), dtype=np.uint16)
    fov = MappingFOV(id="t", name="test", width=w, height=h)
    fov.cube = SpectrumCube(data=data, ev_per_channel=10.0)
    fov.metadata["dwell_s"] = 0.01
    energy = np.arange(nch, dtype=np.float64) * 0.01
    fov.spectra.append(
        MapSpectrum(
            spectrum=Spectrum(
                energy=energy,
                counts=np.ones(nch),
                live_time=0.48,
                metadata={"name": "Sum Spectrum"},
            ),
            name="Sum Spectrum",
            kind="sum",
        )
    )
    ms = fov.spectrum_in_region("rect", (1.0, 1.0, 3.0, 3.0))
    assert ms is not None
    assert ms.kind == "roi"
    assert "Rect" in ms.name
    n = ms.metadata["n_pixels"]
    assert n >= 1
    assert np.isclose(ms.spectrum.live_time, 0.01 * n)
    # Must not steal the FOV sum spectrum
    assert fov.sum_spectrum() is not None
    assert fov.sum_spectrum().kind == "sum"
    assert region_label("circle", (2.0, 2.0, 3.5), 10).startswith("Circle")
