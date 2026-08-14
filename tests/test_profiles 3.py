"""Tests for line-profile extraction, band averaging, and drawn line scans."""

import numpy as np

from core.mapping.cube import SpectrumCube
from core.mapping.models import ElementMap, MappingFOV
from core.mapping.profiles import (
    band_offsets,
    extract_array_profile,
    extract_cube_element_profiles,
    extract_line_profile,
    extract_multi_element_profiles,
    line_band_edges,
    perpendicular_unit,
)


def test_band_offsets_and_perpendicular():
    assert list(band_offsets(1)) == [0.0]
    np.testing.assert_allclose(band_offsets(3), [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(band_offsets(2), [-0.5, 0.5])

    px, py = perpendicular_unit((0.0, 0.0), (10.0, 0.0))
    np.testing.assert_allclose((px, py), (0.0, 1.0))


def test_line_profile_width_averages_neighbors():
    data = np.zeros((5, 9), dtype=np.float64)
    data[2, :] = 10.0  # center row
    data[1, :] = 4.0
    data[3, :] = 4.0
    em = ElementMap(name="Fe", data=data, element="Fe")

    _d, thin = extract_line_profile(em, (0, 2), (8, 2), width=1)
    _d, wide = extract_line_profile(em, (0, 2), (8, 2), width=3)

    np.testing.assert_allclose(thin, 10.0)
    np.testing.assert_allclose(wide, (4.0 + 10.0 + 4.0) / 3.0)
    assert wide.mean() < thin.mean()


def test_extract_cube_element_profiles_separate_rois():
    cube = np.zeros((20, 4, 8), dtype=np.uint16)
    # Channel 5 bright on the left, channel 15 bright on the right
    cube[5, :, :4] = 100
    cube[15, :, 4:] = 80
    sc = SpectrumCube(data=cube, ev_per_channel=1000.0)  # 1 keV / channel

    rois = [("Lo", 4.5, 5.5), ("Hi", 14.5, 15.5)]
    profiles = extract_cube_element_profiles(
        sc, rois, (0.0, 1.0), (7.0, 1.0), width=1
    )
    assert set(profiles) == {"Lo", "Hi"}
    lo = profiles["Lo"][1]
    hi = profiles["Hi"][1]
    assert lo[0] > lo[-1]
    assert hi[-1] > hi[0]


def test_multi_element_same_transect():
    a = ElementMap(name="Si", data=np.ones((6, 6)), element="Si")
    b = ElementMap(name="Fe", data=np.full((6, 6), 3.0), element="Fe")
    out = extract_multi_element_profiles([a, b], (0, 2), (5, 2), width=1)
    assert set(out) == {"Si", "Fe"}
    np.testing.assert_allclose(out["Si"][1], 1.0)
    np.testing.assert_allclose(out["Fe"][1], 3.0)
    np.testing.assert_allclose(out["Si"][0], out["Fe"][0])


def test_drawn_line_scan_from_cube():
    nch, h, w = 16, 6, 10
    data = np.zeros((nch, h, w), dtype=np.uint16)
    data[:, :, :] = 2
    data[3, :, :] = 20
    fov = MappingFOV(id="t", name="test", width=w, height=h)
    fov.cube = SpectrumCube(data=data, ev_per_channel=10.0)

    ls = fov.line_scan_from_drawn(0.0, 2.0, 9.0, 2.0, width=1)
    assert ls is not None
    assert ls.source == "drawn"
    assert ls.n_points >= 10
    assert ls.points[0].spectrum.num_channels == nch
    assert ls.points[0].spectrum.counts[3] == 20.0
    dist = ls.distances()
    assert dist[0] == 0.0
    assert dist[-1] > 0

    wide = fov.line_scan_from_drawn(0.0, 2.0, 9.0, 2.0, width=3)
    assert wide is not None
    assert wide.metadata.get("width_px") == 3
    assert wide.n_points == ls.n_points


def test_line_band_edges_span_width():
    a0, a1, b0, b1 = line_band_edges((0.0, 0.0), (10.0, 0.0), 4)
    # Perp is +y; half-width = 2
    np.testing.assert_allclose(a0, (0.0, 2.0))
    np.testing.assert_allclose(b0, (0.0, -2.0))
    np.testing.assert_allclose(a1, (10.0, 2.0))


def test_array_profile_width_matches_center_when_uniform():
    data = np.full((8, 8), 7.0)
    d1, v1 = extract_array_profile(data, (0, 3), (7, 3), width=1)
    d5, v5 = extract_array_profile(data, (0, 3), (7, 3), width=5)
    np.testing.assert_allclose(v1, v5)
    np.testing.assert_allclose(d1, d5)


def test_coerce_element_symbols_from_analysis_dicts():
    from core.mapping.models import coerce_element_symbols, _element_from_line_name

    assert coerce_element_symbols(
        [{"symbol": "Ca", "z": 20, "name": "Calcium"}, {"symbol": "Cr", "z": 24}]
    ) == ["Ca", "Cr"]
    assert coerce_element_symbols(["Fe", "Si"]) == ["Fe", "Si"]
    assert coerce_element_symbols([str({"symbol": "Ca"})]) == []
    assert coerce_element_symbols(None) == []
    assert _element_from_line_name("Ca Ka1") == "Ca"
    assert _element_from_line_name("Cr Ka1") == "Cr"
    assert _element_from_line_name("Na Ka1_2") == "Na"
    assert _element_from_line_name("Total counts (cube)") == ""
    em = ElementMap(name="Ca Ka1", data=np.ones((4, 4)))
    assert em.element == "Ca"
    total = ElementMap(
        name="Total counts (cube)",
        data=np.ones((4, 4)),
        metadata={"source": "cube_total"},
    )
    assert total.element == ""
