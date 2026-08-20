"""Tests for map correlation matrix, enhancement, PCA, and particles."""

import numpy as np
import pytest

from core.mapping.correlations import map_correlation, map_correlation_matrix
from core.mapping.display import (
    apply_contrast,
    bilateral_filter,
    clahe_2d,
    enhance_map,
    percentile_stretch,
    ratio_map,
    tophat_enhance,
)
from core.mapping.models import ElementMap, MappingFOV
from core.mapping.multivariate import (
    find_particles,
    particle_label_map_as_element,
    pca_element_maps,
)


def _maps():
    rng = np.random.default_rng(0)
    base = rng.random((16, 16))
    m0 = ElementMap(name="Fe", data=base)
    m1 = ElementMap(name="Si", data=base * 0.5 + rng.random((16, 16)) * 0.1)
    m2 = ElementMap(name="Ca", data=1.0 - base + rng.random((16, 16)) * 0.05)
    return m0, m1, m2


def test_map_correlation_matrix_symmetric():
    maps = list(_maps())
    matrix, names = map_correlation_matrix(maps)
    assert names == ["Fe", "Si", "Ca"]
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, matrix.T, equal_nan=True)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    # Fe and Si should be positively correlated
    assert matrix[0, 1] > 0.5


def test_bilateral_preserves_edge_better_than_mean():
    data = np.zeros((21, 21), dtype=np.float64)
    data[:, 11:] = 10.0
    bi = bilateral_filter(data, size=5)
    mean = enhance_map(data, smooth="mean", neighborhood=5)
    # Edge pixel jump should stay sharper for bilateral
    edge_bi = abs(bi[10, 11] - bi[10, 10])
    edge_mean = abs(mean[10, 11] - mean[10, 10])
    assert edge_bi >= edge_mean - 1e-9


def test_contrast_percentile_and_clahe():
    data = np.linspace(0, 100, 64).reshape(8, 8)
    stretched = percentile_stretch(data)
    assert stretched.min() >= 0.0
    assert stretched.max() <= 1.0
    clahe = clahe_2d(data, tile_grid=2)
    assert clahe.shape == data.shape
    hat = tophat_enhance(data, size=3)
    assert hat.shape == data.shape
    out = enhance_map(data, contrast="clahe")
    assert out.shape == data.shape
    out2 = apply_contrast(data, "percentile")
    assert out2.max() <= 1.0 + 1e-9


def test_ratio_map_zeros_low_denominator():
    num = np.ones((4, 4))
    den = np.zeros((4, 4))
    den[1, 1] = 10.0
    out = ratio_map(num, den, eps=1.0)
    assert out[1, 1] == pytest.approx(0.1)
    assert out[0, 0] == 0.0


def test_pca_score_maps():
    maps = list(_maps())
    result = pca_element_maps(maps, n_components=2)
    assert len(result.score_maps) == 2
    assert result.score_maps[0].name == "PC1"
    assert result.score_maps[0].metadata["source"] == "pca"
    assert result.explained_variance_ratio.sum() <= 1.0 + 1e-6
    assert result.loadings.shape == (2, 3)
    # Scores cover full map
    assert result.score_maps[0].shape == maps[0].shape


def test_find_particles_on_blobs():
    data = np.zeros((40, 40), dtype=np.float64)
    data[5:10, 5:10] = 80.0
    data[25:32, 20:28] = 80.0
    result = find_particles(
        data,
        threshold_percentile=50.0,
        min_area=4,
        source_map="Fe",
        use_watershed=False,
    )
    assert len(result.particles) >= 2
    assert result.label_map.max() == len(result.particles)
    em = particle_label_map_as_element(result)
    assert em.metadata["source"] == "particles"
    assert em.metadata["n_particles"] == len(result.particles)


def test_fov_upsert_and_remove_by_source():
    fov = MappingFOV(id="t", name="t", width=8, height=8)
    em = ElementMap(
        name="PC1",
        data=np.zeros((8, 8)),
        metadata={"source": "pca"},
    )
    fov.upsert_map(em)
    assert fov.find_map("PC1") is not None
    fov.upsert_map(
        ElementMap(name="PC1", data=np.ones((8, 8)), metadata={"source": "pca"})
    )
    assert len([m for m in fov.element_maps if m.name == "PC1"]) == 1
    np.testing.assert_allclose(fov.find_map("PC1").data, 1.0)
    n = fov.remove_maps_by_source("pca")
    assert n == 1
    assert fov.find_map("PC1") is None


def test_map_correlation_spearman_matrix():
    maps = list(_maps())
    matrix, _ = map_correlation_matrix(maps, method="spearman")
    assert matrix.shape == (3, 3)
    _, _, _r, rho = map_correlation(maps[0], maps[1])
    if np.isfinite(rho):
        assert matrix[0, 1] == pytest.approx(rho)
