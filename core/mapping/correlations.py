"""Element-map correlation helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from core.mapping.models import ElementMap


def map_correlation(
    map_a: ElementMap,
    map_b: ElementMap,
    *,
    mask_zeros: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Flatten two maps for scatter / correlation analysis.

    Returns:
        x values, y values, Pearson r, Spearman rho (rank correlation)
    """
    a = np.asarray(map_a.data, dtype=np.float64).ravel()
    b = np.asarray(map_b.data, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"Map shapes differ: {map_a.shape} vs {map_b.shape}"
        )
    if mask_zeros:
        keep = (a > 0) | (b > 0)
        if not np.any(keep):
            keep = np.ones_like(a, dtype=bool)
        a = a[keep]
        b = b[keep]

    if a.size < 2:
        return a, b, float("nan"), float("nan")

    pearson = float(np.corrcoef(a, b)[0, 1])
    # Spearman via rank
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    return a, b, pearson, spearman


def rgb_composite(
    r_map: ElementMap | None,
    g_map: ElementMap | None,
    b_map: ElementMap | None,
    *,
    percentile: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Build an RGB image (H, W, 3) float in [0, 1] from up to three maps.
    Missing channels are zeros. Maps are independently percentile-scaled.
    """
    ref = r_map or g_map or b_map
    if ref is None:
        raise ValueError("At least one map is required")
    h, w = ref.shape
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for i, m in enumerate((r_map, g_map, b_map)):
        if m is None:
            continue
        if m.shape != (h, w):
            raise ValueError("All RGB maps must share the same shape")
        rgb[:, :, i] = _scale_percentile(m.data, percentile)
    return rgb


def _scale_percentile(
    data: np.ndarray,
    percentile: Tuple[float, float],
) -> np.ndarray:
    lo, hi = np.percentile(data, percentile)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((data.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
