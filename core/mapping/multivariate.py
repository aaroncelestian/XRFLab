"""Multivariate analysis and particle finding on element maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import (
        binary_opening,
        distance_transform_edt,
        label,
        maximum_filter,
    )
except ImportError:  # pragma: no cover
    binary_opening = None
    distance_transform_edt = None
    label = None
    maximum_filter = None

from core.mapping.models import ElementMap


@dataclass
class PCAResult:
    """PCA on stacked element maps."""

    score_maps: List[ElementMap]
    loadings: np.ndarray  # (n_components, n_maps)
    explained_variance_ratio: np.ndarray
    map_names: List[str]
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class Particle:
    """One connected bright region on a map."""

    id: int
    label: int
    area_px: int
    centroid_xy: Tuple[float, float]  # (x, y) in map pixels
    mean_intensity: float
    max_intensity: float
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    mean_elements: Dict[str, float] = field(default_factory=dict)


@dataclass
class ParticleResult:
    """Labeled particle mask plus table of particles."""

    label_map: np.ndarray
    particles: List[Particle]
    source_map: str
    threshold: float


def pca_element_maps(
    maps: Sequence[ElementMap],
    *,
    n_components: int = 3,
    mask_zeros: bool = True,
    standardize: bool = True,
) -> PCAResult:
    """
    PCA on pixel vectors from stacked element maps.

    Each pixel is a feature vector of map intensities. Returns score maps
    PC1…PCk as ElementMap objects (metadata source=\"pca\").
    """
    maps = list(maps)
    if len(maps) < 2:
        raise ValueError("PCA needs at least two element maps")
    shape = maps[0].shape
    for m in maps:
        if m.shape != shape:
            raise ValueError(f"Map shape mismatch: {m.name} {m.shape} vs {shape}")

    stack = np.stack([np.asarray(m.data, dtype=np.float64) for m in maps], axis=-1)
    h, w, n_feat = stack.shape
    flat = stack.reshape(-1, n_feat)

    if mask_zeros:
        keep = np.any(flat > 0, axis=1) & np.all(np.isfinite(flat), axis=1)
    else:
        keep = np.all(np.isfinite(flat), axis=1)
    if int(np.count_nonzero(keep)) < 2:
        raise ValueError("Not enough finite pixels for PCA")

    X = flat[keep]
    mean = X.mean(axis=0)
    Xc = X - mean
    if standardize:
        scale = X.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        Xc = Xc / scale
    else:
        scale = np.ones(n_feat, dtype=np.float64)

    # SVD: Xc = U S Vt
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = int(max(1, min(n_components, n_feat, vt.shape[0])))
    loadings = vt[:n_comp]
    # Scores for kept pixels
    scores_kept = Xc @ loadings.T

    # Variance explained from singular values
    var = (s ** 2) / max(Xc.shape[0] - 1, 1)
    total = float(var.sum()) if var.size else 1.0
    explained = var[:n_comp] / total if total > 0 else np.zeros(n_comp)

    # Full score maps (zeros where masked out)
    full_scores = np.zeros((flat.shape[0], n_comp), dtype=np.float64)
    full_scores[keep] = scores_kept
    score_maps: List[ElementMap] = []
    names = [m.name for m in maps]
    for i in range(n_comp):
        data = full_scores[:, i].reshape(h, w)
        pct = float(explained[i] * 100.0)
        label = f"PC{i + 1}"
        score_maps.append(
            ElementMap(
                name=label,
                data=data,
                line=label,
                element="",
                metadata={
                    "source": "pca",
                    "explained_variance_pct": pct,
                    "input_maps": list(names),
                    "n_components": n_comp,
                },
            )
        )

    return PCAResult(
        score_maps=score_maps,
        loadings=loadings,
        explained_variance_ratio=np.asarray(explained, dtype=np.float64),
        map_names=names,
        mean=mean,
        scale=scale,
    )


def find_particles(
    data: np.ndarray,
    *,
    threshold_percentile: float = 90.0,
    min_area: int = 5,
    open_radius: int = 1,
    source_map: str = "",
    element_maps: Optional[Sequence[ElementMap]] = None,
    use_watershed: bool = True,
) -> ParticleResult:
    """
    Find bright particles via threshold + connected components.

    Optionally splits touching blobs with a simple distance-transform watershed.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("find_particles expects a 2D array")
    if label is None:
        raise ImportError("scipy.ndimage is required for particle finding")

    finite = np.isfinite(arr) & (arr > 0)
    if not np.any(finite):
        return ParticleResult(
            label_map=np.zeros(arr.shape, dtype=np.int32),
            particles=[],
            source_map=source_map,
            threshold=0.0,
        )

    thr = float(np.percentile(arr[finite], threshold_percentile))
    mask = finite & (arr >= thr)

    if open_radius > 0 and binary_opening is not None:
        struct = np.ones((2 * open_radius + 1, 2 * open_radius + 1), dtype=bool)
        mask = binary_opening(mask, structure=struct)

    if use_watershed and distance_transform_edt is not None and maximum_filter is not None:
        labels = _watershed_split(mask, arr)
    else:
        labels, _ = label(mask)

    particles = _particles_from_labels(
        labels, arr, min_area=min_area, element_maps=element_maps
    )
    # Relabel to contiguous 1..N matching particle ids
    new_labels = np.zeros_like(labels, dtype=np.int32)
    for p in particles:
        new_labels[labels == p.label] = p.id
        p.label = p.id

    return ParticleResult(
        label_map=new_labels,
        particles=particles,
        source_map=source_map,
        threshold=thr,
    )


def _watershed_split(mask: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Split touching blobs using distance-transform peaks as seeds."""
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)
    dist = distance_transform_edt(mask)
    # Peaks in distance map
    neighborhood = maximum_filter(dist, size=3)
    peaks = (dist == neighborhood) & (dist > 0) & mask
    seeds, n_seeds = label(peaks)
    if n_seeds == 0:
        labeled, _ = label(mask)
        return labeled

    # Simple marker-controlled watershed via iterative expansion on -dist
    # Priority: expand from seeds into mask by descending distance
    labels = seeds.astype(np.int32).copy()
    # Flatten candidates sorted by distance descending
    ys, xs = np.where(mask & (labels == 0))
    if ys.size == 0:
        return labels
    order = np.argsort(-dist[ys, xs])
    ys, xs = ys[order], xs[order]
    h, w = mask.shape
    for y, x in zip(ys, xs):
        # Neighbor labels
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        neigh = labels[y0:y1, x0:x1]
        vals = neigh[neigh > 0]
        if vals.size == 0:
            continue
        # Assign majority / first unique
        labels[y, x] = int(np.bincount(vals).argmax())
    # Anything still unlabeled in mask → connected component fill from nearest
    leftover = mask & (labels == 0)
    if np.any(leftover):
        extra, _ = label(leftover)
        offset = int(labels.max())
        labels[extra > 0] = extra[extra > 0] + offset
    return labels


def _particles_from_labels(
    labels: np.ndarray,
    intensity: np.ndarray,
    *,
    min_area: int,
    element_maps: Optional[Sequence[ElementMap]],
) -> List[Particle]:
    particles: List[Particle] = []
    max_lab = int(labels.max()) if labels.size else 0
    next_id = 1
    for lab in range(1, max_lab + 1):
        ys, xs = np.where(labels == lab)
        area = int(xs.size)
        if area < min_area:
            continue
        vals = intensity[ys, xs]
        cy = float(ys.mean())
        cx = float(xs.mean())
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        means: Dict[str, float] = {}
        if element_maps:
            for em in element_maps:
                if em.data.shape == intensity.shape:
                    means[em.name] = float(np.asarray(em.data, dtype=np.float64)[ys, xs].mean())
        particles.append(
            Particle(
                id=next_id,
                label=lab,
                area_px=area,
                centroid_xy=(cx, cy),
                mean_intensity=float(vals.mean()),
                max_intensity=float(vals.max()),
                bbox=(x0, y0, x1, y1),
                mean_elements=means,
            )
        )
        next_id += 1
    return particles


def particle_label_map_as_element(
    result: ParticleResult,
    name: str = "Particles",
) -> ElementMap:
    """Convert a ParticleResult label mask into an ElementMap for the Data tree."""
    return ElementMap(
        name=name,
        data=result.label_map.astype(np.float64),
        line=name,
        element="",
        metadata={
            "source": "particles",
            "n_particles": len(result.particles),
            "threshold": result.threshold,
            "source_map": result.source_map,
        },
    )
