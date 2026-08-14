"""Map XGT stage millimetres onto the sample-camera bitmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# XGT-7200 sample stage travel is 100 × 100 mm. The overview BMP is 4:3
# (typically 2592×1944) and shows a wider camera FOV; Horiba hatches the
# margins outside the probeable square. Square pixels, 100 mm on the short
# image axis, stage (0, 0) at the image centre; +X right, +Y up.
XGT_STAGE_TRAVEL_MM = 100.0


@dataclass
class StageCamera:
    """Affine map between stage millimetres and sample-camera pixels."""

    width_px: int
    height_px: int
    fov_width_mm: float
    fov_height_mm: float
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0

    @property
    def mm_per_px_x(self) -> float:
        return float(self.fov_width_mm) / max(int(self.width_px), 1)

    @property
    def mm_per_px_y(self) -> float:
        return float(self.fov_height_mm) / max(int(self.height_px), 1)

    def stage_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """Stage (mm) → image pixel (col, row), origin top-left of the BMP."""
        px = (self.width_px - 1) * 0.5 + (x_mm - self.origin_x_mm) / self.mm_per_px_x
        py = (self.height_px - 1) * 0.5 - (y_mm - self.origin_y_mm) / self.mm_per_px_y
        return float(px), float(py)

    def pixel_to_stage(self, px: float, py: float) -> Tuple[float, float]:
        x_mm = self.origin_x_mm + (px - (self.width_px - 1) * 0.5) * self.mm_per_px_x
        y_mm = self.origin_y_mm - (py - (self.height_px - 1) * 0.5) * self.mm_per_px_y
        return float(x_mm), float(y_mm)

    def stages_to_pixels(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        px = (self.width_px - 1) * 0.5 + (np.asarray(xs, dtype=np.float64) - self.origin_x_mm) / self.mm_per_px_x
        py = (self.height_px - 1) * 0.5 - (np.asarray(ys, dtype=np.float64) - self.origin_y_mm) / self.mm_per_px_y
        return px, py

    def stage_bounds_to_pixel_rect(
        self, bounds_mm: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Map stage (x0, y0, x1, y1) mm → photo pixel (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = bounds_mm
        xs = np.array([x0, x1, x1, x0], dtype=np.float64)
        ys = np.array([y0, y0, y1, y1], dtype=np.float64)
        px, py = self.stages_to_pixels(xs, ys)
        return float(px.min()), float(py.min()), float(px.max()), float(py.max())

    def probeable_pixel_rect(self) -> Tuple[float, float, float, float]:
        """Photo pixels covering the 100×100 mm stage (centred square)."""
        half = min(self.fov_width_mm, self.fov_height_mm) * 0.5
        return self.stage_bounds_to_pixel_rect((-half, -half, half, half))


def camera_from_image(image, *, stage_travel_mm: float = XGT_STAGE_TRAVEL_MM) -> Optional[StageCamera]:
    """Build a StageCamera from a sample-camera OverviewImage (or HxW array).

    The 100×100 mm stage is a centred square on the photo (the short image
    axis spans ``stage_travel_mm``). Extra pixels on the long axis are
    camera FOV outside the probeable area.
    """
    data = getattr(image, "data", image)
    arr = np.asarray(data)
    if arr.ndim < 2:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if width < 2 or height < 2:
        return None
    travel = float(stage_travel_mm)
    short = min(width, height)
    mm_per_px = travel / short
    return StageCamera(
        width_px=width,
        height_px=height,
        fov_width_mm=mm_per_px * width,
        fov_height_mm=mm_per_px * height,
    )


def _as_gray_u8(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        gray = np.mean(arr[:, :, :3], axis=2)
    else:
        gray = arr
    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)
    g = gray.astype(np.float64)
    if g.size and g.max() <= 1.5:
        g = g * 255.0
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def locate_image_crop(
    image,
    crop,
    *,
    min_score: float = 0.98,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Locate ``crop`` as a pixel-aligned sub-image of ``image``.

    XGT map-area thumbnails are stored as exact crops of the sample-camera
    BMP. Returns photo-pixel ``(x0, y0, x1, y1)``, or None if the crop is
    larger than the photo or no match is found.
    """
    img = np.asarray(image)
    templ = np.asarray(crop)
    if img.ndim < 2 or templ.ndim < 2:
        return None
    ih, iw = int(img.shape[0]), int(img.shape[1])
    th, tw = int(templ.shape[0]), int(templ.shape[1])
    if th < 2 or tw < 2 or th >= ih or tw >= iw:
        return None

    gi = _as_gray_u8(img)
    gt = _as_gray_u8(templ)
    found = _locate_exact_gray_crop(gi, gt)
    if found is None:
        found = _locate_ncc_gray_crop(gi, gt, min_score=min_score)
    if found is None:
        return None
    x0, y0 = found
    return float(x0), float(y0), float(x0 + tw), float(y0 + th)


def _locate_exact_gray_crop(
    image: np.ndarray, crop: np.ndarray
) -> Optional[Tuple[int, int]]:
    """Return (x0, y0) if ``crop`` appears exactly in ``image``."""
    ih, iw = image.shape
    th, tw = crop.shape
    row_var = crop.var(axis=1)
    ri = int(np.argmax(row_var)) if row_var.size else 0
    probe = crop[ri]
    n_probe = min(12, tw)
    idx = np.linspace(0, tw - 1, num=n_probe, dtype=int)
    windows = sliding_window_view(image, tw, axis=1)[:, :, idx]
    hits = np.all(windows == probe[idx], axis=-1)
    ys, xs = np.where(hits)
    if ys.size == 0 or ys.size > 2000:
        return None
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0 = int(y) - ri
        x0 = int(x)
        if y0 < 0 or x0 < 0 or y0 + th > ih or x0 + tw > iw:
            continue
        if np.array_equal(image[y0 : y0 + th, x0 : x0 + tw], crop):
            return x0, y0
    return None


def _locate_ncc_gray_crop(
    image: np.ndarray,
    crop: np.ndarray,
    *,
    min_score: float,
) -> Optional[Tuple[int, int]]:
    """Normalized cross-correlation fallback for near-exact crops."""
    try:
        from scipy.signal import fftconvolve
    except ImportError:  # pragma: no cover
        return None
    img = image.astype(np.float64)
    t = crop.astype(np.float64)
    t0 = t - t.mean()
    denom_t = float(np.sqrt((t0 * t0).sum()))
    if denom_t < 1e-9:
        return None
    ones = np.ones_like(t0)
    corr = fftconvolve(img, t0[::-1, ::-1], mode="valid")
    sum_p = fftconvolve(img, ones, mode="valid")
    sum_p2 = fftconvolve(img * img, ones, mode="valid")
    n = float(t0.size)
    var = np.maximum(sum_p2 - (sum_p * sum_p) / n, 0.0)
    ncc = corr / (np.sqrt(var) * denom_t + 1e-12)
    iy, ix = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    if float(ncc[iy, ix]) < float(min_score):
        return None
    return int(ix), int(iy)
