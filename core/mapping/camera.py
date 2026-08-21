"""Map XGT stage millimetres onto the sample-camera bitmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# XGT-7200 sample stage travel is 100 × 100 mm. The overview BMP is 4:3
# (typically 2592×1944) and shows a wider camera FOV; Horiba hatches the
# margins outside the probeable square. Square pixels, 100 mm on the short
# image axis, stage (0, 0) at the image centre; +X right.
#
# After the XGT BMP orientation fix (180° + left/right mirror ≡ vertical flip
# of the decoded bitmap), stage +Y maps toward the *bottom* of the displayed
# photo (larger row index) so overlays match the corrected image.
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
        # +Y → larger row after XGT orientation correction (see module note)
        py = (self.height_px - 1) * 0.5 + (y_mm - self.origin_y_mm) / self.mm_per_px_y
        return float(px), float(py)

    def pixel_to_stage(self, px: float, py: float) -> Tuple[float, float]:
        x_mm = self.origin_x_mm + (px - (self.width_px - 1) * 0.5) * self.mm_per_px_x
        y_mm = self.origin_y_mm + (py - (self.height_px - 1) * 0.5) * self.mm_per_px_y
        return float(x_mm), float(y_mm)

    def stages_to_pixels(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        px = (self.width_px - 1) * 0.5 + (np.asarray(xs, dtype=np.float64) - self.origin_x_mm) / self.mm_per_px_x
        py = (self.height_px - 1) * 0.5 + (np.asarray(ys, dtype=np.float64) - self.origin_y_mm) / self.mm_per_px_y
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


def locate_scaled_template(
    image,
    template,
    target_width_px: float,
    target_height_px: float,
    *,
    center_xy: Optional[Tuple[float, float]] = None,
    search_px: float = 300.0,
    min_score: float = 0.45,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Locate a resized ``template`` inside ``image`` by normalized cross-correlation.

    Used when MapAreaImage is a magnified optical of the map FOV (not an exact
    crop of the sample camera). Prefer a search window around ``center_xy``
    (typically the stage-predicted centre) to avoid false matches.
    """
    img = np.asarray(image)
    templ = np.asarray(template)
    if img.ndim < 2 or templ.ndim < 2:
        return None
    ih, iw = int(img.shape[0]), int(img.shape[1])
    th = max(8, int(round(target_height_px)))
    tw = max(8, int(round(target_width_px)))
    if th >= ih or tw >= iw:
        return None
    try:
        from scipy.ndimage import zoom
        from scipy.signal import fftconvolve
    except ImportError:  # pragma: no cover
        return None

    gi = _as_gray_u8(img).astype(np.float64)
    gt = _as_gray_u8(templ).astype(np.float64)
    scaled = zoom(gt, (th / gt.shape[0], tw / gt.shape[1]), order=1)
    t = scaled - scaled.mean()
    denom_t = float(np.sqrt((t * t).sum()))
    if denom_t < 1e-9:
        return None

    if center_xy is None:
        ya, yb, xa, xb = 0, ih, 0, iw
    else:
        cx, cy = float(center_xy[0]), float(center_xy[1])
        half = float(search_px)
        ya = max(0, int(np.floor(cy - half - th * 0.5)))
        yb = min(ih, int(np.ceil(cy + half + th * 0.5)))
        xa = max(0, int(np.floor(cx - half - tw * 0.5)))
        xb = min(iw, int(np.ceil(cx + half + tw * 0.5)))
    region = gi[ya:yb, xa:xb]
    if region.shape[0] < th or region.shape[1] < tw:
        return None

    ones = np.ones_like(t)
    corr = fftconvolve(region, t[::-1, ::-1], mode="valid")
    sum_p = fftconvolve(region, ones, mode="valid")
    sum_p2 = fftconvolve(region * region, ones, mode="valid")
    n = float(t.size)
    var = np.maximum(sum_p2 - (sum_p * sum_p) / n, 0.0)
    ncc = corr / (np.sqrt(var) * denom_t + 1e-12)
    ncc = np.clip(ncc, -1.0, 1.0)
    iy, ix = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    if float(ncc[iy, ix]) < float(min_score):
        return None
    x0 = float(xa + ix)
    y0 = float(ya + iy)
    return x0, y0, x0 + float(tw), y0 + float(th)


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


def _as_rgb_u8(image) -> Optional[np.ndarray]:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    rgb = arr[:, :, :3]
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)
    g = rgb.astype(np.float64)
    if g.size and g.max() <= 1.5:
        g = g * 255.0
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def locate_red_map_rect(
    image,
    *,
    min_side_px: int = 12,
    max_aspect: float = 3.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Locate the Horiba-drawn red map-area rectangle on a sample-camera BMP.

    Returns interior pixel bounds ``(x0, y0, x1, y1)`` (inside the stroke), or
    None if no suitable hollow red rectangle is found.
    """
    rgb = _as_rgb_u8(image)
    if rgb is None:
        return None
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask = (r > 180) & (g < 100) & (b < 100) & (r > g + 60) & (r > b + 60)
    if int(mask.sum()) < 4 * int(min_side_px):
        return None

    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        return _red_rect_from_bbox(mask, min_side_px=min_side_px, max_aspect=max_aspect)

    labeled, n_labels = ndimage.label(mask)
    best = None
    best_score = -1.0
    for lab in range(1, int(n_labels) + 1):
        ys, xs = np.where(labeled == lab)
        if ys.size < 4 * int(min_side_px):
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        if w < min_side_px or h < min_side_px:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue
        sub = mask[y0:y1, x0:x1]
        # Hollow outline: edges red, interior mostly not
        edge = np.zeros_like(sub, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        if w > 6 and h > 6:
            edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
        edge_frac = float(sub[edge].mean()) if edge.any() else 0.0
        inner = sub[2:-2, 2:-2] if h > 6 and w > 6 else sub
        inner_frac = float(inner.mean()) if inner.size else 1.0
        if edge_frac < 0.15 or inner_frac > 0.45:
            continue
        # Prefer square-ish, strong outline, larger boxes
        score = edge_frac * (1.0 - inner_frac) * min(w, h) / (1.0 + abs(aspect - 1.0))
        if score > best_score:
            best_score = score
            # Inset by ~1 px so overlay sits inside the stroke
            best = (
                float(x0 + 1),
                float(y0 + 1),
                float(x1 - 1),
                float(y1 - 1),
            )
    if best is not None and best[2] > best[0] + 2 and best[3] > best[1] + 2:
        return best
    return _red_rect_from_bbox(mask, min_side_px=min_side_px, max_aspect=max_aspect)


def _red_rect_from_bbox(
    mask: np.ndarray,
    *,
    min_side_px: int,
    max_aspect: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Fallback: single bbox of all red pixels if it looks like a hollow rect."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w, h = x1 - x0, y1 - y0
    if w < min_side_px or h < min_side_px:
        return None
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > max_aspect:
        return None
    sub = mask[y0:y1, x0:x1]
    if float(sub.mean()) > 0.45:
        return None
    return float(x0 + 1), float(y0 + 1), float(x1 - 1), float(y1 - 1)


def calibrate_stage_camera(
    image,
    stage_center_mm: Tuple[float, float],
    pixel_rect: Tuple[float, float, float, float],
    stage_size_mm: Optional[Tuple[float, float]] = None,
    *,
    stage_travel_mm: float = XGT_STAGE_TRAVEL_MM,
) -> Optional[StageCamera]:
    """
    Build a StageCamera whose stage origin matches a known map rectangle.

    XGT sum-spectrum XY and the overview BMP do not always share the same
    origin as a naive image-centred model. Given one correspondence
    (stage centre ↔ pixel rect from MapAreaImage crop or red ROI), solve for
    ``origin_x_mm`` / ``origin_y_mm`` (and refine mm/px from rect size when
    ``stage_size_mm`` is known).
    """
    data = getattr(image, "data", image)
    arr = np.asarray(data)
    if arr.ndim < 2:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if width < 2 or height < 2:
        return None
    x0, y0, x1, y1 = (float(v) for v in pixel_rect)
    rw = max(x1 - x0, 1e-6)
    rh = max(y1 - y0, 1e-6)
    short = min(width, height)
    mpp = float(stage_travel_mm) / float(short)
    if stage_size_mm is not None:
        sw, sh = float(stage_size_mm[0]), float(stage_size_mm[1])
        if sw > 0 and sh > 0:
            mpp = 0.5 * (sw / rw + sh / rh)
    if not np.isfinite(mpp) or mpp <= 0:
        return None
    cx0 = (width - 1) * 0.5
    cy0 = (height - 1) * 0.5
    pcx = 0.5 * (x0 + x1)
    pcy = 0.5 * (y0 + y1)
    sx, sy = float(stage_center_mm[0]), float(stage_center_mm[1])
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return None
    # stage_to_pixel: px = cx0 + (x - ox)/mpp, py = cy0 + (y - oy)/mpp
    ox = sx - (pcx - cx0) * mpp
    oy = sy - (pcy - cy0) * mpp
    return StageCamera(
        width_px=width,
        height_px=height,
        fov_width_mm=mpp * width,
        fov_height_mm=mpp * height,
        origin_x_mm=float(ox),
        origin_y_mm=float(oy),
    )


def camera_from_sample_sites(
    image,
    sites,
    *,
    stage_travel_mm: float = XGT_STAGE_TRAVEL_MM,
) -> Optional[StageCamera]:
    """
    StageCamera for a sample overview, calibrated when a site provides a
    MapAreaImage match (exact crop or scaled optical) or red ROI plus stage
    geometry.

    Falls back to the default image-centred 100 mm model when no
    correspondence is available (multipoint-only projects without a map photo).
    """
    base = camera_from_image(image, stage_travel_mm=stage_travel_mm)
    photo = getattr(image, "data", image)
    photo_arr = np.asarray(photo)
    if base is None:
        return None

    best_cam = None
    best_rank = -1  # exact crop (3) > scaled optical (2) > red (1)
    for site in sites or []:
        center = getattr(site, "stage_center_mm", None)
        size = getattr(site, "stage_size_mm", None)
        if center is None:
            continue
        rect = None
        rank = 0
        optical = getattr(site, "optical", None)
        if optical is not None:
            opt = getattr(optical, "data", optical)
            rect = locate_image_crop(photo_arr, opt)
            if rect is not None:
                rank = 3
            elif size is not None:
                tw = float(size[0]) / base.mm_per_px_x
                th = float(size[1]) / base.mm_per_px_y
                cx, cy = base.stage_to_pixel(float(center[0]), float(center[1]))
                rect = locate_scaled_template(
                    photo_arr,
                    opt,
                    tw,
                    th,
                    center_xy=(cx, cy),
                )
                if rect is not None:
                    rank = 2
        if rect is None:
            red = locate_red_map_rect(photo_arr)
            if red is None:
                continue
            if size is not None:
                rw = red[2] - red[0]
                rh = red[3] - red[1]
                exp_w = float(size[0]) / base.mm_per_px_x
                exp_h = float(size[1]) / base.mm_per_px_y
                if exp_w > 1 and exp_h > 1:
                    scale_err = abs(rw - exp_w) / exp_w + abs(rh - exp_h) / exp_h
                    if scale_err > 0.75:
                        continue
            rect = red
            rank = 1
        cam = calibrate_stage_camera(
            photo_arr,
            center,
            rect,
            size,
            stage_travel_mm=stage_travel_mm,
        )
        if cam is not None and rank > best_rank:
            best_cam = cam
            best_rank = rank
            if rank >= 3:
                break
    return best_cam if best_cam is not None else base
