"""Map XGT stage millimetres onto the sample-camera bitmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# XGT-7200 sample stage travel is 100 × 100 mm. The overview BMP is 4:3
# (typically 2592×1944), so the long image axis is mapped to 100 mm with
# square pixels. Stage (0, 0) sits at the image centre; +X right, +Y up.
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


def camera_from_image(image, *, stage_travel_mm: float = XGT_STAGE_TRAVEL_MM) -> Optional[StageCamera]:
    """Build a StageCamera from a sample-camera OverviewImage (or HxW array)."""
    data = getattr(image, "data", image)
    arr = np.asarray(data)
    if arr.ndim < 2:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if width < 2 or height < 2:
        return None
    travel = float(stage_travel_mm)
    if width >= height:
        fov_w = travel
        fov_h = travel * height / width
    else:
        fov_h = travel
        fov_w = travel * width / height
    return StageCamera(
        width_px=width,
        height_px=height,
        fov_width_mm=fov_w,
        fov_height_mm=fov_h,
    )
