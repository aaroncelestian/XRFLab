"""Tests for stage-millimetre ↔ sample-camera pixel mapping."""

import numpy as np

from core.mapping.camera import StageCamera, camera_from_image
from core.mapping.models import OverviewImage


def test_stage_camera_roundtrip_center_and_offset():
    cam = StageCamera(
        width_px=2592,
        height_px=1944,
        fov_width_mm=100.0,
        fov_height_mm=75.0,
    )
    cx, cy = cam.stage_to_pixel(0.0, 0.0)
    np.testing.assert_allclose(cx, (2592 - 1) / 2.0)
    np.testing.assert_allclose(cy, (1944 - 1) / 2.0)

    px, py = cam.stage_to_pixel(10.0, 5.0)
    x, y = cam.pixel_to_stage(px, py)
    np.testing.assert_allclose([x, y], [10.0, 5.0], atol=1e-9)

    # +Y is toward the top of the image (smaller row index)
    _px0, py0 = cam.stage_to_pixel(0.0, 0.0)
    _px1, py1 = cam.stage_to_pixel(0.0, 10.0)
    assert py1 < py0


def test_camera_from_image_inscribes_100mm_stage_square():
    img = OverviewImage(
        name="Sample camera",
        data=np.zeros((1944, 2592, 3), dtype=np.uint8),
        metadata={"kind": "whole_image"},
    )
    cam = camera_from_image(img)
    assert cam is not None
    assert cam.width_px == 2592
    assert cam.height_px == 1944
    # Short axis (height) is 100 mm; extra width is camera-only margin
    np.testing.assert_allclose(cam.fov_height_mm, 100.0)
    np.testing.assert_allclose(cam.fov_width_mm, 100.0 * 2592 / 1944)
    np.testing.assert_allclose(cam.mm_per_px_x, cam.mm_per_px_y)

    x0, y0, x1, y1 = cam.probeable_pixel_rect()
    # 100×100 mm square fills the image height and is inset left/right
    np.testing.assert_allclose(y1 - y0, 1944.0, atol=2.0)
    np.testing.assert_allclose(x1 - x0, 1944.0, atol=2.0)
    assert x0 > 200
    assert x1 < 2592 - 200

    xs = np.array([6.6, 6.6])
    ys = np.array([-5.8, 5.5])
    px, py = cam.stages_to_pixels(xs, ys)
    assert px[0] > (2592 - 1) / 2.0  # +X is to the right of centre
    assert py[1] < py[0]  # more positive Y is higher on the photo


def test_stage_bounds_to_pixel_rect():
    cam = StageCamera(
        width_px=1000,
        height_px=800,
        fov_width_mm=100.0,
        fov_height_mm=80.0,
    )
    # 10×10 mm box centred on stage origin → centred on image
    x0, y0, x1, y1 = cam.stage_bounds_to_pixel_rect((-5.0, -5.0, 5.0, 5.0))
    assert x0 < 499.5 < x1
    assert y0 < 399.5 < y1
    np.testing.assert_allclose(x1 - x0, 100.0, atol=1.0)
    np.testing.assert_allclose(y1 - y0, 100.0, atol=1.0)


def test_locate_image_crop_finds_exact_subimage():
    from core.mapping.camera import locate_image_crop

    photo = np.zeros((80, 120, 3), dtype=np.uint8)
    photo[10:40, 25:70] = (12, 80, 200)
    photo[20, 40] = (255, 255, 0)
    crop = photo[10:40, 25:70].copy()
    rect = locate_image_crop(photo, crop)
    assert rect == (25.0, 10.0, 70.0, 40.0)
    # Larger template cannot be a crop
    assert locate_image_crop(crop, photo) is None


def test_locate_image_crop_uses_distinctive_row():
    from core.mapping.camera import locate_image_crop

    photo = np.full((60, 90), 7, dtype=np.uint8)
    crop = np.full((20, 15), 7, dtype=np.uint8)
    crop[12] = np.arange(15, dtype=np.uint8) + 40
    photo[30:50, 10:25] = crop
    rect = locate_image_crop(photo, crop)
    assert rect == (10.0, 30.0, 25.0, 50.0)
