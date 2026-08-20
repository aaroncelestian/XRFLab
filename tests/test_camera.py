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


def test_locate_red_map_rect_finds_hollow_square():
    from core.mapping.camera import locate_red_map_rect

    photo = np.zeros((200, 300, 3), dtype=np.uint8)
    photo[:] = (30, 40, 35)
    # Hollow red square at (80,50)-(140,110)
    photo[50:111, 80:141] = (30, 40, 35)
    photo[50:53, 80:141] = (255, 20, 10)
    photo[108:111, 80:141] = (255, 20, 10)
    photo[50:111, 80:83] = (255, 20, 10)
    photo[50:111, 138:141] = (255, 20, 10)
    rect = locate_red_map_rect(photo)
    assert rect is not None
    x0, y0, x1, y1 = rect
    assert 78 <= x0 <= 84
    assert 48 <= y0 <= 54
    assert 136 <= x1 <= 142
    assert 106 <= y1 <= 112


def test_locate_red_map_rect_ignores_noise():
    from core.mapping.camera import locate_red_map_rect

    photo = np.zeros((100, 120, 3), dtype=np.uint8)
    photo[10, 10] = (255, 0, 0)
    photo[50, 60] = (200, 30, 20)
    assert locate_red_map_rect(photo) is None


def test_calibrate_stage_camera_aligns_center_and_size():
    from core.mapping.camera import calibrate_stage_camera

    photo = np.zeros((1944, 2592, 3), dtype=np.uint8)
    # Fake map rect on the right side of the photo
    rect = (1684.0, 873.0, 1726.0, 909.0)
    center = (6.56756591796875, 5.240478515625)
    size = (2.304, 1.962)
    cam = calibrate_stage_camera(photo, center, rect, size)
    assert cam is not None
    px, py = cam.stage_to_pixel(*center)
    np.testing.assert_allclose([px, py], [1705.0, 891.0], atol=0.5)
    mapped = cam.stage_bounds_to_pixel_rect(
        (center[0] - size[0] / 2, center[1] - size[1] / 2,
         center[0] + size[0] / 2, center[1] + size[1] / 2)
    )
    np.testing.assert_allclose(mapped, rect, atol=1.5)


def test_locate_scaled_template_near_center():
    from core.mapping.camera import locate_scaled_template

    rng = np.random.default_rng(0)
    photo = rng.integers(20, 60, size=(200, 300), dtype=np.uint8)
    # Unique textured patch
    patch = rng.integers(0, 255, size=(20, 30), dtype=np.uint8)
    photo[90:110, 150:180] = patch
    templ = np.repeat(np.repeat(patch, 5, axis=0), 5, axis=1)
    rect = locate_scaled_template(
        photo,
        templ,
        target_width_px=30,
        target_height_px=20,
        center_xy=(165, 100),
        search_px=50,
        min_score=0.5,
    )
    assert rect is not None
    x0, y0, x1, y1 = rect
    assert abs(x0 - 150) <= 2
    assert abs(y0 - 90) <= 2
    assert abs((x1 - x0) - 30) <= 1
    assert abs((y1 - y0) - 20) <= 1


def test_camera_from_sample_sites_prefers_crop_calibration():
    from core.mapping.camera import camera_from_sample_sites, locate_image_crop
    from core.mapping.models import MappingFOV, OverviewImage, ElementMap

    photo = np.zeros((200, 300, 3), dtype=np.uint8)
    photo[40:80, 100:160] = (12, 80, 200)
    photo[55, 120] = (255, 255, 0)
    optical = OverviewImage(name="Map area", data=photo[40:80, 100:160].copy())
    em = ElementMap(name="Ca", data=np.ones((10, 15), dtype=np.float64))
    fov = MappingFOV(
        id="s1",
        name="Site 1",
        width=15,
        height=10,
        element_maps=[em],
        optical=optical,
        stage_center_mm=(5.0, -2.0),
        metadata={"map_extra": {"size_mm": (3.0, 2.0)}},
    )
    # Confirm crop works
    assert locate_image_crop(photo, optical.data) == (100.0, 40.0, 160.0, 80.0)
    cam = camera_from_sample_sites(photo, [fov])
    assert cam is not None
    assert abs(cam.origin_x_mm) > 0.1 or abs(cam.origin_y_mm) > 0.1
    px, py = cam.stage_to_pixel(5.0, -2.0)
    np.testing.assert_allclose([px, py], [130.0, 60.0], atol=0.5)
