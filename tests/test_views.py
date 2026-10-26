import cv2
import numpy as np
import pytest

from artigate.data.views import dullrazor, letterbox, make_view


def iou(first, second):
    return np.logical_and(first, second).sum() / np.logical_or(first, second).sum()


def test_letterbox_pads_color_images_and_masks_identically():
    image = np.full((300, 600, 3), 200, np.uint8)
    mask = np.ones((300, 600), np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(image, 640)
    mask_canvas, *_ = letterbox(mask, 640, color=0, interpolation=cv2.INTER_NEAREST)
    assert (pad_x, pad_y) == (0, 160) and scale == pytest.approx(640 / 600)
    assert (canvas[:160] == 114).all() and (mask_canvas[:160] == 0).all()
    assert (canvas[160:480] == 200).all() and (mask_canvas[160:480] == 1).all()


def test_v1_view_keeps_lesion_and_mask_aligned():
    image = np.full((360, 480, 3), 210, np.uint8)
    mask = np.zeros((360, 480), bool)
    mask[100:200, 150:300] = True
    image[mask] = (40, 40, 40)
    view_image, view_mask, transform = make_view(image, mask, "v1")
    dark = cv2.cvtColor(view_image, cv2.COLOR_BGR2GRAY) < 100
    assert iou(dark, view_mask) > 0.98
    assert transform["crop_box"] == [0, 0, 480, 360]


def test_v7_view_crops_dark_frame_and_mask_together():
    image = np.zeros((400, 500, 3), np.uint8)
    image[50:350, 60:440] = (180, 190, 220)
    mask = np.zeros((400, 500), bool)
    mask[150:250, 200:320] = True
    image[mask] = (50, 60, 90)
    view_image, view_mask, transform = make_view(image, mask, "v7")
    assert transform["crop_box"] == [60, 50, 380, 300]
    dark = cv2.cvtColor(view_image, cv2.COLOR_BGR2GRAY) < 100
    assert iou(dark, view_mask) > 0.95


def test_v7_removes_hair_on_large_images_at_canvas_resolution():
    # a 2000 x 1500 photo with 20-px hair: wider than the 17x17 kernel at full size, ~6 px on the canvas
    image = np.full((1500, 2000, 3), 200, np.uint8)
    cv2.line(image, (0, 750), (1999, 750), (30, 30, 30), 20)
    mask = np.zeros((1500, 2000), bool)
    mask[200:400, 200:400] = True
    view_image, view_mask, transform = make_view(image, mask, "v7")
    assert transform["scale"] == pytest.approx(640 / 2000)
    assert view_image[320, 100:540].mean() > 150  # the hair line, mapped onto the canvas, is gone
    assert view_mask.sum() == pytest.approx((200 * 0.32) ** 2, rel=0.05)


def test_dullrazor_removes_thin_dark_lines():
    image = np.full((100, 100, 3), 200, np.uint8)
    image[:, 49:51] = 30
    cleaned = dullrazor(image)
    assert cleaned[:, 49:51].mean() > 150


def test_make_view_validates_inputs():
    image = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="unknown view"):
        make_view(image, np.zeros((10, 10), bool), "v3")
    with pytest.raises(ValueError, match="differs"):
        make_view(image, np.zeros((5, 10), bool), "v1")
