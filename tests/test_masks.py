import cv2
import numpy as np
import ultralytics  # noqa: F401  (patches cv2.imread, which read_mask must tolerate)

from artigate.data.masks import label_lines_to_mask, mask_to_polygons, polygons_to_label_lines, read_mask, write_mask


def ellipse_mask(size=640):
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (320, 300), (180, 120), 25, 0, 360, 1, -1)
    return mask.astype(bool)


def test_polygon_round_trip_preserves_the_lesion():
    mask = ellipse_mask()
    lines = polygons_to_label_lines(mask_to_polygons(mask), 640, 640)
    restored = label_lines_to_mask(lines, 640, 640)
    iou = np.logical_and(mask, restored).sum() / np.logical_or(mask, restored).sum()
    assert len(lines) == 1 and iou > 0.97


def test_small_components_are_not_exported():
    mask = ellipse_mask()
    mask[10:13, 10:13] = True
    assert len(mask_to_polygons(mask)) == 1


def test_read_mask_returns_2d_after_ultralytics_import(tmp_path):
    write_mask(tmp_path / "mask.png", ellipse_mask())
    mask = read_mask(tmp_path / "mask.png")
    assert mask.shape == (640, 640) and mask.dtype == bool


def test_label_lines_are_normalized_yolo_polygons():
    line = polygons_to_label_lines(mask_to_polygons(ellipse_mask()), 640, 640)[0].split()
    coordinates = np.array([float(value) for value in line[1:]])
    assert line[0] == "0" and len(coordinates) % 2 == 0 and len(coordinates) >= 6
    assert coordinates.min() >= 0.0 and coordinates.max() <= 1.0
