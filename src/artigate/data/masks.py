"""Binary lesion masks <-> YOLO-seg polygon labels."""
from pathlib import Path

import cv2
import numpy as np

# Connected components smaller than this (canvas pixels) are not exported as instances;
# ICEEIS used the same floor for polygons after augmentation.
MIN_POLYGON_AREA_PX = 50


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unreadable mask: {path}")
    # Importing ultralytics replaces cv2.imread with a version that returns grayscale as (H, W, 1).
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127


def write_mask(path, mask):
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise OSError(f"failed to write mask: {path}")


def mask_to_polygons(mask, min_area=MIN_POLYGON_AREA_PX):
    """Outer contours of each connected component as (N, 2) float pixel arrays; holes are ignored."""
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        contour.reshape(-1, 2).astype(np.float32)
        for contour in contours
        if len(contour) >= 3 and cv2.contourArea(contour) >= min_area
    ]


def polygons_to_label_lines(polygons, width, height, class_id=0):
    lines = []
    for polygon in polygons:
        normalized = np.clip(polygon / np.array([width, height], dtype=np.float32), 0.0, 1.0)
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized.reshape(-1)))
    return lines


def label_lines_to_mask(lines, width, height):
    """Rasterize the union of YOLO-seg polygons; used when no ground-truth mask file exists."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        points = np.array([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        points = np.rint(points * np.array([width, height], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def read_label_mask(path, width, height):
    return label_lines_to_mask(Path(path).read_text(encoding="utf-8").splitlines(), width, height)
