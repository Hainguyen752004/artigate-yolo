"""Per-image binary segmentation metrics.

Conventions: two empty masks score Dice/IoU/boundary F1 = 1 and HD95 = 0. When exactly one
mask is empty, HD95 is the canvas diagonal (the largest possible distance) and boundary F1 is 0.
"""
import numpy as np
from scipy import ndimage

METRIC_NAMES = ("dice", "iou", "hd95", "boundary_f1")
# Boundary F1 tolerance as a fraction of the image diagonal (Csurka et al., BMVC 2013).
BOUNDARY_TOLERANCE_FRACTION = 0.0075
_CROSS = ndimage.generate_binary_structure(2, 1)


def dice_iou(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0, 1.0
    return float(2 * intersection / total), float(intersection / (total - intersection))


def _boundary(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=_CROSS, border_value=0)


def _distances_to(boundary, mask):
    """Distance from every boundary pixel of ``boundary`` to the nearest boundary pixel of ``mask``."""
    return ndimage.distance_transform_edt(~_boundary(mask))[boundary]


def hd95(pred, gt):
    """95th percentile of the pooled symmetric boundary distances (MedPy convention), in pixels."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(np.hypot(*gt.shape))
    distances = np.concatenate([_distances_to(_boundary(pred), gt), _distances_to(_boundary(gt), pred)])
    return float(np.percentile(distances, 95))


def boundary_f1(pred, gt, tolerance=None):
    """F1 of boundary pixels matched within ``tolerance`` pixels (default: 0.75% of the diagonal)."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if tolerance is None:
        tolerance = BOUNDARY_TOLERANCE_FRACTION * np.hypot(*gt.shape)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    pred_boundary, gt_boundary = _boundary(pred), _boundary(gt)
    precision = float((_distances_to(pred_boundary, gt) <= tolerance).mean())
    recall = float((_distances_to(gt_boundary, pred) <= tolerance).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def binary_metrics(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} differs from ground truth {gt.shape}")
    dice, iou = dice_iou(pred, gt)
    return {"dice": dice, "iou": iou, "hd95": hd95(pred, gt), "boundary_f1": boundary_f1(pred, gt)}
