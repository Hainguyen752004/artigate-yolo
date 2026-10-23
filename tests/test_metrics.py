import numpy as np
import pytest

from artigate.metrics import binary_metrics, boundary_f1, dice_iou, hd95


def square(top=16, left=16, side=20, size=64):
    mask = np.zeros((size, size), dtype=bool)
    mask[top:top + side, left:left + side] = True
    return mask


def test_identical_masks_are_perfect():
    assert binary_metrics(square(), square()) == {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "boundary_f1": 1.0}


def test_partial_overlap_dice_and_iou():
    dice, iou = dice_iou(square(), square(top=26))
    assert dice == pytest.approx(0.5)
    assert iou == pytest.approx(1 / 3)


def test_disjoint_masks_score_zero_overlap():
    dice, iou = dice_iou(square(left=0, side=10), square(left=40, side=10))
    assert (dice, iou) == (0.0, 0.0)


def test_hd95_of_shifted_square_equals_shift():
    assert hd95(square(), square(left=19)) == pytest.approx(3.0)


def test_boundary_f1_respects_tolerance():
    shifted = square(left=19)
    assert boundary_f1(square(), shifted, tolerance=3) == pytest.approx(1.0)
    assert boundary_f1(square(), shifted, tolerance=1) < 1.0


def test_empty_mask_conventions():
    empty = np.zeros((64, 64), dtype=bool)
    assert binary_metrics(empty, empty) == {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "boundary_f1": 1.0}
    one_empty = binary_metrics(empty, square())
    assert one_empty["dice"] == 0.0 and one_empty["boundary_f1"] == 0.0
    assert one_empty["hd95"] == pytest.approx(np.hypot(64, 64))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="differs"):
        binary_metrics(np.zeros((4, 4)), np.zeros((5, 5)))
