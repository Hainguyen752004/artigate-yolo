import numpy as np
import pytest

from artigate.data.synth_artifacts import TYPES, composite


def skin(H=240, W=320):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (150, 170, 215)
    img[90:150, 120:200] = (40, 60, 110)  # lesion
    return img, (120, 90, 200, 150)


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("sev", [1, 2, 3, 4])
def test_each_type(t, sev):
    img, box = skin()
    orig = img.copy()
    masks, touched, applied = composite(img, np.random.default_rng(sev), box, severity=sev, force=t)
    k = TYPES.index(t)
    if t == "ruler" and not applied:  # may legitimately not fit
        return
    assert applied == [t] and masks[..., k].any()
    assert (touched[..., k] >= masks[..., k]).all()
    changed = np.any(img != orig, axis=2)
    assert not (changed & ~touched.any(-1)).any()  # every changed pixel is accounted for
    x1, y1, x2, y2 = box
    if t != "hair":  # only hair may cover the lesion
        assert not changed[y1:y2, x1:x2].any(), f"{t} touched the lesion"


def test_deterministic():
    a, box = skin()
    b = a.copy()
    composite(a, np.random.default_rng(7), box)
    composite(b, np.random.default_rng(7), box)
    assert (a == b).all()
