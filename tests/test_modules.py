from pathlib import Path

import pytest
import torch
from ultralytics import YOLO

from artigate.modules import CBAM, P2CompatibleSegment26, register_modules

MODELS = Path(__file__).resolve().parents[1] / "configs" / "models"


def test_cbam_preserves_shape_and_rejects_channel_change():
    layer = CBAM(64)
    assert layer(torch.randn(1, 64, 8, 8)).shape == (1, 64, 8, 8)
    with pytest.raises(ValueError, match="preserve channels"):
        CBAM(64, 128)


def test_register_modules_exposes_custom_classes():
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    register_modules()
    assert modules.CBAM is CBAM and tasks.CBAM is CBAM
    assert modules.Segment26 is P2CompatibleSegment26 and tasks.Segment26 is P2CompatibleSegment26


@pytest.mark.parametrize("name, cbam_layers", [("yolo26n-seg-p2.yaml", 0), ("yolo26n-seg-p2-cbam.yaml", 2)])
def test_p2_models_predict_on_four_strides_with_stride4_prototypes(name, cbam_layers):
    register_modules()
    model = YOLO(str(MODELS / name)).model
    head = model.model[-1]
    assert isinstance(head, P2CompatibleSegment26)
    assert head.nl == 4 and head.end2end is True
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert sum(isinstance(layer, CBAM) for layer in model.model) == cbam_layers
    model.eval()
    with torch.no_grad():
        (detections, prototypes), _ = model(torch.zeros(1, 3, 256, 256))
    assert torch.isfinite(detections).all()
    assert prototypes.shape[-2:] == (64, 64)
