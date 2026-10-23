from pathlib import Path

import pytest
import torch
import yaml
from ultralytics import YOLO

from artigate.modules import register_modules
from artigate.transfer import build_transfer_report, stock_layer_map

MODELS = Path(__file__).resolve().parents[1] / "configs" / "models"
# Hand-written map from the ICEEIS train script (03_train_p2_cbam.py).
ICEEIS_LAYER_MAP = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12,
    11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18,
}


def test_layer_map_matches_iceeis_for_cbam_model():
    assert stock_layer_map(MODELS / "yolo26n-seg-p2-cbam.yaml") == ICEEIS_LAYER_MAP


def test_layer_map_is_identity_for_plain_p2_model():
    assert stock_layer_map(MODELS / "yolo26n-seg-p2.yaml") == {index: index for index in range(17)}


def test_layer_map_rejects_mismatched_layers():
    cfg = yaml.safe_load((MODELS / "yolo26n-seg-p2.yaml").read_text(encoding="utf-8"))
    cfg["backbone"][2][3] = [128, False, 0.25]
    with pytest.raises(ValueError, match="does not match"):
        stock_layer_map(cfg)


def test_transfer_report_classifies_tensors():
    destination = {"model.0.same": torch.zeros(2), "model.0.different": torch.zeros(3), "model.1.missing": torch.zeros(1)}
    source = {"model.0.same": torch.ones(2), "model.0.different": torch.ones(4), "model.99.extra": torch.ones(1)}
    report, matched = build_transfer_report(destination, source, {0: 0, 1: 1})
    assert report["mapped"] == [{"source": "model.0.same", "destination": "model.0.same"}]
    assert report["missing_destination"] == ["model.1.missing"]
    assert report["shape_mismatched"] == [{"source": "model.0.different", "destination": "model.0.different"}]
    assert report["skipped_unmapped"] == ["model.99.extra"]
    assert list(matched) == ["model.0.same"]


def test_transfer_follows_layer_shift_from_inserted_cbam():
    register_modules()
    model_yaml = MODELS / "yolo26n-seg-p2-cbam.yaml"
    stock = YOLO("yolo26n-seg.yaml").model.state_dict()
    custom = YOLO(str(model_yaml)).model
    marker = torch.full_like(stock["model.10.cv1.conv.weight"], 0.125)
    stock["model.10.cv1.conv.weight"] = marker
    report, matched = build_transfer_report(custom.state_dict(), stock, stock_layer_map(model_yaml))
    custom.load_state_dict(matched, strict=False)
    assert torch.equal(custom.state_dict()["model.12.cv1.conv.weight"], marker)
    assert torch.equal(custom.state_dict()["model.10.cv1.conv.weight"], stock["model.8.cv1.conv.weight"])
    assert not report["shape_mismatched"] and not report["missing_destination"]
    assert not any(entry["destination"].startswith("model.19.") for entry in report["mapped"])
