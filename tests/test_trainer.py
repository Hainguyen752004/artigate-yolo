"""Regression test for the ICEEIS bug: weights loaded before model.train() were discarded."""
from pathlib import Path

import pytest
import torch
from ultralytics import YOLO

from artigate.data.dataset_yaml import runtime_dataset_yaml
from artigate.modules import register_modules
from artigate.trainer import transfer_trainer

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "weights" / "yolo26n-seg.pt"


class StopTraining(Exception):
    pass


@pytest.mark.skipif(not PRETRAINED.is_file(), reason="weights/yolo26n-seg.pt is not present")
def test_training_starts_from_transferred_stock_weights(yolo_dataset, tmp_path):
    register_modules()
    stock = YOLO(str(PRETRAINED)).model.float().state_dict()
    model = YOLO(str(ROOT / "configs" / "models" / "yolo26n-seg-p2-cbam.yaml"))
    captured = {}

    def on_train_start(trainer):
        captured.update({key: value.detach().float().cpu() for key, value in trainer.model.state_dict().items()})
        raise StopTraining

    model.add_callback("on_train_start", on_train_start)
    with runtime_dataset_yaml(yolo_dataset) as data, pytest.raises(StopTraining):
        model.train(trainer=transfer_trainer(PRETRAINED), data=str(data), epochs=1, batch=2, imgsz=320,
                    device="cpu", workers=0, project=str(tmp_path), name="transfer", verbose=False,
                    plots=False, amp=False)
    for stock_key, custom_key in [
        ("model.0.conv.weight", "model.0.conv.weight"),
        ("model.10.cv1.conv.weight", "model.12.cv1.conv.weight"),
        ("model.16.cv1.conv.weight", "model.18.cv1.conv.weight"),
    ]:
        assert torch.equal(captured[custom_key], stock[stock_key]), custom_key
