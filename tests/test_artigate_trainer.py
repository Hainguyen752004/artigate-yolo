"""ArtiGate training path: 7-channel data, artifact loss, weight loading."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ultralytics import YOLO

from artigate.data.dataset_yaml import runtime_dataset_yaml
from artigate.modules import register_modules
from artigate.trainer import artigate_trainer

ROOT = Path(__file__).resolve().parents[1]
A4 = ROOT / "configs" / "models" / "yolo26n-seg-p2-artigate.yaml"
PRETRAINED = ROOT / "weights" / "yolo26n-seg.pt"


class StopTraining(Exception):
    pass


@pytest.fixture
def art_dataset(yolo_dataset, tmp_path):
    config = yaml.safe_load(yolo_dataset.read_text(encoding="utf-8"))
    config["artigate"] = {"p_synth": 1.0, "severity": [2, 4]}
    path = yolo_dataset.parent / "data_art.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_dataset_adds_artifact_channels_inside_the_content_area(art_dataset):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    from artigate.data.artifact_dataset import ArtiGateDataset

    with runtime_dataset_yaml(art_dataset) as runtime:
        data = check_det_dataset(str(runtime))
        dataset = ArtiGateDataset(
            img_path=data["train"], imgsz=640, batch_size=2, augment=True, hyp=get_cfg(overrides={"task": "segment"}),
            rect=False, cache=None, single_cls=False, stride=32, pad=0.0, prefix="", task="segment", classes=None,
            data=data, fraction=1.0, art_cfg=data["artigate"],
        )
    assert dataset.content, "metadata/<split>.jsonl was not read"
    np.random.seed(0)
    for i in range(len(dataset)):
        image, _, _ = dataset.load_image(i)
        assert image.shape[2] == 7
        (x1, y1, x2, y2), _ = dataset._boxes(i, *image.shape[:2])
        art = image[..., 3:]
        outside = np.ones(image.shape[:2], bool)
        outside[y1:y2, x1:x2] = False
        assert not (art[outside] == 255).any(), "artifact drawn on the letterbox padding"
        assert (art == 255).any() and (art == 0).any()


def _train_one_batch(model_yaml, data_yaml, tmp_path, pretrained=None, **trainer_kwargs):
    register_modules()
    model = YOLO(str(model_yaml))
    captured = {}

    def on_train_batch_end(trainer):
        captured["names"] = trainer.loss_names if hasattr(trainer, "loss_names") else None
        captured["items"] = trainer.loss_items.detach().cpu().clone()
        captured["state"] = {k: v.detach().float().cpu() for k, v in trainer.model.state_dict().items()}
        raise StopTraining

    model.add_callback("on_train_batch_end", on_train_batch_end)
    with runtime_dataset_yaml(data_yaml) as data, pytest.raises(StopTraining):
        model.train(trainer=artigate_trainer(pretrained, **trainer_kwargs), data=str(data), epochs=1, batch=2,
                    imgsz=320, device="cpu", workers=0, project=str(tmp_path), name="run", verbose=False,
                    plots=False, amp=False, mosaic=1.0)
    return captured


def test_artigate_training_step_has_a_positive_artifact_loss(art_dataset, tmp_path):
    captured = _train_one_batch(A4, art_dataset, tmp_path)
    items = captured["items"]
    assert len(items) == 6  # box, seg, cls, dfl, sem + art
    assert items[-1] > 0


def test_art_lambda_zero_switches_the_artifact_loss_off(art_dataset, tmp_path):
    captured = _train_one_batch(A4, art_dataset, tmp_path, art_lambda=0.0)
    assert captured["items"][-1] == 0


def test_baseline_rows_train_with_the_same_trainer(yolo_dataset, tmp_path):
    captured = _train_one_batch(ROOT / "configs" / "models" / "yolo26n-seg-p2-cbam.yaml", yolo_dataset, tmp_path)
    assert len(captured["items"]) == 6 and captured["items"][-1] == 0


def test_same_architecture_checkpoint_loads_every_tensor(yolo_dataset, tmp_path):
    """Resuming must restore the head too (a backbone-only reload once reset mask mAP to ~0)."""
    register_modules()
    from artigate.artifact_loss import ArtiGateSegModel

    source = ArtiGateSegModel(str(A4), nc=1, verbose=False)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.321)
    trainer_class = artigate_trainer(PRETRAINED if PRETRAINED.is_file() else None)
    with runtime_dataset_yaml(yolo_dataset) as data:
        trainer = trainer_class(overrides={"model": str(A4), "data": str(data), "task": "segment", "device": "cpu",
                                           "project": str(tmp_path), "name": "t", "exist_ok": True})
        model = trainer.get_model(cfg=source.yaml, weights=source)
    expected = source.state_dict()
    loaded = model.state_dict()
    assert all(torch.equal(expected[k].float(), loaded[k].float()) for k in expected if expected[k].is_floating_point())


@pytest.mark.skipif(not PRETRAINED.is_file(), reason="weights/yolo26n-seg.pt is not present")
def test_artigate_starts_from_transferred_stock_weights(art_dataset, tmp_path):
    stock = YOLO(str(PRETRAINED)).model.float().state_dict()
    captured = _train_one_batch(A4, art_dataset, tmp_path, pretrained=PRETRAINED)
    # stock 3 (P3 conv) -> A4 5, stock 16 (top-down P3 C3k2) -> A4 19 (B2/B3 inserts shift the indices);
    # batch 2 with nbs 64 accumulates gradients, so no optimizer step has happened yet
    for stock_key, custom_key in [("model.3.conv.weight", "model.5.conv.weight"),
                                  ("model.16.cv1.conv.weight", "model.19.cv1.conv.weight")]:
        assert torch.allclose(captured["state"][custom_key], stock[stock_key], atol=5e-3), custom_key
