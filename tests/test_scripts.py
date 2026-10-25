import csv
import json
import types

import numpy as np
import pytest
import torch

from artigate.data.masks import write_mask


def test_train_passes_shared_schedule_and_transfer_trainer(yolo_dataset, tmp_path, monkeypatch, load_script):
    train = load_script("train")
    calls = {}

    class FakeYOLO:
        def __init__(self, path):
            calls["model"] = path

        def train(self, **kwargs):
            calls["train"] = kwargs
            report = {"mapped": [{}], "missing_destination": [], "shape_mismatched": [], "skipped_unmapped": []}
            self.trainer = types.SimpleNamespace(save_dir=tmp_path / "run", transfer_report=report)

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    pretrained = tmp_path / "stock.pt"
    pretrained.write_bytes(b"weights")
    train.main(["--data", str(yolo_dataset), "--pretrained", str(pretrained), "--epochs", "2",
                "--batch", "4", "--seed", "3", "--device", "cpu", "--workers", "0"])
    kwargs = calls["train"]
    assert (kwargs["epochs"], kwargs["batch"], kwargs["seed"], kwargs["workers"]) == (2, 4, 3, 0)
    assert (kwargs["mask_ratio"], kwargs["degrees"], kwargs["close_mosaic"]) == (2, 180.0, 10)
    assert kwargs["name"] == "yolo26n-seg-p2-cbam_s3"
    assert kwargs["trainer"].pretrained == pretrained
    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["transfer"]["counts"]["mapped"] == 1
    assert provenance["pretrained"]["sha256"]


def test_train_fails_when_pretrained_weights_are_missing(yolo_dataset, tmp_path, load_script):
    with pytest.raises(FileNotFoundError, match="--pretrained none"):
        load_script("train").main(["--data", str(yolo_dataset), "--pretrained", str(tmp_path / "missing.pt")])


class FakeMasks:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)


def test_eval_mask_rules(load_script):
    evaluate = load_script("eval")
    masks = torch.zeros(3, 8, 8)
    masks[0, 0:2] = masks[1, 2:4] = masks[2, 4:6] = 1
    result = types.SimpleNamespace(masks=FakeMasks(masks), boxes=types.SimpleNamespace(conf=torch.tensor([0.2, 0.9, 0.5])))
    top1 = evaluate.predicted_mask(result, "top1", 0.25, (8, 8))
    union = evaluate.predicted_mask(result, "union", 0.4, (8, 8))
    assert top1[2:4].all() and top1.sum() == 16
    assert union[2:6].all() and union.sum() == 32
    empty = evaluate.predicted_mask(types.SimpleNamespace(masks=None), "top1", 0.25, (8, 8))
    assert not empty.any()


@pytest.mark.parametrize("rule, conf, max_det", [("top1", 0.001, 1), ("union", 0.25, 300)])
def test_eval_predicts_in_batch_sized_chunks(rule, conf, max_det, tmp_path, load_script):
    evaluate = load_script("eval")
    calls = []

    class FakeModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return [types.SimpleNamespace(masks=None, orig_shape=(8, 8)) for _ in kwargs["source"]]

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    paths = [tmp_path / f"img{index}.jpg" for index in range(5)]
    for path in paths:
        write_mask(mask_dir / f"{path.stem}.png", np.ones((8, 8), bool))
    args = types.SimpleNamespace(mask_rule=rule, conf=0.25, batch=2)
    rows = evaluate.per_image_rows(FakeModel(), paths, mask_dir, tmp_path, args, "cpu")
    assert [len(call["source"]) for call in calls] == [2, 2, 1]
    assert all((call["conf"], call["max_det"]) == (conf, max_det) for call in calls)
    assert [row["image_id"] for row in rows] == [path.stem for path in paths]
    assert all(row["dice"] == 0.0 for row in rows)


def write_metrics(path, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "dice", "iou", "hd95", "boundary_f1"])
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow({"image_id": f"img{index}", "dice": value, "iou": value, "hd95": 10 * (1 - value),
                             "boundary_f1": value})


def test_bootstrap_script_reports_paired_differences(tmp_path, load_script):
    rng = np.random.default_rng(0)
    baseline = rng.uniform(0.7, 0.9, 100)
    (tmp_path / "A1").mkdir()
    (tmp_path / "A4").mkdir()
    write_metrics(tmp_path / "A1" / "per_image_metrics.csv", baseline)
    write_metrics(tmp_path / "A4" / "per_image_metrics.csv", baseline + 0.01)
    report = load_script("bootstrap_ci").main([
        str(tmp_path / "A4" / "per_image_metrics.csv"), "--baseline", str(tmp_path / "A1" / "per_image_metrics.csv"),
        "--n-boot", "200",
    ])
    assert report["paired"]["dice"]["mean_difference"] == pytest.approx(0.01)
    assert (tmp_path / "A4" / "bootstrap_vs_A1.json").is_file()
    assert (tmp_path / "A4" / "bootstrap_vs_A1.md").is_file()


def test_bootstrap_script_rejects_different_image_sets(tmp_path, load_script):
    write_metrics(tmp_path / "a.csv", [0.8, 0.9])
    write_metrics(tmp_path / "b.csv", [0.8])
    with pytest.raises(ValueError, match="different images"):
        load_script("bootstrap_ci").main([str(tmp_path / "a.csv"), "--baseline", str(tmp_path / "b.csv")])
