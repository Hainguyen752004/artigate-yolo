"""Non-YOLO baselines: networks, canvas data, training outputs, evaluation format and the nnU-Net dataset."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from artigate.baselines.models import THIRD_PARTY, build_model, split_output
from artigate.baselines.recipe import build_transform, load_recipe
from artigate.data.masks import read_mask
from artigate.metrics import METRIC_NAMES

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("einops")
pytest.importorskip("timm")


def test_unet_keeps_the_input_size_and_has_the_original_widths():
    model = build_model("unet", load_recipe("unet")[0]["model"]["kwargs"]).eval()
    with torch.no_grad():
        aux, prob = split_output(model(torch.rand(2, 3, 64, 96)))
    assert aux is None and prob.shape == (2, 1, 64, 96) and 0 <= prob.min() and prob.max() <= 1
    assert [e[0].out_channels for e in model.encoders] == [64, 128, 256, 512, 1024]
    assert 31.0e6 < sum(p.numel() for p in model.parameters()) < 31.1e6  # 31.04 M, as the padded/BN U-Net


def test_egeunet_is_the_upstream_network_with_deep_supervision():
    recipe = load_recipe("egeunet")[0]
    assert recipe["model"]["source"]["commit"].startswith("f52ba30")
    model = build_model("egeunet", recipe["model"]["kwargs"]).eval()
    with torch.no_grad():
        aux, prob = split_output(model(torch.rand(1, 3, 64, 64)))
    assert len(aux) == 5 and all(a.shape == (1, 1, 64, 64) for a in aux) and prob.shape == (1, 1, 64, 64)
    assert 50_000 < sum(p.numel() for p in model.parameters()) < 56_000  # ~53 K parameters


@pytest.mark.skipif(not (THIRD_PARTY / "malunet" / "malunet.py").is_file(), reason="run scripts/fetch_malunet.py")
def test_malunet_runs_on_canvas_sized_inputs():
    model = build_model("malunet", load_recipe("malunet")[0]["model"]["kwargs"]).eval()
    with torch.no_grad():
        aux, prob = split_output(model(torch.rand(1, 3, 64, 64)))
    assert aux is None and prob.shape == (1, 1, 64, 64)


def test_recipe_transform_matches_the_official_pipeline():
    recipe = load_recipe("egeunet")[0]
    image = np.random.default_rng(0).integers(0, 256, (640, 640, 3)).astype(np.uint8)
    mask = np.zeros((640, 640, 1))
    mask[100:300, 200:400] = 1
    test_transform, angle = build_transform(recipe, train=False)
    tensor, target = test_transform((image, mask))
    assert angle is None and tensor.shape == (3, 640, 640) and target.shape == (1, 640, 640)
    # myNormalize ends with a per-image min-max to [0, 255]; myToTensor does not divide by 255
    assert float(tensor.min()) == pytest.approx(0.0) and float(tensor.max()) == pytest.approx(255.0)
    assert torch.equal(target, torch.tensor(mask).permute(2, 0, 1))
    _, angle = build_transform(recipe, train=True)
    assert 0 <= angle <= 360  # drawn once per run, as myRandomRotation does


def test_canvas_dataset_reads_exported_images_and_masks(yolo_dataset):
    from artigate.baselines.data import CanvasSegDataset

    transform, _ = build_transform(load_recipe("egeunet")[0], train=False)
    dataset = CanvasSegDataset(yolo_dataset, "train", transform)
    image, mask = dataset[0]
    assert len(dataset) == 4 and image.shape == (3, 640, 640) and image.dtype == torch.float32
    assert mask.shape == (1, 640, 640) and set(torch.unique(mask).tolist()) <= {0.0, 1.0} and mask.sum() > 0
    assert len(CanvasSegDataset(yolo_dataset, "train", transform, fraction=0.5, seed=1)) == 2


def test_training_writes_the_run_files_and_eval_writes_the_eval_format(yolo_dataset, tmp_path, load_script):
    run = load_script("train_seg_baseline").main([
        "--model", "egeunet", "--seed", "0", "--data", str(yolo_dataset), "--epochs", "2", "--batch", "2",
        "--workers", "0", "--device", "cpu", "--project", str(tmp_path / "runs"),
    ])
    assert run.name == "egeunet_s0" and all((run / f).is_file() for f in ("best.pt", "last.pt", "history.csv", "provenance.json"))
    history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
    assert [int(r["epoch"]) for r in history] == [1, 2] and "val_dice" in history[0]
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["batch_requested"] == provenance["batch_used"] == 2 and provenance["overrides"]["epochs"] == 2
    assert provenance["upstream"]["commit"].startswith("f52ba30") and provenance["upstream"]["sha256"]
    with pytest.raises(SystemExit, match="finished run"):
        load_script("train_seg_baseline").main(["--model", "egeunet", "--data", str(yolo_dataset), "--project", str(tmp_path / "runs")])

    evaluate = load_script("eval_seg_baseline")
    out = evaluate.main(["--weights", str(run / "best.pt"), "--data", str(yolo_dataset), "--split", "val",
                         "--device", "cpu", "--project", str(tmp_path / "eval")])
    assert out.name == f"egeunet_s0_{yolo_dataset.parent.name}_val"  # data.yaml is named by its folder
    rows = list(csv.DictReader((out / "per_image_metrics.csv").open(encoding="utf-8")))
    assert list(rows[0]) == ["image_id", *METRIC_NAMES] and len(rows) == 1
    # the test split stays locked until the test-set artifact labels are committed
    if not (ROOT / "data" / "splits" / "isic2018_test_artifacts.csv").is_file():
        with pytest.raises(SystemExit, match="locked"):
            evaluate.main(["--weights", str(run / "best.pt"), "--data", str(yolo_dataset), "--split", "test",
                           "--device", "cpu", "--project", str(tmp_path / "eval")])


def test_interrupted_run_continues_with_resume(yolo_dataset, tmp_path, load_script):
    train = load_script("train_seg_baseline")
    common = ["--model", "egeunet", "--seed", "1", "--data", str(yolo_dataset), "--batch", "2", "--workers", "0",
              "--device", "cpu", "--project", str(tmp_path)]
    run = train.main(common + ["--epochs", "1"])
    angle = json.loads((run / "provenance.json").read_text(encoding="utf-8"))["rotation_angle"]
    (run / "provenance.json").unlink()  # as if the process had died after epoch 1
    with pytest.raises(SystemExit, match="interrupted run"):
        train.main(common + ["--epochs", "2"])
    train.main(common + ["--epochs", "2", "--resume"])
    history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert [int(r["epoch"]) for r in history] == [1, 2] and provenance["rotation_angle"] == angle


def test_nnunet_predictions_are_scored_from_label_pngs(yolo_dataset, tmp_path, load_script):
    from artigate.baselines.data import canvas_items

    preds = tmp_path / "pred"
    preds.mkdir()
    for image_path, mask_path in canvas_items(yolo_dataset, "val"):
        Image.fromarray(read_mask(mask_path).astype(np.uint8)).save(preds / f"{image_path.stem}.png")  # labels 0/1
    checkpoint = tmp_path / "checkpoint_final.pth"
    checkpoint.write_bytes(b"nnunet")
    out = load_script("eval_seg_baseline").main([
        "--pred-dir", str(preds), "--checkpoint", str(checkpoint), "--run", "nnunet_s0", "--data", str(yolo_dataset),
        "--split", "val", "--project", str(tmp_path / "eval"),
    ])
    row = next(csv.DictReader((out / "per_image_metrics.csv").open(encoding="utf-8")))
    assert out.name.startswith("nnunet_s0_") and float(row["dice"]) == 1.0 and float(row["hd95"]) == 0.0


def test_run_baselines_keeps_the_order_skips_finished_and_resumes_interrupted(tmp_path, load_script, capsys):
    run_baselines = load_script("run_baselines")
    assert [n for n, _, _ in run_baselines.plan(["unet", "egeunet"], [0, 1])] == ["egeunet_s0", "egeunet_s1", "unet_s0", "unet_s1"]
    (tmp_path / "egeunet_s0").mkdir()
    (tmp_path / "egeunet_s0" / "provenance.json").write_text("{}", encoding="utf-8")
    (tmp_path / "egeunet_s1").mkdir()
    (tmp_path / "egeunet_s1" / "last.pt").write_bytes(b"interrupted")
    run_baselines.main(["--models", "egeunet", "--seeds", "0", "1", "2", "--project", str(tmp_path), "--dry-run"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("egeunet_s0: finished, skipped")
    assert lines[1].endswith("--resume") and "--resume" not in lines[2]
    with pytest.raises(SystemExit, match="unknown baselines"):
        run_baselines.plan(["nnunet"], [0])


def test_prepare_nnunet_writes_rgb_images_binary_labels_and_the_official_split(yolo_dataset, tmp_path, load_script):
    raw, pre = tmp_path / "raw", tmp_path / "pre"
    root, splits = load_script("prepare_nnunet").main(["dataset", "--data", str(yolo_dataset), "--raw", str(raw),
                                                        "--preprocessed", str(pre)])
    info = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert info["channel_names"] == {"0": "R", "1": "G", "2": "B"} and info["labels"] == {"background": 0, "lesion": 1}
    assert info["numTraining"] == 5 and info["file_ending"] == ".png"
    assert splits == pre / "Dataset501_ISIC2018Canvas" / "splits_final.json"
    fold = json.loads(splits.read_text(encoding="utf-8"))
    assert len(fold) == 1 and len(fold[0]["train"]) == 4 and len(fold[0]["val"]) == 1
    assert sorted(p.name for p in (root / "imagesTr").iterdir()) == sorted(f"{c}_0000.png" for c in fold[0]["train"] + fold[0]["val"])
    assert len(list((root / "imagesTs").iterdir())) == 1 and len(list((root / "imagesVal").iterdir())) == 1
    image = np.array(Image.open(next((root / "imagesTr").iterdir())))
    label = np.array(Image.open(next((root / "labelsTr").iterdir())))
    assert image.shape == (640, 640, 3) and set(np.unique(label)) <= {0, 1} and label.max() == 1
