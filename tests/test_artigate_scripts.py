"""Smoke tests for the ArtiGate helper scripts on the exported fixture dataset."""
import csv

import numpy as np
import pytest
import torch
import yaml

from artigate.data.dataset_yaml import load_dataset_yaml, split_dir
from artigate.modules import register_modules
from test_scripts import write_metrics


def test_severity_sets_keep_labels_and_make_evaluable_roots(yolo_dataset, tmp_path, load_script):
    out = tmp_path / "sev"
    load_script("make_severity_sets").main(["--data", str(yolo_dataset), "--split", "test", "--out", str(out),
                                            "--types", "hair", "ruler", "--severities", "3"])
    config, root = load_dataset_yaml(yolo_dataset)
    source = split_dir(root, config, "test", "images")
    for name in ("hair_s3", "ruler_s3"):
        images = sorted((out / name / "images" / "test").iterdir())
        assert [p.name for p in images] == sorted(p.name for p in source.iterdir())
        assert (out / name / "labels" / "test" / f"{images[0].stem}.txt").is_file()
        assert (out / name / "masks" / "test" / f"{images[0].stem}.png").is_file()
        assert yaml.safe_load((out / name / "data.yaml").read_text())["test"] == "images/test"
    import cv2

    original = cv2.imread(str(source / images[0].name)).astype(int)
    hairy = cv2.imread(str(out / "hair_s3" / "images" / "test" / images[0].name)).astype(int)
    assert np.abs(original - hairy).max() > 50


def test_bootstrap_strata_by_artifact_flags(tmp_path, load_script):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.7, 0.9, 40)
    write_metrics(tmp_path / "a1.csv", base)
    write_metrics(tmp_path / "a4.csv", base + 0.02)
    with (tmp_path / "a1.csv").open(newline="") as stream:
        ids = [row["image_id"] for row in csv.DictReader(stream)]
    with (tmp_path / "flags.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "hair", "ruler"])
        for index, image_id in enumerate(ids):
            writer.writerow([image_id.replace("_v1", ""), int(index % 2 == 0), 0])
    report = load_script("bootstrap_ci").main([str(tmp_path / "a4.csv"), "--baseline", str(tmp_path / "a1.csv"),
                                               "--n-boot", "100", "--strata", str(tmp_path / "flags.csv"),
                                               "--out", str(tmp_path / "report")])
    assert report["strata"]["with_hair"]["n_images"] == 20
    assert report["strata"]["with_hair"]["paired"]["dice"]["mean_difference"] == pytest.approx(0.02)
    assert "without_ruler" in report["strata"] and "with_ruler" not in report["strata"]


def test_make_artifact_flags_matches_columns(tmp_path, load_script):
    src = tmp_path / "bissoto.csv"
    src.write_text("image,dark_corner,hair,gel_border,ruler,ink,patches\nISIC_0000001.jpg,1,0,0,1,0,0\n", encoding="utf-8")
    load_script("make_artifact_flags").main(["--src", str(src), "--out", str(tmp_path / "flags.csv")])
    with (tmp_path / "flags.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row == {"image_id": "ISIC_0000001", "hair": "0", "ruler": "1", "marker": "0", "dark_corner": "1"}


def test_eval_artifact_head_runs_on_a_checkpoint(yolo_dataset, tmp_path, load_script, capsys):
    register_modules()
    from artigate.artifact_loss import ArtiGateSegModel
    from test_artigate_trainer import A4

    model = ArtiGateSegModel(str(A4), nc=1, verbose=False)
    weights = tmp_path / "a4.pt"
    torch.save({"model": model, "train_args": {}}, weights)
    load_script("eval_artifact_head").main(["--weights", str(weights), "--data", str(yolo_dataset), "--split", "val",
                                            "--n", "1", "--device", "cpu"])
    assert "dark_corner" in capsys.readouterr().out
