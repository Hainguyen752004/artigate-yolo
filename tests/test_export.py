import json

import cv2
import pytest

from artigate.data.export import check_manifest, read_manifest
from artigate.data.masks import read_label_mask, read_mask


def test_export_writes_aligned_images_labels_and_masks(raw_dataset, tmp_path, load_script):
    raw, manifest = raw_dataset
    out = tmp_path / "yolo"
    summary = load_script("export_yolo").main([
        "--manifest", str(manifest), "--raw-root", str(raw), "--out", str(out), "--train-views", "v1,v7",
    ])
    assert summary["splits"]["train"]["images"] == 8
    assert summary["splits"]["val"]["images"] == summary["splits"]["test"]["images"] == 1
    for label in (out / "labels" / "train").glob("*.txt"):
        image = cv2.imread(str(out / "images" / "train" / f"{label.stem}.jpg"))
        assert image.shape == (640, 640, 3)
        truth = read_mask(out / "masks" / "train" / f"{label.stem}.png")
        restored = read_label_mask(label, 640, 640)
        assert (truth & restored).sum() / (truth | restored).sum() > 0.95
    records = [json.loads(line) for line in (out / "metadata" / "train.jsonl").read_text().splitlines()]
    assert sorted(record["view"] for record in records) == ["v1"] * 4 + ["v7"] * 4
    assert (out / "metadata" / "export_summary.json").is_file()


def test_export_refuses_to_overwrite_silently(raw_dataset, tmp_path, load_script):
    raw, manifest = raw_dataset
    export = load_script("export_yolo")
    arguments = ["--manifest", str(manifest), "--raw-root", str(raw), "--out", str(tmp_path / "yolo")]
    export.main(arguments)
    with pytest.raises(FileExistsError, match="--overwrite"):
        export.main(arguments)
    export.main(arguments + ["--overwrite"])


def test_check_manifest_rejects_group_leaks(raw_dataset):
    _, manifest = raw_dataset
    rows = read_manifest(manifest)
    rows[0]["group"] = rows[-1]["group"] = "HAM_0000001"
    with pytest.raises(ValueError, match="span several splits"):
        check_manifest(rows)


def test_check_manifest_rejects_shared_isic_ids(raw_dataset):
    _, manifest = raw_dataset
    rows = read_manifest(manifest)
    rows[-1]["sample_id"] = rows[0]["sample_id"] + "_dup"
    with pytest.raises(ValueError, match="ISIC IDs shared"):
        check_manifest(rows)
