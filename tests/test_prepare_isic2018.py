import cv2
import numpy as np
import pytest

from artigate.data.export import read_manifest

LAYOUT = {"Training": ["ISIC_0000000", "ISIC_0000001"], "Validation": ["ISIC_0012255"], "Test": ["ISIC_0036309"]}


def official_layout(raw, skip_mask=None):
    for official, ids in LAYOUT.items():
        image_dir = raw / "isic2018" / f"ISIC2018_Task1-2_{official}_Input"
        mask_dir = raw / "isic2018" / f"ISIC2018_Task1_{official}_GroundTruth"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        (image_dir / "LICENSE.txt").write_text("cc-0", encoding="utf-8")
        for sample_id in ids:
            cv2.imwrite(str(image_dir / f"{sample_id}.jpg"), np.zeros((8, 8, 3), np.uint8))
            if sample_id != skip_mask:
                cv2.imwrite(str(mask_dir / f"{sample_id}_segmentation.png"), np.zeros((8, 8), np.uint8))


def test_manifest_keeps_the_official_split(tmp_path, load_script):
    official_layout(tmp_path / "raw")
    out = tmp_path / "isic2018.csv"
    load_script("prepare_isic2018").main(["--raw-root", str(tmp_path / "raw"), "--out", str(out), "--no-count-check"])
    rows = read_manifest(out)
    assert [(row["sample_id"], row["split"]) for row in rows] == [
        ("ISIC_0000000", "train"), ("ISIC_0000001", "train"), ("ISIC_0012255", "val"), ("ISIC_0036309", "test"),
    ]
    assert rows[0]["image"] == "isic2018/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg"
    assert rows[0]["mask"] == "isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC_0000000_segmentation.png"
    assert all(row["group"] == row["sample_id"] for row in rows)


def test_unpaired_images_fail(tmp_path, load_script):
    official_layout(tmp_path / "raw", skip_mask="ISIC_0000001")
    with pytest.raises(ValueError, match="no mask for ISIC_0000001"):
        load_script("prepare_isic2018").main(["--raw-root", str(tmp_path / "raw"), "--out", str(tmp_path / "m.csv"),
                                              "--no-count-check"])


def test_split_sizes_must_match_the_release(tmp_path, load_script):
    official_layout(tmp_path / "raw")
    with pytest.raises(ValueError, match="official release"):
        load_script("prepare_isic2018").main(["--raw-root", str(tmp_path / "raw"), "--out", str(tmp_path / "m.csv")])
