import cv2
import numpy as np

from artigate.data.export import read_manifest, write_manifest
from artigate.data.views import letterbox

RELEASE = "A skin lesion hair mask dataset with fine-grained annotations"


def test_write_manifest_keeps_extra_columns(tmp_path):
    rows = [{"sample_id": "ISIC_0000004", "split": "test", "image": "a.png", "mask": "b.png", "group": "ISIC_0000004",
             "subset": "seen", "isic_split": "train"}]
    write_manifest(tmp_path / "m.csv", rows)
    assert read_manifest(tmp_path / "m.csv") == rows


def test_hair_manifest_labels_seen_and_unseen_images(tmp_path, load_script):
    root = tmp_path / "raw" / "hair_hossain" / RELEASE
    (root / "dermoscopic_image").mkdir(parents=True)
    (root / "hair_mask").mkdir()
    for sample_id in ("ISIC_0000004", "ISIC_0036099", "ISIC_0099999"):
        cv2.imwrite(str(root / "dermoscopic_image" / f"{sample_id}.png"), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(root / "hair_mask" / f"{sample_id}.png"), np.zeros((8, 8, 3), np.uint8))
    isic = tmp_path / "isic2018.csv"
    write_manifest(isic, [
        {"sample_id": "ISIC_0000004", "split": "train", "image": "x", "mask": "y", "group": "ISIC_0000004"},
        {"sample_id": "ISIC_0036099", "split": "test", "image": "x", "mask": "y", "group": "ISIC_0036099"},
    ])
    out = tmp_path / "hair.csv"
    load_script("prepare_hair_hossain").main(["--raw-root", str(tmp_path / "raw"), "--isic-manifest", str(isic),
                                              "--out", str(out), "--no-count-check"])
    rows = read_manifest(out)
    assert [(r["sample_id"], r["subset"], r["isic_split"]) for r in rows] == [
        ("ISIC_0000004", "seen", "train"), ("ISIC_0036099", "unseen", "test"), ("ISIC_0099999", "unseen", "none"),
    ]
    assert rows[0]["image"].endswith("dermoscopic_image/ISIC_0000004.png")


def test_audit_maps_the_mask_back_and_counts_faint_hair_as_absent(load_script):
    audit = load_script("audit_hair_negatives")
    source = np.full((300, 400, 3), 200, np.uint8)  # original ISIC image, 400 x 300
    cv2.line(source, (50, 40), (350, 40), (20, 20, 20), 3)  # dark hair: not an auto negative
    cv2.line(source, (50, 250), (350, 250), (196, 196, 196), 3)  # faint hair: black-hat below threshold
    hair = np.zeros((300, 400), np.uint8)
    cv2.line(hair, (50, 40), (350, 40), 1, 3)
    cv2.line(hair, (50, 250), (350, 250), 1, 3)
    train_image = letterbox(source, 640)[0]  # the exported training image
    hossain_image = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)  # stretched release
    hossain_mask = cv2.resize(hair, (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
    counts = audit.audit_image(hossain_image, hossain_mask, train_image, (400, 300))
    assert counts["corr"] > 0.97
    assert 0.4 < counts["absent_auto"] / counts["hair_pixels"] < 0.6  # the faint line, not the dark one
