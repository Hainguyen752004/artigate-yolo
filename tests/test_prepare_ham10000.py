import cv2
import numpy as np
import pytest

from artigate.data.export import read_manifest

METADATA = (
    "lesion_id,image_id,dx,dx_type,age,sex,localization,dataset\n"
    "HAM_0000118,ISIC_0027419,bkl,histo,80.0,male,scalp,vidir_modern\n"
    "HAM_0000118,ISIC_0025030,bkl,histo,80.0,male,scalp,vidir_modern\n"
    "HAM_0002730,ISIC_0026769,mel,histo,75.0,female,back,vidir_modern\n"
)


def ham_layout(raw, skip_mask=None):
    root = raw / "ham10000"
    for name in ("HAM10000_images_part_1", "HAM10000_images_part_2", "HAM10000_segmentations_lesion_tschandl"):
        (root / name).mkdir(parents=True)
    (root / "HAM10000_metadata.csv").write_text(METADATA, encoding="utf-8")
    for index, image_id in enumerate(("ISIC_0025030", "ISIC_0026769", "ISIC_0027419")):
        part = "HAM10000_images_part_1" if index % 2 == 0 else "HAM10000_images_part_2"
        cv2.imwrite(str(root / part / f"{image_id}.jpg"), np.zeros((6, 8, 3), np.uint8))
        if image_id != skip_mask:
            cv2.imwrite(str(root / "HAM10000_segmentations_lesion_tschandl" / f"{image_id}_segmentation.png"),
                        np.zeros((6, 8), np.uint8))


def test_one_image_per_lesion_with_the_smallest_image_id(tmp_path, load_script):
    ham_layout(tmp_path / "raw")
    out = tmp_path / "ham.csv"
    load_script("prepare_ham10000").main(["--raw-root", str(tmp_path / "raw"), "--out", str(out)])
    rows = read_manifest(out)
    assert [(r["sample_id"], r["group"], r["dx"], r["split"]) for r in rows] == [
        ("ISIC_0025030", "HAM_0000118", "bkl", "test"), ("ISIC_0026769", "HAM_0002730", "mel", "test"),
    ]
    assert rows[1]["image"] == "ham10000/HAM10000_images_part_2/ISIC_0026769.jpg"


def test_missing_mask_of_a_chosen_image_fails(tmp_path, load_script):
    ham_layout(tmp_path / "raw", skip_mask="ISIC_0026769")
    with pytest.raises(ValueError, match="ISIC_0026769"):
        load_script("prepare_ham10000").main(["--raw-root", str(tmp_path / "raw"), "--out", str(tmp_path / "m.csv")])
