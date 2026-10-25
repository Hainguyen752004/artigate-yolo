import csv

import cv2
import numpy as np
import pytest

from artigate.data.export import read_manifest
from artigate.data.masks import read_mask


def ph2_layout(raw, cases, skip_mask=None):
    for case in cases:
        root = raw / "ph2" / "PH2 Dataset images" / case
        (root / f"{case}_Dermoscopic_Image").mkdir(parents=True)
        (root / f"{case}_lesion").mkdir()
        cv2.imwrite(str(root / f"{case}_Dermoscopic_Image" / f"{case}.bmp"), np.zeros((6, 8, 3), np.uint8))
        if case != skip_mask:
            cv2.imwrite(str(root / f"{case}_lesion" / f"{case}_lesion.bmp"), np.zeros((6, 8), np.uint8))


def test_ph2_manifest_puts_every_case_in_test(tmp_path, load_script):
    ph2_layout(tmp_path / "raw", ["IMD002", "IMD003"])
    out = tmp_path / "ph2.csv"
    load_script("prepare_ph2").main(["--raw-root", str(tmp_path / "raw"), "--out", str(out), "--no-count-check"])
    rows = read_manifest(out)
    assert [(row["sample_id"], row["split"]) for row in rows] == [("IMD002", "test"), ("IMD003", "test")]
    assert rows[0]["mask"] == "ph2/PH2 Dataset images/IMD002/IMD002_lesion/IMD002_lesion.bmp"


def test_ph2_incomplete_cases_fail(tmp_path, load_script):
    ph2_layout(tmp_path / "raw", ["IMD002", "IMD003"], skip_mask="IMD003")
    with pytest.raises(ValueError, match="IMD003"):
        load_script("prepare_ph2").main(["--raw-root", str(tmp_path / "raw"), "--out", str(tmp_path / "m.csv"),
                                         "--no-count-check"])


def ddi_layout(raw, labels, conflicting=False):
    images = raw / "ddi" / "DDI_images"
    images.mkdir(parents=True)
    with (images / "ddi_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "DDI_ID", "DDI_file", "skin_tone", "malignant", "disease"])
        for index, stem in enumerate(labels):
            writer.writerow([index, index + 1, f"{stem}.png", "56", "True", "melanoma"])
    for stem in labels:
        cv2.imwrite(str(images / f"{stem}.png"), np.zeros((40, 80, 3), np.uint8))
    for folder in ("training/5%/test", "testing/test_dark"):
        (raw / "ddi" / "fedd" / "ddi_labels" / folder).mkdir(parents=True)
    for stem, label in labels.items():
        np.save(raw / "ddi" / "fedd" / "ddi_labels" / "training/5%/test" / f"{stem}.npy", label)
        copy = label.copy()
        if conflicting:
            copy[0, 0] = 4
        np.save(raw / "ddi" / "fedd" / "ddi_labels" / "testing/test_dark" / f"{stem}.npy", copy)


def test_ddi_fedd_masks_are_resized_back_to_the_image(tmp_path, load_script):
    label = np.zeros((256, 256), np.int32)
    label[64:192, 64:192] = 1  # lesion in the central half of the stretched image
    label[0:16, :] = 3  # ruler along the top
    ddi_layout(tmp_path / "raw", {"000001": label})
    load_script("prepare_ddi_fedd").main(["--raw-root", str(tmp_path / "raw"), "--splits-dir", str(tmp_path),
                                          "--no-count-check"])
    rows = read_manifest(tmp_path / "ddi_fedd.csv")
    assert [(row["sample_id"], row["split"]) for row in rows] == [("DDI_000001", "test")]
    lesion = read_mask(tmp_path / "raw" / rows[0]["mask"])
    assert lesion.shape == (40, 80)
    assert lesion[10:30, 20:60].all() and not lesion[:, :19].any()
    with (tmp_path / "ddi_fedd_artifacts.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{"image_id": "DDI_000001", "marker": "0", "ruler": "1"}]


def test_ddi_fedd_conflicting_copies_fail(tmp_path, load_script):
    ddi_layout(tmp_path / "raw", {"000001": np.ones((256, 256), np.int32)}, conflicting=True)
    with pytest.raises(ValueError, match="differ between split folders"):
        load_script("prepare_ddi_fedd").main(["--raw-root", str(tmp_path / "raw"), "--splits-dir", str(tmp_path),
                                              "--no-count-check"])
