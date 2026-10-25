import csv

import cv2
import numpy as np


def noise(seed, size=(64, 64)):
    return np.random.default_rng(seed).integers(0, 255, (*size, 3), dtype=np.uint8)


def test_shared_ids_ignore_view_suffix_and_file_type(tmp_path, load_script):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    cv2.imwrite(str(a / "ISIC_0000380_v1.jpg"), noise(0))
    cv2.imwrite(str(a / "IMD002_v1.jpg"), noise(1))
    cv2.imwrite(str(b / "ISIC_0000380.png"), noise(2))  # same ISIC image, reprocessed: pixels differ
    cv2.imwrite(str(b / "IMD002.bmp"), noise(3))
    cv2.imwrite(str(b / "ISIC_0000999.png"), noise(4))
    load_script("check_leakage").main(["--a", str(a), "--b", str(b), "--out", str(tmp_path / "out.csv")])
    with (tmp_path / "out.csv").open(newline="", encoding="utf-8") as stream:
        same = sorted((row["a"], row["b"]) for row in csv.DictReader(stream) if row["kind"] == "same_id")
    assert same == [("IMD002_v1", "IMD002"), ("ISIC_0000380_v1", "ISIC_0000380")]


def test_identical_pixels_under_other_names_are_likely_duplicates(tmp_path, load_script):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    image = cv2.GaussianBlur(noise(5, (128, 128)), (9, 9), 0)
    cv2.imwrite(str(a / "ISIC_0000001_v1.png"), image)
    cv2.imwrite(str(b / "DDI_000123_v1.png"), image)
    load_script("check_leakage").main(["--a", str(a), "--b", str(b), "--out", str(tmp_path / "out.csv")])
    with (tmp_path / "out.csv").open(newline="", encoding="utf-8") as stream:
        kinds = [row["kind"] for row in csv.DictReader(stream)]
    assert kinds == ["likely_duplicate"]
