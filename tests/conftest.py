import importlib.util
import os
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "false")

import cv2
import numpy as np
import pytest
import yaml

from artigate.data.export import write_manifest

ROOT = Path(__file__).resolve().parents[1]


def lesion_image(seed, size=(480, 360)):
    """Synthetic dermoscopy-like image with a dark elliptical lesion, plus its mask."""
    width, height = size
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), (170, 180, 210), np.uint8)
    image += rng.integers(0, 20, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), np.uint8)
    center = (int(width * rng.uniform(0.4, 0.6)), int(height * rng.uniform(0.4, 0.6)))
    cv2.ellipse(mask, center, (int(width * 0.2), int(height * 0.15)), 0, 0, 360, 1, -1)
    image[mask > 0] = (60, 70, 110)
    return image, mask.astype(bool)


@pytest.fixture
def load_script():
    def load(name):
        spec = importlib.util.spec_from_file_location(f"scripts_{name}", ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def raw_dataset(tmp_path):
    """Six raw image/mask pairs (4 train, 1 val, 1 test) and their manifest."""
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "masks").mkdir()
    rows = []
    for index, split in enumerate(["train"] * 4 + ["val", "test"]):
        image, mask = lesion_image(index)
        sample_id = f"ISIC_{index:07d}"
        cv2.imwrite(str(raw / "images" / f"{sample_id}.jpg"), image)
        cv2.imwrite(str(raw / "masks" / f"{sample_id}_segmentation.png"), mask.astype(np.uint8) * 255)
        rows.append({"sample_id": sample_id, "split": split, "image": f"images/{sample_id}.jpg",
                     "mask": f"masks/{sample_id}_segmentation.png", "group": ""})
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    return raw, manifest


@pytest.fixture
def yolo_dataset(raw_dataset, tmp_path, load_script):
    """The raw fixture exported to YOLO-seg, with a dataset YAML using a relative path."""
    raw, manifest = raw_dataset
    load_script("export_yolo").main(["--manifest", str(manifest), "--raw-root", str(raw), "--out", str(tmp_path / "yolo")])
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": "yolo", "train": "images/train", "val": "images/val",
                                    "test": "images/test", "names": {0: "lesion"}}), encoding="utf-8")
    return data
