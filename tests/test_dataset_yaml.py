from pathlib import Path

import pytest
import yaml

from artigate.data.dataset_yaml import load_dataset_yaml, runtime_dataset_yaml, split_dir


def test_runtime_yaml_has_absolute_path_and_is_removed(yolo_dataset):
    original = yolo_dataset.read_bytes()
    with runtime_dataset_yaml(yolo_dataset) as runtime:
        config = yaml.safe_load(runtime.read_text(encoding="utf-8"))
        assert Path(config["path"]).is_absolute()
        assert Path(config["path"]) == (yolo_dataset.parent / "yolo").resolve()
    assert not runtime.exists()
    assert yolo_dataset.read_bytes() == original


def test_missing_split_directories_fail_early(tmp_path):
    source = tmp_path / "data.yaml"
    source.write_text(yaml.safe_dump({"path": "missing", "train": "images/train", "val": "images/val"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"labels[/\\]train"):
        with runtime_dataset_yaml(source):
            pass


def test_split_dir_mirrors_the_images_path(yolo_dataset):
    config, root = load_dataset_yaml(yolo_dataset)
    assert split_dir(root, config, "test", "masks") == root / "masks" / "test"
    assert split_dir(root, config, "test", "labels").is_dir()
