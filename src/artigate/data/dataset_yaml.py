"""Dataset YAML helpers shared by training and evaluation."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

import yaml

SPLITS = ("train", "val", "test")


def load_dataset_yaml(source):
    source = Path(source).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    path = Path(config.get("path") or ".").expanduser()
    root = path.resolve() if path.is_absolute() else (source.parent / path).resolve()
    return config, root


def split_dir(root, config, split, kind):
    """Directory of ``kind`` (images, labels or masks) for a split, mirroring the images path."""
    relative = Path(config.get(split, f"images/{split}"))
    parts = list(relative.parts)
    if kind != "images":
        if "images" not in parts:
            return root / kind / split
        parts[parts.index("images")] = kind
    return root.joinpath(*parts)


@contextmanager
def runtime_dataset_yaml(source):
    """Yield a temporary copy of a dataset YAML whose ``path`` is absolute.

    Ultralytics resolves a relative ``path`` against its global datasets directory rather than
    the YAML file, so the configs in configs/data are rewritten before use. Missing split
    directories fail early.
    """
    config, root = load_dataset_yaml(source)
    missing = [
        str(directory)
        for split in SPLITS if split in config
        for directory in (split_dir(root, config, split, "images"), split_dir(root, config, split, "labels"))
        if not directory.is_dir()
    ]
    if missing:
        raise FileNotFoundError("Dataset directories are missing: " + ", ".join(missing))
    handle, name = tempfile.mkstemp(suffix=".yaml", prefix="artigate_data_")
    runtime_path = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(config, path=root.as_posix()), stream, sort_keys=False, allow_unicode=True)
        yield runtime_path
    finally:
        runtime_path.unlink(missing_ok=True)
