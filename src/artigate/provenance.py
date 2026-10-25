"""Record what produced a run: code version, environment and input file hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

PINNED_ULTRALYTICS = "8.4.13"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CODE_DIRS = ("src/", "scripts/", "configs/", "tests/")


def git_state(root):
    """Return the commit and dirty flag of the repository at ``root``, or None outside git.

    ``dirty`` means a tracked file differs from the commit, or an untracked file sits where code is
    read from (CODE_DIRS). Other untracked files, such as personal notes, are listed but do not count.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line for line in status.splitlines() if line.strip()]
    untracked = [line[3:].strip('"') for line in lines if line.startswith("??")]
    tracked_changes = [line for line in lines if not line.startswith("??")]
    dirty = bool(tracked_changes) or any(path.startswith(CODE_DIRS) for path in untracked)
    return {"commit": commit, "dirty": dirty, "untracked": untracked[:20]}


def environment():
    import torch
    import ultralytics

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        # Training on CPU makes Ultralytics set CUDA_VISIBLE_DEVICES="" after CUDA may already be
        # initialised; is_available() can then stay True while device_count() drops to 0.
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
        "ultralytics": ultralytics.__version__,
    }


def warn_if_unpinned():
    import warnings

    import ultralytics

    if ultralytics.__version__ != PINNED_ULTRALYTICS:
        warnings.warn(
            f"Ultralytics {ultralytics.__version__} is installed; results are pinned to {PINNED_ULTRALYTICS}.",
            RuntimeWarning,
        )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
