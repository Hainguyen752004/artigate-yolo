"""Train one YOLO26-seg configuration with the shared Paper 1 schedule."""
import argparse
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from ultralytics import YOLO

from artigate.data.dataset_yaml import runtime_dataset_yaml
from artigate.modules import register_modules
from artigate.provenance import environment, git_state, sha256_file, warn_if_unpinned, write_json
from artigate.trainer import transfer_trainer

DEFAULT_MODEL = ROOT / "configs" / "models" / "yolo26n-seg-p2-cbam.yaml"
DEFAULT_DATA = ROOT / "configs" / "data" / "isic2018.yaml"
DEFAULT_HYP = ROOT / "configs" / "train" / "default.yaml"
DEFAULT_PRETRAINED = ROOT / "weights" / "yolo26n-seg.pt"
DEFAULT_PROJECT = ROOT / "runs" / "segment"


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _fraction(value):
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in the range (0, 1]")
    return parsed


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=_project_path, default=DEFAULT_MODEL, help="model YAML")
    parser.add_argument("--data", type=_project_path, default=DEFAULT_DATA, help="dataset YAML")
    parser.add_argument("--hyp", type=_project_path, default=DEFAULT_HYP, help="shared training schedule YAML")
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED),
                        help="stock YOLO26n-seg checkpoint for partial transfer, or 'none' to train from scratch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run name (default: <model>_s<seed>)")
    parser.add_argument("--project", type=_project_path, default=DEFAULT_PROJECT)
    parser.add_argument("--device", default=None, help="CUDA device index or 'cpu' (default: 0 if available)")
    parser.add_argument("--epochs", type=_positive_int, default=None, help="override the schedule (smoke tests only)")
    parser.add_argument("--batch", type=_positive_int, default=None, help="override the schedule batch size")
    parser.add_argument("--workers", type=_nonnegative_int, default=None)
    parser.add_argument("--fraction", type=_fraction, default=None, help="train on a fraction of the data (smoke tests)")
    return parser.parse_args(argv)


def training_arguments(args):
    """Merge the shared schedule with CLI overrides; Windows defaults to in-process data loading."""
    hyp = yaml.safe_load(args.hyp.read_text(encoding="utf-8")) or {}
    overrides = {"epochs": args.epochs, "batch": args.batch, "workers": args.workers, "fraction": args.fraction}
    hyp.update({key: value for key, value in overrides.items() if value is not None})
    if args.workers is None and platform.system() == "Windows":
        hyp["workers"] = 0
    return hyp


def main(argv=None):
    args = parse_args(argv)
    register_modules()
    warn_if_unpinned()
    pretrained = None if args.pretrained.lower() == "none" else _project_path(args.pretrained)
    if pretrained is not None and not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained} (pass --pretrained none to train from scratch)")
    hyp = training_arguments(args)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    name = args.name or f"{args.model.stem}_s{args.seed}"
    with runtime_dataset_yaml(args.data) as runtime_yaml:
        model = YOLO(str(args.model))
        model.train(
            trainer=transfer_trainer(pretrained), data=str(runtime_yaml), seed=args.seed, device=device,
            project=str(args.project), name=name, exist_ok=False, verbose=False, **hyp,
        )
    save_dir = Path(model.trainer.save_dir)
    report = model.trainer.transfer_report
    write_json(save_dir / "provenance.json", {
        "model_yaml": str(args.model),
        "model_yaml_sha256": sha256_file(args.model),
        "data_yaml": str(args.data),
        "hyp": hyp,
        "seed": args.seed,
        "device": str(device),
        "pretrained": None if pretrained is None else {"path": str(pretrained), "sha256": sha256_file(pretrained)},
        "transfer": None if report is None else {
            "counts": {key: len(value) for key, value in report.items()},
            "report": report,
        },
        "git": git_state(ROOT),
        "environment": environment(),
    })
    print(f"[INFO] Run saved to {save_dir}")
    return save_dir


if __name__ == "__main__":
    main()
