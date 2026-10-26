"""Evaluate a checkpoint on one split: Ultralytics mAP plus per-image binary mask metrics.

Per-image metrics compare one binary prediction per image with the exported ground-truth mask
on the 640x640 canvas. --mask-rule top1 keeps the highest-confidence instance (the ICEEIS rule);
union merges every instance at or above --conf.

A test split (ISIC 2018 or an external set) is scored once per checkpoint, dataset and mask rule:
every test scoring is appended to <project>/test_scoring_log.jsonl, and a second scoring of the same
checkpoint (by its sha256) is refused, whatever --name says. --rescore-reason "..." allows it: the
new results go to <name>_rescore<k>, nothing is overwritten, and the reason is appended to
rescore_log.txt in the original folder. Validation and training splits can be rerun freely.

No test split is scored before the ISIC 2018 test-set artifact labels
(data/splits/isic2018_test_artifacts.csv, docs/annotation_test_artifacts.md) are committed: the
paper states that they were made before any model saw the test set. The commit of that file is
recorded in eval_summary.json and in the scoring log.
"""
import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from ultralytics import YOLO

from artigate.data.dataset_yaml import load_dataset_yaml, runtime_dataset_yaml, split_dir
from artigate.data.masks import read_label_mask, read_mask
from artigate.metrics import METRIC_NAMES, binary_metrics
from artigate.modules import register_modules
from artigate.provenance import environment, git_state, sha256_file, warn_if_unpinned, write_json
from artigate.scoring import (  # noqa: F401  (re-exported: tests and older notes use eval.<name>)
    committed_test_labels, default_name, output_dir, record_test_scoring, test_label_record, test_scorings,
)

DEFAULT_DATA = ROOT / "configs" / "data" / "isic2018.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "eval"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
ULTRALYTICS_KEYS = (
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)",
)
TOP1_CONF = 0.001


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=_project_path, required=True)
    parser.add_argument("--data", type=_project_path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--name", default=None, help="output name (default: <run>_<data>_<split>[_union])")
    parser.add_argument("--project", type=_project_path, default=DEFAULT_PROJECT)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mask-rule", choices=("top1", "union"), default="top1")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for --mask-rule union")
    parser.add_argument("--rescore-reason", default=None,
                        help="why a test split already scored with this checkpoint must be scored again (logged)")
    return parser.parse_args(argv)


def predicted_mask(result, rule, conf, shape):
    """Collapse instance predictions into one binary mask of ``shape``."""
    if result.masks is None or len(result.masks) == 0:
        return np.zeros(shape, dtype=bool)
    masks = result.masks.data.cpu().numpy() > 0.5
    scores = result.boxes.conf.cpu().numpy()
    if rule == "top1":
        mask = masks[int(np.argmax(scores))]
    else:
        keep = scores >= conf
        mask = masks[keep].any(axis=0) if keep.any() else np.zeros(masks.shape[1:], dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError(f"predicted mask shape {mask.shape} differs from ground truth {shape}")
    return mask


def ground_truth_mask(stem, mask_dir, label_dir, shape):
    mask_path = mask_dir / f"{stem}.png"
    if mask_path.is_file():
        return read_mask(mask_path)
    return read_label_mask(label_dir / f"{stem}.txt", shape[1], shape[0])


def per_image_rows(model, image_paths, mask_dir, label_dir, args, device):
    rows = []
    # Predictions come sorted by confidence, so top1 needs only the first one.
    conf, max_det = (TOP1_CONF, 1) if args.mask_rule == "top1" else (args.conf, 300)
    # Ultralytics predicts a list source as ONE batch whatever `batch` says, so feed it in chunks.
    for start in range(0, len(image_paths), args.batch):
        chunk = image_paths[start:start + args.batch]
        results = model.predict(
            source=[str(path) for path in chunk], imgsz=640, conf=conf, iou=0.7, max_det=max_det,
            device=device, retina_masks=True, verbose=False,
        )
        for path, result in zip(chunk, results):
            shape = result.orig_shape
            truth = ground_truth_mask(path.stem, mask_dir, label_dir, shape)
            prediction = predicted_mask(result, args.mask_rule, args.conf, shape)
            rows.append({"image_id": path.stem, **binary_metrics(prediction, truth)})
    return rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    register_modules()
    warn_if_unpinned()
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    name = args.name or default_name(args.weights, args.data, args.split, args.mask_rule)
    weights_sha = sha256_file(args.weights)
    save_dir, reuse = output_dir(args.project, name, args.split, weights_sha, args.data, args.mask_rule,
                                 args.rescore_reason)
    config, root = load_dataset_yaml(args.data)
    image_dir = split_dir(root, config, args.split, "images")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    labels = test_label_record(args.data, [p.stem for p in image_paths]) if args.split == "test" else None

    model = YOLO(str(args.weights))
    with runtime_dataset_yaml(args.data) as runtime_yaml:
        metrics = model.val(
            data=str(runtime_yaml), split=args.split, imgsz=640, batch=args.batch, device=device, iou=0.7,
            plots=True, project=str(args.project), name=save_dir.name, exist_ok=reuse, verbose=False,
        )
    if Path(metrics.save_dir).resolve() != save_dir.resolve():
        raise RuntimeError(f"Ultralytics wrote to {metrics.save_dir}, expected {save_dir}")
    rows = per_image_rows(
        model, image_paths, split_dir(root, config, args.split, "masks"),
        split_dir(root, config, args.split, "labels"), args, device,
    )
    write_rows(save_dir / "per_image_metrics.csv", rows)
    results = metrics.results_dict
    summary = {
        "weights": str(args.weights),
        "weights_sha256": weights_sha,
        "data_yaml": str(args.data),
        "split": args.split,
        "rescore_reason": args.rescore_reason,
        "test_artifact_labels": labels,
        "n_images": len(rows),
        "mask_rule": args.mask_rule,
        "union_conf": args.conf if args.mask_rule == "union" else None,
        "ultralytics_metrics": {key: float(results[key]) for key in ULTRALYTICS_KEYS if key in results},
        "per_image_mean": {metric: float(np.mean([row[metric] for row in rows])) for metric in METRIC_NAMES},
        "git": git_state(ROOT),
        "environment": environment(),
    }
    write_json(save_dir / "eval_summary.json", summary)
    if args.split == "test":
        record_test_scoring(args.project, name, save_dir, args.weights, weights_sha, args.data, args.mask_rule,
                            args.rescore_reason, labels)
    print(f"[INFO] Wrote {save_dir / 'eval_summary.json'}")
    for metric, value in summary["per_image_mean"].items():
        print(f"{metric}: {value:.4f}")
    return save_dir


if __name__ == "__main__":
    main()
