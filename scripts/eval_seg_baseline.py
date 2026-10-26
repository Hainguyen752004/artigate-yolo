"""Score a non-YOLO baseline on one split with the metric code and test-scoring rules of scripts/eval.py.

    python scripts/eval_seg_baseline.py --weights runs/baselines/egeunet_s0/best.pt --split val
    python scripts/eval_seg_baseline.py --pred-dir <nnUNetv2_predict output> \
        --checkpoint <nnUNet_results>/Dataset501_ISIC2018Canvas/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth \
        --run nnunet_s0 --split val

Prediction (CLAUDE.md, "Baseline ngoài YOLO"): foreground probability >= 0.5 on the 640x640 canvas, no TTA,
no post-processing. A checkpoint from train_seg_baseline.py carries its recipe (normalisation, input
size). nnU-Net predictions are the label PNGs of nnUNetv2_predict --disable_tta with checkpoint_final.pth,
named <image stem>.png (argmax of the two softmax classes = foreground probability >= 0.5); --checkpoint
identifies the model in the scoring log.

Writes <project>/<run>_<data>_<split>/per_image_metrics.csv (same columns as eval.py, so compare_family.py
and bootstrap_ci.py read it) and eval_summary.json. Works for any exported set with canvas masks: ISIC 2018,
PH2, HAM10000, DDI/FEDD and the severity sets. A test split is scored once per checkpoint and dataset,
and only after the test-set artifact labels are committed (artigate.scoring), exactly as eval.py.
"""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image
import torch

from artigate.baselines.data import canvas_items, predict_probability
from artigate.baselines.models import build_model
from artigate.baselines.recipe import build_transform
from artigate.data.masks import read_mask
from artigate.metrics import METRIC_NAMES, binary_metrics
from artigate.provenance import environment, git_state, sha256_file, write_json
from artigate.scoring import default_name, output_dir, record_test_scoring, test_label_record

DEFAULT_DATA = ROOT / "configs" / "data" / "isic2018.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "eval"
MASK_RULE = "prob0.5"  # key of the scoring log: one binary mask per image, foreground probability >= 0.5


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=_project_path, help="best.pt of scripts/train_seg_baseline.py")
    source.add_argument("--pred-dir", type=_project_path, help="nnU-Net label PNGs named <image stem>.png")
    parser.add_argument("--checkpoint", type=_project_path, help="with --pred-dir: the nnU-Net checkpoint that made them")
    parser.add_argument("--run", default=None, help="run name in the output name (required with --pred-dir)")
    parser.add_argument("--data", type=_project_path, default=DEFAULT_DATA)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--name", default=None, help="output name (default <run>_<data>_<split>)")
    parser.add_argument("--project", type=_project_path, default=DEFAULT_PROJECT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rescore-reason", default=None,
                        help="why a test split already scored with this checkpoint must be scored again (logged)")
    args = parser.parse_args(argv)
    if args.pred_dir is not None and (args.checkpoint is None or args.run is None):
        parser.error("--pred-dir needs --checkpoint and --run")
    return args


def read_label_png(path):
    """Prediction PNG with labels 0/1 (nnU-Net) as a boolean mask."""
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"missing or unreadable prediction {path}")
    return (label[..., 0] if label.ndim == 3 else label) > 0


class CheckpointPredictor:
    """Binary canvas mask from a train_seg_baseline.py checkpoint, with its recipe's test transform."""

    def __init__(self, weights, device):
        state = torch.load(weights, map_location="cpu", weights_only=False)
        self.model = build_model(state["model_name"], state["kwargs"])
        self.model.load_state_dict(state["state_dict"])
        self.model.to(device).eval()
        self.transform, _ = build_transform(state["recipe"], train=False)
        self.threshold, self.device = state["recipe"]["threshold"], device
        self.info = {"model": state["model_name"], "epoch": state["epoch"], "val_dice": state["val_dice"],
                     "seed": state["seed"], "threshold": self.threshold}

    def __call__(self, image_path, truth):
        image = np.array(Image.open(image_path).convert("RGB"))
        tensor, _ = self.transform((image, truth[..., None].astype(np.float64)))
        prob = predict_probability(self.model, tensor, self.device)
        if prob.shape != truth.shape:  # recipes at another input size: back to the canvas
            prob = cv2.resize(prob, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_LINEAR)
        return prob >= self.threshold


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model_file = args.weights or args.checkpoint
    model_sha = sha256_file(model_file)
    name = args.name or default_name(model_file, args.data, args.split, MASK_RULE, run=args.run)
    save_dir, _ = output_dir(args.project, name, args.split, model_sha, args.data, MASK_RULE, args.rescore_reason)
    items = canvas_items(args.data, args.split)
    labels = test_label_record(args.data, [p.stem for p, _ in items]) if args.split == "test" else None

    if args.weights is not None:
        predictor = CheckpointPredictor(args.weights, device)
        info = predictor.info
    else:
        predictor = lambda image_path, truth: read_label_png(args.pred_dir / f"{image_path.stem}.png")  # noqa: E731
        info = {"model": "nnunet", "pred_dir": str(args.pred_dir)}
    rows = []
    for image_path, mask_path in items:
        truth = read_mask(mask_path)
        prediction = predictor(image_path, truth)
        rows.append({"image_id": image_path.stem, **binary_metrics(prediction, truth)})

    save_dir.mkdir(parents=True, exist_ok=True)
    write_rows(save_dir / "per_image_metrics.csv", rows)
    summary = {
        "weights": str(model_file),
        "weights_sha256": model_sha,
        **info,
        "data_yaml": str(args.data),
        "split": args.split,
        "rescore_reason": args.rescore_reason,
        "test_artifact_labels": labels,
        "n_images": len(rows),
        "mask_rule": MASK_RULE,
        "per_image_mean": {m: float(np.mean([r[m] for r in rows])) for m in METRIC_NAMES},
        "git": git_state(ROOT),
        "environment": environment(),
    }
    write_json(save_dir / "eval_summary.json", summary)
    if args.split == "test":
        record_test_scoring(args.project, name, save_dir, model_file, model_sha, args.data, MASK_RULE,
                            args.rescore_reason, labels)
    print(f"[INFO] Wrote {save_dir / 'eval_summary.json'}")
    for metric, value in summary["per_image_mean"].items():
        print(f"{metric}: {value:.4f}")
    return save_dir


if __name__ == "__main__":
    main()
