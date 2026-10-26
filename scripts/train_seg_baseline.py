"""Train a non-YOLO baseline (U-Net, MALUNet, EGE-UNet) with its official recipe on the exported canvases.

    python scripts/train_seg_baseline.py --model egeunet --seed 0
    python scripts/train_seg_baseline.py --model unet --seed 0 --epochs 2 --fraction 0.05   # smoke test

The recipe (loss, optimizer, schedule, epochs, batch, augmentation, normalisation) is
configs/baselines/<model>.yaml, transcribed from the official code (CLAUDE.md, "Baseline ngoài YOLO").
Training images are images/train and masks/train of --data (default configs/data/isic2018.yaml, view v1),
validation images/val. After every epoch the 100 val images are predicted (foreground probability >= 0.5)
and scored with artigate.metrics.binary_metrics; best.pt keeps the epoch with the highest mean Dice
(the earlier epoch on ties). If the recipe's batch does not fit the GPU it is halved until it does, and the
batch used is recorded in provenance.json.

Writes runs/baselines/<model>_s<seed>/{best.pt, last.pt, history.csv, provenance.json}. A folder with
provenance.json is a finished run; one with only last.pt is interrupted and continues with --resume.
"""
import argparse
import csv
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from artigate.baselines.data import CanvasSegDataset
from artigate.baselines.models import MODELS, THIRD_PARTY, build_model, split_output, upstream_utils
from artigate.baselines.recipe import (
    build_criterion, build_optimizer, build_scheduler, build_transform, compute_loss, load_recipe,
)
from artigate.metrics import METRIC_NAMES, binary_metrics
from artigate.provenance import environment, git_state, sha256_file, write_json

DEFAULT_DATA = ROOT / "configs" / "data" / "isic2018.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "baselines"
HISTORY_FIELDS = ["epoch", "lr", "train_loss", "val_loss", *[f"val_{m}" for m in METRIC_NAMES], "seconds"]


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=_project_path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=_project_path, default=None, help="recipe (default configs/baselines/<model>.yaml)")
    parser.add_argument("--project", type=_project_path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=None, help="run name (default <model>_s<seed>)")
    parser.add_argument("--device", default=None, help="CUDA index or 'cpu' (default: 0 if available)")
    parser.add_argument("--workers", type=int, default=None, help="data loader workers (default from the recipe)")
    parser.add_argument("--epochs", type=int, default=None, help="override the recipe (smoke tests only)")
    parser.add_argument("--batch", type=int, default=None, help="override the recipe batch (smoke tests only)")
    parser.add_argument("--fraction", type=float, default=None, help="train on a seeded subset (smoke tests only)")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from last.pt")
    parser.add_argument("--probe-batch", type=int, default=None, help=argparse.SUPPRESS)  # internal, see fitting_batch
    return parser.parse_args(argv)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def probe_batch(args, batch, device):
    """Child process of fitting_batch: two full training steps (incl. AdamW states) at ``batch``."""
    recipe, _ = load_recipe(args.model, args.config)
    transform, _ = build_transform(recipe, train=True)
    dataset = CanvasSegDataset(args.data, "train", transform)
    model = build_model(args.model, recipe["model"]["kwargs"]).to(device).train()
    criterion = build_criterion(recipe["loss"])
    optimizer = build_optimizer(recipe["optimizer"], model.parameters())
    images, masks = next(iter(DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)))
    for _ in range(2):
        optimizer.zero_grad()
        aux, prob = split_output(model(images.to(device)))
        compute_loss(criterion, aux, prob, masks.to(device)).backward()
        optimizer.step()
    torch.cuda.synchronize(device)


def fitting_batch(args, batch, device):
    """Largest of batch, batch/2, ... whose training step fits the GPU.

    Each size is tried in a fresh process: after a CUDA out-of-memory error inside cuDNN the process's
    CUDA context can stay unusable, and the parent must not hold GPU memory while probing.
    """
    if device.type != "cuda":
        return batch
    while batch >= 1:
        command = [sys.executable, str(Path(__file__).resolve()), "--model", args.model, "--data", str(args.data),
                   "--device", str(device.index or 0), "--probe-batch", str(batch)]
        if args.config is not None:
            command += ["--config", str(args.config)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return batch
        if "out of memory" not in result.stderr.lower():
            raise SystemExit(f"batch probe failed for another reason:\n{result.stderr[-3000:]}")
        batch //= 2
    raise SystemExit("even batch 1 does not fit the GPU")


@torch.no_grad()
def validate(model, dataset, criterion, threshold, device):
    """Mean val loss and mean per-image metrics; one image at a time, as the official val loader (batch 1)."""
    model.eval()
    losses, rows = [], []
    for index in range(len(dataset)):
        image, mask = dataset[index]
        aux, prob = split_output(model(image.unsqueeze(0).to(device)))
        losses.append(compute_loss(criterion, aux, prob, mask.unsqueeze(0).to(device)).item())
        rows.append(binary_metrics(prob[0, 0].cpu().numpy() >= threshold, mask[0].numpy() > 0.5))
    return float(np.mean(losses)), {m: float(np.mean([r[m] for r in rows])) for m in METRIC_NAMES}


def upstream_source(recipe):
    source = dict(recipe["model"].get("source", {}))
    name = recipe["model"]["name"]
    if name in ("egeunet", "malunet"):
        path = THIRD_PARTY / name / f"{name}.py"
        source["sha256"] = sha256_file(path) if path.is_file() else None
    return source


def main(argv=None):
    args = parse_args(argv)
    recipe, recipe_path = load_recipe(args.model, args.config)
    if recipe["model"]["name"] != args.model:
        raise SystemExit(f"{recipe_path} is a {recipe['model']['name']} recipe, not {args.model}")
    overrides = {k: v for k, v in (("epochs", args.epochs), ("batch", args.batch), ("fraction", args.fraction)) if v is not None}
    epochs = args.epochs or recipe["epochs"]
    workers = recipe["num_workers"] if args.workers is None else args.workers
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    if args.probe_batch is not None:
        return probe_batch(args, args.probe_batch, device)

    run_dir = args.project / (args.name or f"{args.model}_s{args.seed}")
    last_path = run_dir / "last.pt"
    if (run_dir / "provenance.json").is_file():
        raise SystemExit(f"{run_dir} is a finished run (provenance.json exists)")
    if last_path.is_file() and not args.resume:
        raise SystemExit(f"{run_dir} holds an interrupted run: pass --resume, or move the folder away")
    run_dir.mkdir(parents=True, exist_ok=True)
    requested = args.batch or recipe["batch"]
    resuming = args.resume and last_path.is_file()
    batch = requested if resuming else fitting_batch(args, requested, device)

    upstream_utils().set_seed(args.seed)  # the official set_seed: python, numpy, torch, cudnn deterministic
    train_transform, angle = build_transform(recipe, train=True)
    val_transform, _ = build_transform(recipe, train=False)
    train_set = CanvasSegDataset(args.data, "train", train_transform, args.fraction, args.seed)
    val_set = CanvasSegDataset(args.data, "val", val_transform)
    model = build_model(args.model, recipe["model"]["kwargs"]).to(device)
    criterion = build_criterion(recipe["loss"])
    optimizer = build_optimizer(recipe["optimizer"], model.parameters())
    scheduler = build_scheduler(recipe["scheduler"], optimizer)

    start_epoch, best = 1, {"val_dice": -1.0, "epoch": None}
    if resuming:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best, batch, angle = state["epoch"] + 1, state["best"], state["batch"], state["rotation_angle"]
        train_transform.transforms[-2].angle = angle  # myRandomRotation, the angle of the original run
        set_rng_state(state["rng"])
        print(f"[RESUME] {run_dir.name} from epoch {start_epoch}", flush=True)
    else:
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writeheader()
    if batch != requested:
        print(f"[BATCH] {requested} does not fit {device}; training with {batch}", flush=True)
    loader = DataLoader(train_set, batch_size=batch, shuffle=True, pin_memory=device.type == "cuda",
                        num_workers=workers, persistent_workers=workers > 0)

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for images, masks in loader:
            optimizer.zero_grad()
            aux, prob = split_output(model(images.to(device, non_blocking=True)))
            loss = compute_loss(criterion, aux, prob, masks.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        val_loss, val = validate(model, val_set, criterion, recipe["threshold"], device)
        row = {"epoch": epoch, "lr": lr, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               **{f"val_{m}": v for m, v in val.items()}, "seconds": round(time.time() - started, 1)}
        with (run_dir / "history.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(row)
        if val["dice"] > best["val_dice"]:
            best = {"val_dice": val["dice"], "epoch": epoch}
            torch.save({"model_name": args.model, "kwargs": recipe["model"]["kwargs"], "recipe": recipe,
                        "state_dict": model.state_dict(), "epoch": epoch, "val_dice": val["dice"], "seed": args.seed},
                       run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best": best, "batch": batch, "rotation_angle": angle, "rng": rng_state()}, last_path)
        print(f"[EPOCH {epoch}/{epochs}] loss {row['train_loss']:.4f} val loss {val_loss:.4f} "
              f"val dice {val['dice']:.4f} (best {best['val_dice']:.4f} @ {best['epoch']}) {row['seconds']}s", flush=True)

    write_json(run_dir / "provenance.json", {
        "model": args.model,
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "recipe": recipe,
        "upstream": upstream_source(recipe),
        "data_yaml": str(args.data),
        "seed": args.seed,
        "overrides": overrides,
        "epochs": epochs,
        "batch_requested": requested,
        "batch_used": batch,
        "rotation_angle": angle,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "best_epoch": best["epoch"],
        "best_val_dice": best["val_dice"],
        "device": str(device),
        "num_workers": workers,
        "git": git_state(ROOT),
        "environment": environment(),
    })
    print(f"[INFO] Run saved to {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
