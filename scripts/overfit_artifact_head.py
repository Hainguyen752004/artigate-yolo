"""Sanity check: can the artifact head learn the synthetic artifacts at all? (artifact loss only)

Trains with ONLY the artifact loss on the training images (mosaic + synthetic artifacts) for a few
hundred steps and prints the IoU of each type on the training batches at the head's resolution.
On toy data with the stock backbone this reached IoU 0.95 (hair) / 0.93 (dark corner) in 300
steps. If IoU stays near 0, the targets, the loss or the head are broken.

    python scripts/overfit_artifact_head.py --data configs/data/isic2018_art.yaml --iters 300
"""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from artigate.data.dataset_yaml import runtime_dataset_yaml
from artigate.data.synth_artifacts import TYPES
from artigate.modules import register_modules

DEFAULT_MODEL = ROOT / "configs" / "models" / "yolo26n-seg-p2-artigate.yaml"
DEFAULT_DATA = ROOT / "configs" / "data" / "isic2018_art.yaml"
DEFAULT_PRETRAINED = ROOT / "weights" / "yolo26n-seg.pt"


def pick_device(value):
    if value is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(f"cuda:{value}" if str(value).isdigit() else value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED), help="stock checkpoint or 'none'")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    register_modules()
    from torch.utils.data import DataLoader
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    from artigate.artifact_loss import ArtiGateSegModel, artifact_logits, artifact_loss, artifact_targets
    from artigate.data.artifact_dataset import ArtiGateDataset
    from artigate.transfer import load_partial_pretrained, stock_layer_map

    torch.manual_seed(0)
    device = pick_device(args.device)
    with runtime_dataset_yaml(args.data) as runtime:
        data = check_det_dataset(str(runtime))
    hyp = get_cfg(overrides={"imgsz": args.imgsz, "task": "segment"})
    dataset = ArtiGateDataset(
        img_path=data["train"], imgsz=args.imgsz, batch_size=args.batch, augment=True, hyp=hyp, rect=False, cache=None,
        single_cls=False, stride=32, pad=0.0, prefix="overfit: ", task="segment", classes=None, data=data,
        fraction=1.0, art_cfg={**(data.get("artigate") or {}), "p_synth": 1.0},
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=dataset.collate_fn, drop_last=True)
    model = ArtiGateSegModel(args.model, nc=1, verbose=False)
    if args.pretrained.lower() != "none":
        load_partial_pretrained(model, args.pretrained, stock_layer_map(args.model))
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    step, start = 0, time.time()
    inter, union = torch.zeros(len(TYPES)), torch.zeros(len(TYPES))
    while step < args.iters:
        for batch in loader:
            image = batch["img"].to(device).float() / 255
            art, colour = image[:, 3:], image[:, :3].flip(1)
            model(colour)
            logits = artifact_logits(model)
            loss = artifact_loss(logits, art)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                target, weight = artifact_targets(art, logits.shape[-2:])
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter += (pred * target * weight).sum((0, 2, 3)).cpu()
                union += (((pred + target) > 0).float() * weight).sum((0, 2, 3)).cpu()
            step += 1
            if step % args.every == 0 or step == args.iters:
                iou = (inter / union.clamp(min=1)).tolist()
                print(f"step {step:>5}  loss {loss.item():.3f}  IoU " + "  ".join(f"{n}={v:.3f}" for n, v in zip(TYPES, iou))
                      + f"  ({time.time() - start:.0f}s)", flush=True)
                inter.zero_()
                union.zero_()
            if step >= args.iters:
                break


if __name__ == "__main__":
    main()
