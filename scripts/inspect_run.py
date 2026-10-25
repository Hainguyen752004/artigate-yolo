"""Milestone 1 check: has AGA started to use the artifact map? Works on a running or finished run.

    python scripts/inspect_run.py runs/segment/A4_s0

Reads weights/last.pt (rewritten every epoch) and prints, for each AGA block, beta and the norm of the
spatial-attention weights on the artifact channel. Both start at exactly 0, so values still at 0 after
20-30 epochs mean the gate is unused. Also prints the latest art_loss and mask mAP50-95 from
results.csv. For how well the artifact head finds artifacts, run scripts/eval_artifact_head.py on the
same last.pt with --split val.
"""
import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from artigate.modules import AGA, register_modules


def gate_values(model):
    """[(layer name, beta, |artifact-channel weights|)] for every AGA block of ``model``."""
    values = []
    for name, module in model.named_modules():
        if isinstance(module, AGA):
            map_weight = float(module.conv.weight[:, 2].norm()) if module.use_map else float("nan")
            values.append((name, float(module.beta), map_weight))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--weights", default="last.pt", help="checkpoint inside <run>/weights")
    args = parser.parse_args(argv)

    register_modules()
    checkpoint = torch.load(args.run / "weights" / args.weights, map_location="cpu", weights_only=False)
    model = (checkpoint.get("ema") or checkpoint.get("model")).float()
    print(f"{args.run.name}: epoch {checkpoint.get('epoch', '?')} ({args.weights}, EMA weights)")
    values = gate_values(model)
    if not values:
        print("  no AGA block in this model")
    for name, beta, map_weight in values:
        print(f"  AGA {name}: beta {beta:+.4f}   |w on artifact channel| {map_weight:.4f}")
    results = args.run / "results.csv"
    if results.is_file():
        rows = list(csv.DictReader(results.open(encoding="utf-8")))
        if rows:
            last = {key.strip(): value for key, value in rows[-1].items()}
            print(f"  results.csv epoch {last.get('epoch')}: train/art_loss {last.get('train/art_loss', 'n/a')}, "
                  f"val mask mAP50-95 {last.get('metrics/mAP50-95(M)', 'n/a')}")
    return values


if __name__ == "__main__":
    main()
