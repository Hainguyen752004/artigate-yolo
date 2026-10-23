"""Save a grid of augmented ArtiGate training samples with their artifact targets (sanity check).

Each tile: the augmented image, the lesion outline (green) and the artifact targets as colour
overlays (hair red, ruler yellow, marker magenta, dark corner blue). Unknown pixels are not drawn;
the tile title gives their share. Use it to confirm the maps stay aligned after mosaic and affine.

    python scripts/viz_train_batch.py --data configs/data/isic2018_art.yaml --out runs/viz/train_batch.jpg
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import yaml

from artigate.data.dataset_yaml import runtime_dataset_yaml

COLOURS = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0)]  # BGR
DEFAULT_HYP = ROOT / "configs" / "train" / "default.yaml"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "configs" / "data" / "isic2018_art.yaml"))
    parser.add_argument("--hyp", default=str(DEFAULT_HYP), help="training schedule whose augmentations to apply")
    parser.add_argument("--out", default=str(ROOT / "runs" / "viz" / "train_batch.jpg"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    from artigate.data.artifact_dataset import ArtiGateDataset, UNKNOWN

    hyp = yaml.safe_load(Path(args.hyp).read_text(encoding="utf-8")) or {}
    keep = ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale", "shear", "perspective", "flipud",
            "fliplr", "hsv_h", "hsv_s", "hsv_v", "mask_ratio")
    cfg = get_cfg(overrides={"task": "segment", "imgsz": args.imgsz, **{k: v for k, v in hyp.items() if k in keep}})
    with runtime_dataset_yaml(args.data) as runtime:
        data = check_det_dataset(str(runtime))
    dataset = ArtiGateDataset(
        img_path=data["train"], imgsz=args.imgsz, batch_size=args.n, augment=True, hyp=cfg, rect=False, cache=None,
        single_cls=False, stride=32, pad=0.0, prefix="viz: ", task="segment", classes=None, data=data, fraction=1.0,
        art_cfg=data.get("artigate") or {},
    )
    np.random.seed(args.seed)
    tiles = []
    for index in np.random.choice(len(dataset), args.n, replace=len(dataset) < args.n):
        sample = dataset[int(index)]
        x = sample["img"].numpy()
        image = x[:3].transpose(1, 2, 0).copy()
        if x.shape[0] == 3:
            image = image[..., ::-1].copy()
        vis = image.astype(np.float32)
        for k in range(3, x.shape[0]):
            positive = x[k] >= 192
            vis[positive] = 0.45 * vis[positive] + 0.55 * np.array(COLOURS[k - 3], np.float32)
        vis = vis.astype(np.uint8)
        masks = sample["masks"].numpy()
        lesion = (masks > 0).any(0) if masks.ndim == 3 else masks > 0
        lesion = cv2.resize(lesion.astype(np.uint8), vis.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        unknown = (x[3:] == UNKNOWN).mean() if x.shape[0] > 3 else 1.0
        cv2.putText(vis, f"channels={x.shape[0]} unknown={unknown:.0%}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        tiles.append(vis)
    columns = 4
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[r:r + columns], 1) for r in range(0, len(tiles), columns)], 0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print("saved", args.out)


if __name__ == "__main__":
    main()
