"""Paper figure: the procedural artifacts at severities 1-4 with their masks.

Rows: hair, ruler, marker ink, dark corner. Columns: severity 1..4, each as an image tile and a
mask tile (white = present label, grey = touched but unlabelled edge, black = untouched).

Usage:
    # placeholder on a drawn lesion (no data needed)
    python scripts/figure_synthetic_artifacts.py --out paper/figures/synthetic_artifacts.jpg
    # final version on an ISIC 2018 training image and its ground-truth mask
    python scripts/figure_synthetic_artifacts.py --image path/to/ISIC_xxx.jpg \
        --mask path/to/ISIC_xxx_segmentation.png --out paper/figures/synthetic_artifacts.jpg

The same seed draws the same artifacts on every run. Use a training image only: test images must
not appear in method figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artigate.data.synth_artifacts import TYPES, composite  # noqa: E402

ROW_NAMES = {"hair": "Hair", "ruler": "Ruler", "marker": "Marker ink", "dark_corner": "Dark corner"}


def drawn_lesion(size: int = 384, seed: int = 0):
    """A plain skin background with one irregular brown lesion (placeholder only)."""
    rng = np.random.default_rng(seed)
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    skin = np.array([150, 170, 215], np.float32)  # BGR
    img = skin + rng.normal(0, 4, (H, W, 1)).astype(np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 12, (H, W)).astype(np.float32), (0, 0), 12)[..., None]
    ang = np.arctan2(yy - H / 2, xx - W / 2)
    r = np.hypot(xx - W / 2, (yy - H / 2) * 1.15)
    radius = 0.24 * size * (1 + 0.10 * np.sin(3 * ang + 0.7) + 0.06 * np.sin(5 * ang + 2.1))
    t = np.clip((radius - r) / (0.03 * size), 0, 1)
    core = np.array([45, 65, 110], np.float32) + 25 * (r / radius)[..., None]
    img = img * (1 - t[..., None]) + core * t[..., None]
    gt = (t > 0.5).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8), gt


def load_pair(image: str, mask: str, size: int):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    gt = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        raise SystemExit(f"cannot read {image} or {mask}")
    h, w = img.shape[:2]
    s = min(h, w)  # centre square crop so every tile has the same shape
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = cv2.resize(img[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_NEAREST)
    return img, (gt > 127).astype(np.uint8)


def lesion_box(gt):
    ys, xs = np.nonzero(gt)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_tile(core, touched):
    tile = np.zeros(core.shape + (3,), np.uint8)
    tile[touched] = (110, 110, 110)
    tile[core] = (255, 255, 255)
    return tile


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="training image (omit for a drawn placeholder lesion)")
    ap.add_argument("--mask", help="ground-truth lesion mask for --image")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="paper/figures/synthetic_artifacts.jpg")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    if bool(args.image) != bool(args.mask):
        ap.error("--image and --mask go together")
    base, gt = load_pair(args.image, args.mask, args.size) if args.image else drawn_lesion(args.size)
    box = lesion_box(gt)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(TYPES), 8, figsize=(7.16, 3.85))
    for r, name in enumerate(TYPES):
        for sev in range(1, 5):
            img = base.copy()
            rng = np.random.default_rng([args.seed, r, sev])
            masks, touched, applied = composite(img, rng, lesion_box=box, severity=sev, force=name)
            k = TYPES.index(name)
            ax_i, ax_m = axes[r, 2 * (sev - 1)], axes[r, 2 * (sev - 1) + 1]
            ax_i.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax_m.imshow(mask_tile(masks[..., k], touched[..., k]))
            if not applied:
                ax_m.text(0.5, 0.5, "not placed", color="w", ha="center", va="center", fontsize=6, transform=ax_m.transAxes)
            if r == 0:
                ax_i.set_title(f"Severity {sev}", fontsize=7, loc="left", pad=2)
        axes[r, 0].set_ylabel(ROW_NAMES[name], fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.95, bottom=0.005, wspace=0.03, hspace=0.04)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, pil_kwargs={"quality": 92})
    print(f"wrote {out} ({'image ' + args.image if args.image else 'drawn placeholder lesion'}, seed {args.seed})")


if __name__ == "__main__":
    main()
