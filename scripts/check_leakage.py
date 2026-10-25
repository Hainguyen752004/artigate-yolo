#!/usr/bin/env python
"""Find the same image (by id or near-duplicate pixels) in two image folders.

export_yolo.py already rejects the same ISIC id in two splits of one manifest; this script also
catches the same photo under different names and across datasets. Run it for every train/test pair
you report, e.g. ISIC 2018 train vs HAM10000 test, and ISIC 2018 train vs PH2. Candidates come from a 64-bit difference hash (Hamming distance <= --max-dist);
each candidate is then scored by the correlation of 64x64 grey thumbnails (corr >= 0.95 = likely the
same photo). Dermoscopy images look alike, so always check the flagged pairs by eye.

    python scripts/check_leakage.py --a data/yolo/isic2018/images/train --b data/yolo/ham10000/images/test \
        --out runs/leakage/isic_train_vs_ham_test.csv
Exported images are letterboxed, so compare exports made with the same canvas size and view.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def dhash(path, size=8):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    return np.packbits(bits).view(np.uint64)[0]


def hashes(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXT)
    return files, np.array([dhash(p) for p in files], np.uint64)


def thumb_corr(pa, pb):
    a, b = (cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for p in (pa, pb))
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


_LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def popcount64(x):
    x = np.ascontiguousarray(x, np.uint64)
    return _LUT[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dist", type=int, default=4)
    a = ap.parse_args()

    fa, ha = hashes(a.a)
    fb, hb = hashes(a.b)
    ka, kb = [p.stem for p in fa], [p.stem for p in fb]
    same_id = sorted(set(ka) & set(kb))
    rows = [(k, k, "same_id", "", "") for k in same_id]
    for i in range(0, len(ha), 512):
        d = popcount64(ha[i : i + 512, None] ^ hb[None, :])
        for r, c in zip(*np.where(d <= a.max_dist)):
            if ka[i + r] != kb[c]:
                corr = thumb_corr(fa[i + r], fb[c])
                kind = "likely_duplicate" if corr >= 0.95 else "hash_match_only"
                rows.append((ka[i + r], kb[c], kind, int(d[r, c]), round(corr, 4)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "kind", "hamming", "thumb_corr"])
        w.writerows(rows)
    nl = sum(r[2] == "likely_duplicate" for r in rows)
    nh = sum(r[2] == "hash_match_only" for r in rows)
    print(f"A={len(ka)} B={len(kb)}: {len(same_id)} shared ids, {nl} likely duplicates, {nh} weaker hash matches -> {a.out}")
    if rows:
        print("Look at the pairs; remove true duplicates from the test side before reporting results.")


if __name__ == "__main__":
    main()
