"""Exported 640x640 canvases (images/<split>, masks/<split>) as a dataset for the baseline networks."""
import math
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from artigate.data.dataset_yaml import load_dataset_yaml, split_dir
from artigate.data.masks import read_mask

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def canvas_items(data_yaml, split):
    """[(image path, mask path)] of a split, sorted by file name."""
    config, root = load_dataset_yaml(data_yaml)
    image_dir, mask_dir = split_dir(root, config, split, "images"), split_dir(root, config, split, "masks")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no images in {image_dir}")
    items = [(p, mask_dir / f"{p.stem}.png") for p in images]
    missing = [str(m) for _, m in items if not m.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} canvas masks missing, e.g. {missing[0]}")
    return items


class CanvasSegDataset(Dataset):
    """Image and mask as datasets/dataset.py of the official code reads them, then the recipe transform.

    Image: RGB uint8 array; mask: (H, W, 1) array of 0/1 (the official code divides the 0/255 PNG by 255).
    ``fraction`` keeps a seeded random subset (smoke tests only).
    """

    def __init__(self, data_yaml, split, transform, fraction=None, seed=0):
        items = canvas_items(data_yaml, split)
        if fraction is not None and fraction < 1:
            keep = random.Random(seed).sample(range(len(items)), max(1, math.ceil(fraction * len(items))))
            items = [items[i] for i in sorted(keep)]
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, mask_path = self.items[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.expand_dims(read_mask(mask_path).astype(np.float64), axis=2)
        image, mask = self.transform((image, mask))
        return image.float(), mask.float()

    def stems(self):
        return [p.stem for p, _ in self.items]


def canvas_mask(prob, shape, threshold):
    """Binary mask on the canvas: the probability map resized to ``shape`` (bilinear), then >= threshold.

    Used for checkpoint selection (train_seg_baseline.validate) and for scoring (eval_seg_baseline), so
    both see the same prediction (CLAUDE.md: probabilities go back to the 640 canvas before the threshold).
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.shape != tuple(shape):
        prob = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob >= threshold


@torch.inference_mode()
def predict_probability(model, image, device):
    """Main foreground probability (H, W) for one transformed image tensor (C, H, W)."""
    from artigate.baselines.models import split_output

    _, prob = split_output(model(image.unsqueeze(0).to(device).float()))
    return prob[0, 0].float().cpu().numpy()
