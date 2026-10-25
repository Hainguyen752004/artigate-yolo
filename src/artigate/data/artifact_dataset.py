"""Training dataset that composites synthetic artifacts and carries artifact targets.

Each training image is returned with 7 channels: B, G, R, hair, ruler, marker, dark_corner. The
artifact channels are uint8 codes: 255 = present (synthetic), 0 = known absent, 114 = unknown
(ignored by the loss). Because all 7 channels go through Ultralytics' mosaic, affine, flip and
letterbox together, the artifact maps stay aligned with the image without touching the
augmentation code; padding added by the augmentations becomes 114 or 0, both harmless.
``ArtiGateTrainer.preprocess_batch`` splits the channels again before the forward pass.

Configured by an ``artigate:`` block in the dataset YAML:

    artigate:
      p_synth: 0.7                       # share of training images that get synthetic artifacts
      severity: [1, 4]                   # inclusive range, sampled per artifact
      type_probs: [0.5, 0.3, 0.3, 0.3]   # hair, ruler, marker, dark_corner
      auto_neg: true                     # conservative pixel negatives (see auto_negatives)
      flags_csv: data/splits/isic2018_train_flags.csv   # optional; relative to the repo root

Artifacts are drawn inside the letterboxed content area (from metadata/<split>.jsonl), never on
the grey padding.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np
from ultralytics.data.augment import Compose, RandomHSV
from ultralytics.data.dataset import YOLODataset

from artigate.data.synth_artifacts import TYPES, composite

POSITIVE, NEGATIVE, UNKNOWN = 255, 0, 114
REPO_ROOT = Path(__file__).resolve().parents[3]
_VIEW_SUFFIX = re.compile(r"_v\d+$")


def sample_id(stem):
    """``ISIC_0000000_v1`` -> ``ISIC_0000000`` (exported stems carry the view suffix)."""
    return _VIEW_SUFFIX.sub("", stem)


def resolve(path):
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def load_flags(path):
    """sample_id -> int8 array (K,) with 1 present, 0 absent, -1 unknown."""
    if not path:
        return {}
    path = resolve(path)
    if not path.is_file():
        print(f"[artigate] WARNING: flags_csv {path} not found - training without image-level flags")
        return {}
    flags = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = sample_id(Path(row.get("image_id") or row.get("sample_id") or next(iter(row.values()))).stem)
            values = []
            for name in TYPES:
                value = str(row.get(name, "")).strip().lower()
                values.append(1 if value in ("1", "1.0", "true", "yes") else 0 if value in ("0", "0.0", "false", "no") else -1)
            flags[key] = np.array(values, np.int8)
    return flags


def auto_negatives(image, lesion_box=None):
    """Pixels that almost surely do NOT show each artifact type, as an (H, W, K) bool array.

    Deliberately conservative; anything that could be an artifact stays unknown:
        hair         no thin dark structure (morphological black-hat <= 8)
        ruler        not white/grey and not very dark
        marker       not purple/blue ink and not very dark
        dark_corner  brightness >= 70
    Rulers and markers are also negative inside the shrunken lesion box.
    """
    height, width = image.shape[:2]
    side = min(height, width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k = max(9, int(0.025 * side) | 1)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    kernel = np.ones((5, 5), np.uint8)

    def grow(candidate):
        return cv2.dilate(candidate.astype(np.uint8), kernel) > 0

    dark = val < 60
    negative = np.zeros((height, width, len(TYPES)), bool)
    negative[..., 0] = ~grow(blackhat > 8)
    negative[..., 1] = ~grow(((sat < 40) & (val > 150)) | dark)
    negative[..., 2] = ~grow(((hue >= 95) & (hue <= 170) & (sat > 40)) | dark)
    negative[..., 3] = val >= 70
    if lesion_box is not None:
        x1, y1, x2, y2 = lesion_box
        dx, dy = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
        negative[y1 + dy:y2 - dy, x1 + dx:x2 - dx, 1:3] = True
    return negative


def content_boxes(image_dir):
    """stem -> (x1, y1, x2, y2) of the non-padded area on the export canvas, from metadata/<split>.jsonl."""
    image_dir = Path(image_dir)
    metadata = image_dir.parent.parent / "metadata" / f"{image_dir.name}.jsonl"
    boxes = {}
    if not metadata.is_file():
        return boxes
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        _, _, crop_w, crop_h = record["crop_box"]
        scale, (pad_x, pad_y) = record["scale"], record["pad"]
        boxes[record["stem"]] = (pad_x, pad_y, pad_x + int(crop_w * scale), pad_y + int(crop_h * scale)), record["canvas_size"]
    return boxes


class HSVFirst3(RandomHSV):
    """RandomHSV applied to the colour channels of a multi-channel image."""

    def __call__(self, labels):
        image = labels["img"]
        if image.shape[-1] == 3:
            return super().__call__(labels)
        colour = {"img": np.ascontiguousarray(image[..., :3])}
        super().__call__(colour)
        image[..., :3] = colour["img"]
        return labels


def _swap_hsv(transform):
    if isinstance(transform, Compose):
        transform.transforms = [_swap_hsv(t) for t in transform.transforms]
        return transform
    if isinstance(transform, RandomHSV) and not isinstance(transform, HSVFirst3):
        return HSVFirst3(hgain=transform.hgain, sgain=transform.sgain, vgain=transform.vgain)
    return transform


class ArtiGateDataset(YOLODataset):
    def __init__(self, *args, art_cfg=None, **kwargs):
        config = dict(art_cfg or {})
        self.p_synth = float(config.get("p_synth", 0.7))
        severity = config.get("severity", [1, 4])
        self.severity = tuple(severity) if isinstance(severity, (list, tuple)) else (int(severity), int(severity))
        self.type_probs = tuple(config.get("type_probs", [0.5, 0.3, 0.3, 0.3]))
        self.auto_neg = bool(config.get("auto_neg", True))
        self.flags = load_flags(config.get("flags_csv"))
        self._art_ready = False  # no compositing while the parent class caches images
        super().__init__(*args, **kwargs)
        self.content = content_boxes(kwargs.get("img_path") or args[0])
        self._art_ready = True

    def build_transforms(self, hyp=None):
        return _swap_hsv(super().build_transforms(hyp))

    def load_image(self, i, rect_mode=True):
        image, hw0, hw = super().load_image(i, rect_mode)
        if not (self.augment and self._art_ready):
            return image, hw0, hw
        return self.add_artifacts(i, image), hw0, hw

    def _boxes(self, i, height, width):
        """Content box and lesion box in pixels of the loaded image."""
        stem = Path(self.im_files[i]).stem
        content = (0, 0, width, height)
        if stem in self.content:
            (x1, y1, x2, y2), canvas = self.content[stem]
            sx, sy = width / canvas, height / canvas
            content = (int(x1 * sx), int(y1 * sy), max(int(x1 * sx) + 1, int(x2 * sx)), max(int(y1 * sy) + 1, int(y2 * sy)))
        lesion = None
        boxes = self.labels[i].get("bboxes")
        if boxes is not None and len(boxes):
            b = np.asarray(boxes, np.float32)  # normalised xywh
            lesion = (int(max(0, ((b[:, 0] - b[:, 2] / 2) * width).min())), int(max(0, ((b[:, 1] - b[:, 3] / 2) * height).min())),
                      int(min(width, ((b[:, 0] + b[:, 2] / 2) * width).max())), int(min(height, ((b[:, 1] + b[:, 3] / 2) * height).max())))
        return content, lesion

    def add_artifacts(self, i, image):
        image = image[..., :3].copy()
        height, width = image.shape[:2]
        (cx1, cy1, cx2, cy2), lesion = self._boxes(i, height, width)
        art = np.full((height, width, len(TYPES)), UNKNOWN, np.uint8)
        if self.auto_neg:
            art[auto_negatives(image, lesion)] = NEGATIVE
        flags = self.flags.get(sample_id(Path(self.im_files[i]).stem))
        if flags is not None:
            art[..., flags == 0] = NEGATIVE
        if np.random.random() < self.p_synth:
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))  # follows Ultralytics worker seeding
            crop = image[cy1:cy2, cx1:cx2]
            local = None if lesion is None else (lesion[0] - cx1, lesion[1] - cy1, lesion[2] - cx1, lesion[3] - cy1)
            masks, touched, _ = composite(crop, rng, local, self.type_probs, self.severity)
            region = art[cy1:cy2, cx1:cx2]
            region[touched & ~masks] = UNKNOWN  # anti-aliased edges and soft fall-off: neither in nor out
            region[masks] = POSITIVE
        return np.concatenate((image, art), axis=2)
