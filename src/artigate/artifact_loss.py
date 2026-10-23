"""Artifact loss and the segmentation model that trains the artifact head.

    L = L_YOLO-seg + lambda * mean_k (BCE_k + Dice_k)      over labelled pixels only

Targets arrive in ``batch["art"]`` with shape (B, K, H, W) and values in [0, 1], encoded by
``artigate.data.artifact_dataset``: >= 0.75 artifact present, <= 0.25 known absent, anything in
between unknown and ignored (partial labels).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import SegmentationModel

from artigate.modules import ArtifactHead

# Thresholds on the artifact channels after /255 (codes 255 present, 0 absent, 114 = 0.447 unknown).
# Absent must stay clearly below the unknown code: mix-up of absent with unknown (~57 = 0.22) is unknown.
POSITIVE, NEGATIVE = 0.75, 0.1


def artifact_targets(art, size):
    """(target, weight) at the logits resolution.

    A low-resolution cell is positive if any pixel inside is positive, negative if every pixel is
    negative, and ignored otherwise.
    """
    art = art.float()
    pos = (art >= POSITIVE).float()
    neg = (art <= NEGATIVE).float()
    if tuple(art.shape[-2:]) != tuple(size):
        pos = F.adaptive_max_pool2d(pos, size)
        neg = -F.adaptive_max_pool2d(-neg, size)
    return pos, torch.clamp(pos + neg, max=1.0)


def artifact_loss(logits, art, smooth=1.0):
    """Partial-label BCE + soft Dice, averaged over the channels that have labelled pixels."""
    logits = logits.float()
    target, weight = artifact_targets(art.to(logits.device), logits.shape[-2:])
    weight_sum = weight.sum((0, 2, 3))
    valid = weight_sum > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_k = (bce * weight).sum((0, 2, 3)) / weight_sum.clamp(min=1.0)
    prob = torch.sigmoid(logits)
    inter = (prob * target * weight).sum((0, 2, 3))
    denom = (prob * weight).sum((0, 2, 3)) + (target * weight).sum((0, 2, 3))
    dice_k = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    return (bce_k + dice_k)[valid].mean()


class ArtiGateLoss:
    """Ultralytics segmentation criterion + artifact loss, logged as the last item (``art_loss``).

    Models without an ArtifactHead get a constant 0 item, so every ablation row logs the same columns.
    """

    def __init__(self, model, base):
        self.base = base
        self.heads = [m for m in model.modules() if isinstance(m, ArtifactHead)]
        config = (getattr(model, "yaml", None) or {}).get("art_loss") or {}
        self.weight = float(config.get("lambda", 1.0))

    def __getattr__(self, name):  # update() and the rest go to the wrapped criterion
        if name.startswith("__") or "base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["base"], name)

    def __call__(self, preds, batch):
        total, items = self.base(preds, batch)
        art = batch.get("art")
        logits = self.heads[0].last_logits if self.heads else None
        if logits is None:
            loss = total.new_zeros(())
        elif art is None or self.weight == 0:
            loss = logits.sum() * 0.0  # keeps the head in the graph (DDP) without training it
        else:
            loss = artifact_loss(logits, art) * self.weight
        batch_size = batch["img"].shape[0]
        total = torch.cat((total, (loss * batch_size).reshape(1).to(total.dtype)))
        items = torch.cat((items, loss.detach().reshape(1).to(items.dtype)))
        return total, items


class ArtiGateSegModel(SegmentationModel):
    """SegmentationModel whose criterion also trains the artifact head when the YAML has one."""

    def init_criterion(self):
        return ArtiGateLoss(self, super().init_criterion())


def artifact_logits(model):
    """Logits from the last forward pass of ``model``'s ArtifactHead (None if it has none)."""
    from ultralytics.utils.torch_utils import unwrap_model

    for module in unwrap_model(model).modules():
        if isinstance(module, ArtifactHead):
            return module.last_logits
    return None
