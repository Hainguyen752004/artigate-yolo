"""The training recipe of configs/baselines/<model>.yaml: loss, optimizer, schedule and transforms.

Everything is built from the upstream classes (third_party/egeunet/egeunet_utils.py) with the
settings of the official config files, so the recipe matches the official code.
"""
from pathlib import Path

import torch
from torchvision import transforms
import yaml

from artigate.baselines.models import upstream_utils

CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "baselines"


def load_recipe(model=None, path=None):
    path = Path(path) if path is not None else CONFIGS / f"{model}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")), path


def build_criterion(loss):
    u = upstream_utils()
    if loss["name"] == "GT_BceDiceLoss":
        return u.GT_BceDiceLoss(wb=loss["wb"], wd=loss["wd"])
    if loss["name"] == "BceDiceLoss":
        return u.BceDiceLoss(wb=loss["wb"], wd=loss["wd"])
    raise ValueError(f"unknown loss {loss['name']!r}")


def compute_loss(criterion, aux, prob, target):
    """GT_BceDiceLoss takes the deep-supervision outputs as well (engine.py); BceDiceLoss does not."""
    if type(criterion).__name__ == "GT_BceDiceLoss":
        return criterion(aux, prob, target)
    return criterion(prob, target)


def build_optimizer(cfg, parameters):
    if cfg["name"] != "AdamW":
        raise ValueError(f"only the AdamW of the official configs is implemented, got {cfg['name']!r}")
    return torch.optim.AdamW(parameters, lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
                             weight_decay=cfg["weight_decay"], amsgrad=cfg["amsgrad"])


def build_scheduler(cfg, optimizer):
    """Stepped once per epoch, after the training loop (engine.train_one_epoch)."""
    if cfg["name"] != "CosineAnnealingLR":
        raise ValueError(f"only the CosineAnnealingLR of the official configs is implemented, got {cfg['name']!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])


def build_transform(recipe, train):
    """config.train_transformer / config.test_transformer of the official config, at the recipe's input size.

    Returns (transform, rotation angle or None). myRandomRotation draws its angle once, when it is
    built, so one angle is used for the whole run (official behaviour, kept); build it after seeding.
    """
    u = upstream_utils()
    height, width = recipe["input_size"]
    steps = [u.myNormalize(recipe["normalize"]["datasets"], train=train), u.myToTensor()]
    angle = None
    if train:
        augment = recipe["augment"]
        rotation = u.myRandomRotation(p=augment["rotation"]["p"], degree=list(augment["rotation"]["degrees"]))
        angle = rotation.angle
        steps += [u.myRandomHorizontalFlip(p=augment["hflip"]), u.myRandomVerticalFlip(p=augment["vflip"]), rotation]
    steps.append(u.myResize(height, width))
    return transforms.Compose(steps), angle
