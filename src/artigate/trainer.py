"""Ultralytics trainers used by ArtiGate-YOLO."""
from copy import copy
from pathlib import Path
import random

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from artigate.transfer import load_partial_pretrained, stock_layer_map


class TransferSegmentationTrainer(SegmentationTrainer):
    """SegmentationTrainer that copies stock YOLO26n-seg weights into the model it builds.

    Model.train() rebuilds a YAML-defined model through get_model(weights=None) and drops
    any weights loaded into the YOLO object beforehand (this silently disabled the ICEEIS
    transfer), so the transfer has to happen on the model the trainer creates.
    """

    pretrained = None
    transfer_report = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if weights is None and self.pretrained is not None:
            self.transfer_report = load_partial_pretrained(model, self.pretrained, stock_layer_map(cfg))
        return model


def transfer_trainer(pretrained):
    """Return a trainer class bound to one stock checkpoint; ``None`` trains from scratch."""
    return type(
        "TransferSegmentationTrainer",
        (TransferSegmentationTrainer,),
        {"pretrained": None if pretrained is None else Path(pretrained)},
    )


class ArtiGateTrainer(TransferSegmentationTrainer):
    """TransferSegmentationTrainer plus the artifact head training. Used for every ablation row.

    - builds ArtiGateSegModel (YOLO-seg loss + artifact loss, logged as ``art_loss``; 0 without a head)
    - the training set is an ArtiGateDataset when the dataset YAML has an ``artigate:`` block
    - splits 7-channel batches into the RGB image and ``batch["art"]``
    Without an artifact head and without an ``artigate:`` block it trains exactly like
    TransferSegmentationTrainer, so A0/A1 and the ArtiGate rows share one pipeline.
    """

    art_lambda = None  # overrides art_loss.lambda of the model YAML
    art_overrides = {}  # overrides keys of the dataset YAML's artigate block

    def get_model(self, cfg=None, weights=None, verbose=True):
        from artigate.artifact_loss import ArtiGateSegModel

        model = ArtiGateSegModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if self.art_lambda is not None:
            model.yaml.setdefault("art_loss", {})["lambda"] = float(self.art_lambda)
        if weights:
            model.load(weights)  # resume / same-architecture checkpoint: every tensor
        elif self.pretrained is not None:
            self.transfer_report = load_partial_pretrained(model, self.pretrained, stock_layer_map(cfg))
        return model

    def art_config(self):
        config = self.data.get("artigate")
        return None if not config else {**config, **self.art_overrides}

    def build_dataset(self, img_path, mode="train", batch=None):
        config = self.art_config()
        if mode != "train" or config is None:
            return super().build_dataset(img_path, mode, batch)
        from artigate.data.artifact_dataset import ArtiGateDataset

        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        # same arguments as ultralytics.data.build.build_yolo_dataset(mode="train"), our dataset class
        return ArtiGateDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=True, hyp=self.args,
            rect=self.args.rect, cache=self.args.cache or None, single_cls=self.args.single_cls or False,
            stride=stride, pad=0.0, prefix=colorstr(f"{mode}: "), task=self.args.task, classes=self.args.classes,
            data=self.data, fraction=self.args.fraction, art_cfg=config,
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        image = batch["img"]
        if image.shape[1] > 3:
            batch["art"] = image[:, 3:].contiguous()
            colour = image[:, :3]
            # Format() converts BGR to RGB only for 3-channel images, so do it here (bgr hyp respected)
            batch["img"] = colour if random.uniform(0, 1) <= self.args.bgr else colour.flip(1)
        return batch

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (*self.loss_names, "art_loss")
        return validator

    def plot_training_samples(self, batch, ni):
        batch = copy(batch)
        batch.pop("art", None)
        super().plot_training_samples(batch, ni)


def artigate_trainer(pretrained, art_lambda=None, art_overrides=None):
    """Return an ArtiGateTrainer class bound to one stock checkpoint and the artifact settings."""
    return type(
        "ArtiGateTrainer",
        (ArtiGateTrainer,),
        {
            "pretrained": None if pretrained is None else Path(pretrained),
            "art_lambda": art_lambda,
            "art_overrides": dict(art_overrides or {}),
        },
    )
