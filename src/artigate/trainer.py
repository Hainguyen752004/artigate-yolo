"""Ultralytics trainers used by ArtiGate-YOLO."""
from pathlib import Path

from ultralytics.models.yolo.segment import SegmentationTrainer

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
