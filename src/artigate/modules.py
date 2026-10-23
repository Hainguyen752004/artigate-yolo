"""Custom Ultralytics modules for the YOLO26n-seg P2 models (ported from ICEEIS)."""
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Proto26
from ultralytics.nn.modules.head import Detect, Segment26


class ChannelAttention(nn.Module):
    """CBAM channel attention (average + max pooling through a shared MLP).

    Global max pooling is computed with ``amax``: same output as ``AdaptiveMaxPool2d(1)``, but its
    backward pass is deterministic on CUDA, so reruns with one seed can match. State dicts are
    unchanged (the pooling layers have no parameters).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=(2, 3), keepdim=True)
        peak = x.amax(dim=(2, 3), keepdim=True)
        return self.sigmoid(self.fc(avg) + self.fc(peak)) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = torch.cat((torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)), dim=1)
        return self.sigmoid(self.conv(pooled)) * x


class CBAM(nn.Module):
    """Apply channel and spatial attention without changing tensor shape."""

    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        super().__init__()
        channels = c1 if c2 is None else c2
        if channels != c1:
            raise ValueError(f"CBAM must preserve channels (received {c1} -> {channels})")
        if channels < reduction:
            raise ValueError(f"CBAM channels ({channels}) must be at least reduction ({reduction})")
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class P2CompatibleSegment26(Segment26):
    """Segment26 that predicts on P2-P5 but builds mask prototypes from P3-P5.

    Proto26 upsamples its first input by 2, so feeding it P3 gives stride-4 prototypes,
    which is what the segmentation validator expects. With three inputs it behaves like
    the stock Segment26.
    """

    def __init__(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        proto_channels = ch[1:] if len(ch) >= 4 else ch
        self.proto = Proto26(proto_channels, self.npr, self.nm, nc)

    def forward(self, x):
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto_input = x[1:] if len(x) >= 4 else x
        proto = self.proto(proto_input)
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


CUSTOM_MODULES = {"CBAM": CBAM, "Segment26": P2CompatibleSegment26}


def register_modules():
    """Expose the custom modules to the Ultralytics YAML parser.

    parse_model resolves module names from ultralytics.nn.tasks globals and special-cases
    Segment26 by identity, so the P2 head has to replace Segment26 rather than use a new name.
    Call this before building a model from YAML or loading a checkpoint trained with it.
    """
    import ultralytics.nn.modules as block_modules
    import ultralytics.nn.tasks as task_modules

    for name, module in CUSTOM_MODULES.items():
        setattr(block_modules, name, module)
        setattr(task_modules, name, module)
