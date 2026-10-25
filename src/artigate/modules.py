"""Custom Ultralytics modules: the ICEEIS P2/CBAM parts and the ArtiGate artifact head and AGA."""
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Proto26
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment26

ARTIFACT_TYPES = ("hair", "ruler", "marker", "dark_corner")


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


def resize_artifact_map(a, size):
    """Resize an artifact probability map; downsampling uses max pooling so thin hair survives."""
    size = tuple(int(v) for v in size)
    if tuple(a.shape[-2:]) == size:
        return a
    if a.shape[-2] >= size[0] and a.shape[-1] >= size[1]:
        return F.adaptive_max_pool2d(a, size)
    return F.interpolate(a, size=size, mode="bilinear", align_corners=False)


class ArtifactHead(nn.Module):
    """Predict K artifact maps (hair, ruler, marker/ink, dark corner) from early backbone features.

    YAML: ``[2, 1, ArtifactHead, [4, 32]]`` (input B2) or ``[[2, 5], 1, ArtifactHead, [4, 32]]``.
    Each input is reduced by a 1x1 conv, resized to the first input and summed, then refined by two
    3x3 convs. Output: logits (B, K, H, W) at the first input's stride. The last output is kept in
    ``last_logits`` so the loss can read it without changing what the layer passes on.
    """

    def __init__(self, c1, k=4, hidden=32, prior=0.01):
        super().__init__()
        channels = list(c1) if isinstance(c1, (list, tuple)) else [c1]
        self.k = k
        self.reduce = nn.ModuleList(Conv(c, hidden, 1) for c in channels)
        self.cv1 = Conv(hidden, hidden, 3)
        self.cv2 = Conv(hidden, hidden, 3)
        self.out = nn.Conv2d(hidden, k, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.constant_(self.out.bias, math.log(prior / (1 - prior)))  # artifacts cover few pixels
        self.last_logits = None

    @staticmethod
    def out_channels(c1, k=4, *args):
        return k

    def forward(self, x):
        xs = list(x) if isinstance(x, (list, tuple)) else [x]
        size = xs[0].shape[-2:]
        feat = sum(F.interpolate(r(f), size=size, mode="nearest") if f.shape[-2:] != size else r(f)
                   for r, f in zip(self.reduce, xs))
        logits = self.out(self.cv2(self.cv1(feat)))
        self.last_logits = logits
        return logits

    def __getstate__(self):  # cached activations must not end up in deepcopies or checkpoints
        state = self.__dict__.copy()
        state["last_logits"] = None
        return state


class AGA(nn.Module):
    """Artifact-Gated Attention: CBAM whose spatial attention also sees the artifact map.

        F'  = ChannelAttention(F)
        s   = sigmoid(conv7x7([mean_c F'; max_c F'; A]))
        F'' = F' * s * max(0, 1 - beta * A),   A = max_k sigmoid(artifact logits)

    Channel attention is the same class CBAM uses. The A-channel weights start at zero and beta at
    0, so AGA is exactly CBAM at initialisation. beta is unconstrained (a clamped beta that drifts
    below 0 stops receiving gradient); weight decay pulls it back towards plain CBAM.

    YAML: ``[[2, 3], 1, AGA, [7, "full", True]]`` = (feature layer, artifact-head layer),
    args = kernel, mode, detach. Modes: "full" (map in spatial attention + gate), "sa" (map only),
    "gate" (gate only), "none" (plain CBAM, map ignored). ``detach`` keeps detection gradients out
    of the artifact head, which is then trained by its own loss only.
    """

    MODES = ("full", "sa", "gate", "none")

    def __init__(self, c1, kernel_size=7, mode="full", detach=True, reduction=16):
        super().__init__()
        channels, _artifact_channels = c1
        if mode not in self.MODES:
            raise ValueError(f"AGA mode must be one of {self.MODES}, got {mode!r}")
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        if channels < reduction:
            raise ValueError(f"AGA channels ({channels}) must be at least reduction ({reduction})")
        self.mode, self.detach = mode, bool(detach)
        self.use_map, self.use_gate = mode in ("full", "sa"), mode in ("full", "gate")
        self.channel_attention = ChannelAttention(channels, reduction)
        self.conv = nn.Conv2d(3 if self.use_map else 2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        if self.use_map:
            with torch.no_grad():
                self.conv.weight[:, 2].zero_()
        self.beta = nn.Parameter(torch.zeros(1))

    @staticmethod
    def out_channels(c1, *args):
        return c1[0]

    def artifact_map(self, logits, size):
        logits = logits.detach() if self.detach else logits
        return resize_artifact_map(torch.sigmoid(logits.float()).amax(dim=1, keepdim=True), size)

    def forward(self, x):
        feature, logits = x
        feature = self.channel_attention(feature)
        pooled = [torch.mean(feature, dim=1, keepdim=True), torch.amax(feature, dim=1, keepdim=True)]
        a = None
        if self.use_map or self.use_gate:
            a = self.artifact_map(logits, feature.shape[-2:]).to(feature.dtype)
        if self.use_map:
            pooled.append(a)
        out = torch.sigmoid(self.conv(torch.cat(pooled, dim=1))) * feature
        if self.use_gate:
            out = out * (1 - self.beta.to(feature.dtype) * a).clamp(min=0.0)
        return out


CUSTOM_MODULES = {"CBAM": CBAM, "Segment26": P2CompatibleSegment26, "ArtifactHead": ArtifactHead, "AGA": AGA}
# Modules whose input channels parse_model must work out itself (they may take several inputs).
AUTO_CHANNEL_MODULES = (ArtifactHead, AGA)
_PARSE_ANCHOR = "        else:\n            c2 = ch[f]\n"
_PARSE_BRANCH = (
    "        elif m in _ARTIGATE_AUTO_CHANNELS:\n"
    "            _c1 = [ch[x] for x in f] if isinstance(f, (list, tuple)) else ch[f]\n"
    "            args = [_c1, *args]\n"
    "            c2 = m.out_channels(*args)\n"
)


def _patch_parse_model(task_modules):
    """Teach parse_model the input channels of ArtifactHead and AGA (several inputs allowed).

    Stock parse_model ends with ``else: c2 = ch[f]``, which fails for list inputs. One branch is
    inserted before it, in memory only; the source is checked against Ultralytics 8.4.13.
    """
    if getattr(task_modules, "_artigate_parse_model", False):
        return
    source = inspect.getsource(task_modules.parse_model)
    if source.count(_PARSE_ANCHOR) != 1:
        raise RuntimeError("parse_model does not have the expected 'else: c2 = ch[f]' line; "
                           "use ultralytics==8.4.13 or update artigate.modules._patch_parse_model")
    task_modules._ARTIGATE_AUTO_CHANNELS = AUTO_CHANNEL_MODULES
    code = compile(source.replace(_PARSE_ANCHOR, _PARSE_BRANCH + _PARSE_ANCHOR), "<artigate parse_model>", "exec")
    exec(code, task_modules.__dict__)
    task_modules._artigate_parse_model = True


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
    _patch_parse_model(task_modules)
