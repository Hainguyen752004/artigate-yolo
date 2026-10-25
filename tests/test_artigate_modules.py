import copy
from pathlib import Path

import pytest
import torch

from artigate.modules import AGA, CBAM, ArtifactHead, register_modules, resize_artifact_map
from artigate.transfer import stock_layer_map

MODELS = Path(__file__).resolve().parents[1] / "configs" / "models"
ARTIGATE_YAMLS = ["yolo26n-seg-p2-cbam-arthead.yaml", "yolo26n-seg-p2-artigate.yaml", "yolo26n-seg-p2-artigate-p2only.yaml"]


def test_aga_starts_as_cbam():
    torch.manual_seed(0)
    feature, logits = torch.randn(2, 64, 20, 20), torch.randn(2, 4, 40, 40) * 5
    aga, cbam = AGA([64, 4]), CBAM(64)
    cbam.channel_attention.load_state_dict(aga.channel_attention.state_dict())
    with torch.no_grad():
        cbam.spatial_attention.conv.weight.copy_(aga.conv.weight[:, :2])
    assert torch.allclose(aga([feature, logits]), cbam(feature), atol=1e-6)


def test_aga_none_mode_is_cbam_and_ignores_the_map():
    aga = AGA([32, 4], mode="none")
    feature = torch.randn(1, 32, 8, 8)
    assert torch.equal(aga([feature, torch.zeros(1, 4, 8, 8)]), aga([feature, torch.ones(1, 4, 8, 8) * 9]))


@pytest.mark.parametrize("beta", [-0.3, 0.0, 0.7])
def test_beta_always_receives_gradient(beta):
    aga = AGA([16, 4], mode="gate")
    with torch.no_grad():
        aga.beta.fill_(beta)
    aga([torch.randn(1, 16, 10, 10), torch.randn(1, 4, 20, 20)]).sum().backward()
    assert aga.beta.grad.abs() > 0


def test_detach_keeps_detection_gradients_out_of_the_artifact_head():
    logits = torch.randn(1, 4, 20, 20, requires_grad=True)
    AGA([16, 4], detach=True)([torch.randn(1, 16, 10, 10), logits]).sum().backward()
    assert logits.grad is None
    aga = AGA([16, 4], detach=False)
    with torch.no_grad():
        aga.beta.fill_(0.5)
    aga([torch.randn(1, 16, 10, 10), logits]).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_resize_keeps_thin_lines():
    a = torch.zeros(1, 1, 64, 64)
    a[..., 31, :] = 1.0
    assert resize_artifact_map(a, (16, 16)).max() == 1.0


def test_artifact_head_single_and_multi_input():
    head = ArtifactHead(16, k=4, hidden=8)
    out = head(torch.randn(2, 16, 40, 40))
    assert out.shape == (2, 4, 40, 40) and head.last_logits is out
    assert torch.sigmoid(out).mean() < 0.05  # prior: artifacts are rare
    assert copy.deepcopy(head).last_logits is None
    multi = ArtifactHead([16, 32], k=4, hidden=8)
    assert multi([torch.randn(1, 16, 40, 40), torch.randn(1, 32, 20, 20)]).shape == (1, 4, 40, 40)


@pytest.mark.parametrize("name", ARTIGATE_YAMLS)
def test_artigate_yamls_build_and_map_stock_layers(name):
    from ultralytics.nn.tasks import SegmentationModel

    register_modules()
    register_modules()  # idempotent
    model = SegmentationModel(str(MODELS / name), ch=3, nc=1, verbose=False).train()
    assert isinstance(model.model[3], ArtifactHead)
    assert [int(s) for s in model.stride] == [4, 8, 16, 32]
    preds = model(torch.zeros(1, 3, 128, 128))
    assert isinstance(preds, dict) and model.model[3].last_logits.shape == (1, 4, 32, 32)
    mapping = stock_layer_map(str(MODELS / name))
    assert (mapping[2], mapping[3], mapping[4], mapping[16]) == (2, 5, 6, 19)
