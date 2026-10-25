import torch

from artigate.artifact_loss import artifact_loss, artifact_targets as art_targets


def enc(pos, neg):
    """Encode boolean maps into the 0 / 114 / 255 code used by the dataset, scaled to [0, 1]."""
    x = torch.full(pos.shape, 114 / 255)
    x[neg] = 0.0
    x[pos] = 1.0
    return x


def test_all_unknown_gives_zero_and_no_nan():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    loss = artifact_loss(logits, torch.full((2, 4, 32, 32), 114 / 255))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_unknown_pixels_are_ignored():
    pos = torch.zeros(1, 4, 32, 32, dtype=torch.bool)
    pos[:, 0, :, :8] = True
    neg = torch.zeros_like(pos)
    neg[:, 0, :, 16:] = True  # columns 8-15 unknown
    art = enc(pos, neg)
    a = torch.full((1, 4, 8, 8), -8.0)
    a[:, 0, :, :2] = 8.0  # correct on labelled cells
    b = a.clone()
    b[:, 0, :, 2:4] = 8.0  # differs only on the unknown cells
    assert torch.isclose(artifact_loss(a, art), artifact_loss(b, art))


def test_downsampling_rule():
    art = torch.full((1, 1, 8, 8), 114 / 255)
    art[..., :4, :4] = 0.0  # all negative block
    art[..., 0, 4] = 1.0  # one positive pixel in the top-right block
    t, w = art_targets(art, (2, 2))
    assert t[0, 0].tolist() == [[0, 1], [0, 0]]
    assert w[0, 0].tolist() == [[1, 1], [0, 0]]


def test_good_prediction_has_lower_loss():
    pos = torch.zeros(1, 4, 32, 32, dtype=torch.bool)
    pos[:, 1, :8] = True
    art = enc(pos, ~pos)
    good = torch.where(art_targets(art, (8, 8))[0] > 0, 6.0, -6.0)
    assert artifact_loss(good, art) < artifact_loss(torch.zeros(1, 4, 8, 8), art)


def test_codes_after_mixing_are_read_conservatively():
    # absent, absent+unknown (mix-up ~57), unknown, present+unknown (~184), present
    values = torch.tensor([0, 57, 114, 184, 255], dtype=torch.float32) / 255
    target, weight = art_targets(values.view(1, 1, 1, 5), (1, 5))
    assert target.flatten().tolist() == [0, 0, 0, 0, 1]
    assert weight.flatten().tolist() == [1, 0, 0, 0, 1]


def test_art_loss_without_targets_stays_finite_in_fp16_validation():
    from artigate.artifact_loss import ArtiGateLoss
    from artigate.modules import ArtifactHead

    head = ArtifactHead(16, k=4, hidden=8)
    model = torch.nn.Sequential(head)
    loss = ArtiGateLoss(model, lambda preds, batch: (torch.zeros(3), torch.zeros(3)))
    batch = {"img": torch.zeros(16, 3, 8, 8)}  # validation batch: no "art" key
    # fp16 logits like the validator's half model: their plain sum overflows to -inf
    head.last_logits = torch.full((16, 4, 160, 160), -4.6, dtype=torch.float16)
    assert torch.isinf(head.last_logits.sum())
    with torch.no_grad():
        total, items = loss(None, batch)
    assert torch.isfinite(total).all() and items[-1] == 0
    # training with lambda 0: still finite, head stays in the graph with zero gradient
    logits = torch.full((16, 4, 160, 160), -4.6, dtype=torch.float16, requires_grad=True)
    head.last_logits = logits
    total, items = loss(None, batch)
    total.sum().backward()
    assert torch.isfinite(total).all() and items[-1] == 0 and logits.grad.abs().sum() == 0
