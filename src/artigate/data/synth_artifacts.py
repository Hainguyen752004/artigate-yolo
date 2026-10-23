"""Procedural dermoscopy artifacts with pixel-exact masks (training augmentation and stress tests).

Four types, matching the ArtifactHead channels:
    0 hair         dark curved strands (cubic Bezier), anywhere, may cross the lesion
    1 ruler        light band with tick marks along one image border, never over the lesion
    2 marker       purple/blue surgical-marker dots, arcs or strokes around the lesion
    3 dark_corner  circular dermatoscope vignette; the lesion always stays inside the aperture

Severity 1 (mild) ... 4 (heavy). Severity 0 means "no artifact".
All functions work on uint8 BGR images (H, W, 3) and draw in place; each returns a bool mask
(H, W) of the pixels it changed, or None if the artifact could not be placed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

TYPES = ("hair", "ruler", "marker", "dark_corner")
_HAIR_COUNT = {1: (2, 6), 2: (6, 15), 3: (15, 35), 4: (35, 70)}
_DARK_R = {1: (0.92, 0.99), 2: (0.82, 0.92), 3: (0.72, 0.82), 4: (0.62, 0.72)}  # x half-diagonal


def _rng(rng=None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _blend_roi(img, alpha, color, x0, y0):
    """img[y0:y0+h, x0:x0+w] = img*(1-a) + color*a, with alpha (h, w) float32 in [0, 1]."""
    h, w = alpha.shape
    roi = img[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
    a = alpha[..., None]
    roi = roi * (1 - a) + np.asarray(color, np.float32) * a
    img[y0 : y0 + h, x0 : x0 + w] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)


def _expand(box, m, W, H):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# --------------------------------------------------------------------------------------- hair
def add_hair(img: np.ndarray, rng=None, severity: int = 2) -> np.ndarray:
    rng = _rng(rng)
    H, W = img.shape[:2]
    s = min(H, W)
    mask = np.zeros((H, W), bool)
    n = int(rng.integers(*_HAIR_COUNT[int(severity)]))
    for _ in range(n):
        length = rng.uniform(0.15, 0.8) * s
        ang = rng.uniform(0, 2 * math.pi)
        p0 = np.array([rng.uniform(-0.1, 1.1) * W, rng.uniform(-0.1, 1.1) * H])
        p3 = p0 + length * np.array([math.cos(ang), math.sin(ang)])
        normal = np.array([-math.sin(ang), math.cos(ang)])
        p1 = p0 + (p3 - p0) * rng.uniform(0.2, 0.45) + normal * rng.normal(0, 0.15 * length)
        p2 = p0 + (p3 - p0) * rng.uniform(0.55, 0.8) + normal * rng.normal(0, 0.15 * length)
        t = np.linspace(0, 1, max(16, int(length / 3)))[:, None]
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        thick = max(1, int(round(rng.uniform(0.002, 0.007) * s)))
        pad = thick + 3
        x0, y0 = int(max(0, pts[:, 0].min() - pad)), int(max(0, pts[:, 1].min() - pad))
        x1, y1 = int(min(W, pts[:, 0].max() + pad + 1)), int(min(H, pts[:, 1].max() + pad + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        layer = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(layer, [np.round(pts - [x0, y0]).astype(np.int32)], False, 255, thick, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), 0.6) * rng.uniform(0.75, 1.0)
        if rng.random() < 0.9:  # dark brown / black
            base = rng.uniform(10, 60)
            color = (base * rng.uniform(0.7, 1.0), base * rng.uniform(0.8, 1.05), base * rng.uniform(0.95, 1.3))
        else:  # light / grey hair
            base = rng.uniform(120, 190)
            color = (base, base * rng.uniform(0.95, 1.05), base * rng.uniform(1.0, 1.1))
        _blend_roi(img, alpha, color, x0, y0)
        mask[y0:y1, x0:x1] |= alpha > 0.35
    return mask


# -------------------------------------------------------------------------------------- ruler
def add_ruler(img: np.ndarray, rng=None, severity: int = 2, avoid_box=None) -> np.ndarray | None:
    rng = _rng(rng)
    H, W = img.shape[:2]
    s = min(H, W)
    avoid = _expand(avoid_box, int(0.03 * s), W, H)
    n_rulers = 2 if severity >= 4 else 1
    width_scale = {1: 0.6, 2: 0.85, 3: 1.1, 4: 1.25}[int(severity)]
    mask = np.zeros((H, W), bool)
    sides = list(rng.permutation(["top", "bottom", "left", "right"]))
    placed = 0
    for side in sides:
        if placed >= n_rulers:
            break
        bw = max(6, int(rng.uniform(0.04, 0.08) * s * width_scale))
        off = int(rng.uniform(0.0, 0.04) * s)
        horiz = side in ("top", "bottom")
        L = W if horiz else H
        frac = rng.uniform(0.5, 1.0) if severity < 3 else rng.uniform(0.75, 1.0)
        a0 = int(rng.uniform(0, 1 - frac) * L)
        a1 = int(a0 + frac * L)
        if side == "top":
            box = (a0, off, a1, off + bw)
        elif side == "bottom":
            box = (a0, H - off - bw, a1, H - off)
        elif side == "left":
            box = (off, a0, off + bw, a1)
        else:
            box = (W - off - bw, a0, W - off, a1)
        if avoid is not None and _overlap(box, avoid):
            continue
        x0, y0, x1, y1 = box
        h, w = y1 - y0, x1 - x0
        # draw a horizontal ruler in a local canvas, rotate if needed
        lw, lh = (w, h) if horiz else (h, w)
        light = rng.random() < 0.85
        body = rng.uniform(215, 255) if light else rng.uniform(20, 60)
        tick_c = rng.uniform(0, 40) if light else rng.uniform(200, 255)
        ticks = np.zeros((lh, lw), np.uint8)  # OpenCV text/lines on uint8, converted below
        step = max(3.0, rng.uniform(0.012, 0.025) * s)
        flip = rng.random() < 0.5
        t_th = 1 if s < 900 else 2
        for i, x in enumerate(np.arange(rng.uniform(0, step), lw, step)):
            frac_len = 0.8 if i % 10 == 0 else 0.55 if i % 5 == 0 else 0.3
            tl = int(frac_len * lh)
            ya, yb = (lh - tl, lh - 1) if flip else (0, tl)
            cv2.line(ticks, (int(x), ya), (int(x), yb), 255, t_th)
            if i % 10 == 0 and lh >= 18 and rng.random() < 0.7:
                yt = int(lh * (0.25 if flip else 0.95))
                cv2.putText(ticks, str(i // 10), (int(x) + 2, yt), cv2.FONT_HERSHEY_SIMPLEX, lh / 60.0, 255, 1, cv2.LINE_AA)
        tk = ticks.astype(np.float32) / 255.0
        alpha = np.maximum(rng.uniform(0.55, 0.9), tk)
        shade = body * (1 - tk) + tick_c * tk
        col = np.stack([shade, shade, shade * rng.uniform(0.95, 1.0)], -1)
        k = {"top": 0, "left": 1, "bottom": 2, "right": -1}[side]  # local top edge -> image border
        alpha, col = np.ascontiguousarray(np.rot90(alpha, k)), np.ascontiguousarray(np.rot90(col, k))
        roi = img[y0:y1, x0:x1].astype(np.float32)
        a = alpha[..., None]
        img[y0:y1, x0:x1] = np.clip(roi * (1 - a) + col * a + 0.5, 0, 255).astype(np.uint8)
        mask[y0:y1, x0:x1] = True
        placed += 1
    return mask if placed else None


# ------------------------------------------------------------------------------------- marker
def add_marker(img: np.ndarray, rng=None, severity: int = 2, avoid_box=None) -> np.ndarray | None:
    rng = _rng(rng)
    H, W = img.shape[:2]
    s = min(H, W)
    if avoid_box is None:
        avoid_box = (W * 0.35, H * 0.35, W * 0.65, H * 0.65)
    bx1, by1, bx2, by2 = avoid_box
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    gap = 0.04 * s
    ax = (bx2 - bx1) / 2 * rng.uniform(1.2, 1.5) + gap
    ay = (by2 - by1) / 2 * rng.uniform(1.2, 1.5) + gap
    r = rng.random()
    color = (
        (rng.uniform(120, 190), rng.uniform(30, 80), rng.uniform(90, 160))  # purple (gentian violet)
        if r < 0.75
        else (rng.uniform(120, 200), rng.uniform(60, 110), rng.uniform(10, 50))  # blue
        if r < 0.9
        else (rng.uniform(20, 50),) * 3  # black
    )
    th = max(2, int(rng.uniform(0.008, 0.02) * s))
    layer = np.zeros((H, W), np.uint8)
    style = rng.choice(["dots", "arc", "strokes"], p=[0.45, 0.35, 0.2])
    sev = int(severity)
    if style == "dots":
        n = int(rng.integers(2 + 2 * sev, 4 + 3 * sev))
        for a in rng.uniform(0, 2 * math.pi, n):
            p = (int(cx + ax * math.cos(a)), int(cy + ay * math.sin(a)))
            cv2.circle(layer, p, max(2, int(th * rng.uniform(0.8, 1.6))), 255, -1, cv2.LINE_AA)
    elif style == "arc":
        span = {1: 60, 2: 120, 3: 220, 4: 340}[sev] + rng.uniform(-20, 20)
        start = rng.uniform(0, 360)
        cv2.ellipse(layer, (int(cx), int(cy)), (int(ax), int(ay)), 0, start, start + span, 255, th, cv2.LINE_AA)
    else:
        for _ in range(int(rng.integers(sev, 2 * sev + 2))):
            a = rng.uniform(0, 2 * math.pi)
            p = np.array([cx + ax * math.cos(a), cy + ay * math.sin(a)])
            d = rng.uniform(0.04, 0.12) * s * np.array([math.cos(a + rng.normal(0, 0.6)), math.sin(a + rng.normal(0, 0.6))])
            cv2.line(layer, tuple(p.astype(int)), tuple((p + d).astype(int)), 255, th, cv2.LINE_AA)
    clear = _expand(tuple(int(v) for v in avoid_box), int(0.01 * s), W, H)  # never paint on the lesion
    layer[clear[1] : clear[3], clear[0] : clear[2]] = 0
    if layer.max() == 0:
        return None
    alpha = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), rng.uniform(0.8, 2.0)) * rng.uniform(0.5, 0.9)
    alpha[clear[1] : clear[3], clear[0] : clear[2]] = 0  # blur must not bleed onto the lesion either
    _blend_roi(img, alpha, color, 0, 0)
    return alpha > 0.2


# -------------------------------------------------------------------------------- dark corner
def add_dark_corner(img: np.ndarray, rng=None, severity: int = 2, keep_box=None) -> np.ndarray | None:
    rng = _rng(rng)
    H, W = img.shape[:2]
    s = min(H, W)
    cx, cy = W / 2 + rng.normal(0, 0.03 * W), H / 2 + rng.normal(0, 0.03 * H)
    half_diag = math.hypot(W / 2, H / 2)
    r = rng.uniform(*_DARK_R[int(severity)]) * half_diag
    if keep_box is not None:
        x1, y1, x2, y2 = keep_box
        need = max(math.hypot(x - cx, y - cy) for x in (x1, x2) for y in (y1, y2)) + 0.03 * s
        r = max(r, need)
    far = max(math.hypot(x - cx, y - cy) for x in (0, W) for y in (0, H))
    if r >= far * 0.98:
        return None
    soft = rng.uniform(0.01, 0.05) * s
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip((d - r) / soft, 0, 1)
    t = t * t * (3 - 2 * t)  # smoothstep
    level = rng.uniform(0, 25)
    noise = rng.normal(0, 3, (H, W)).astype(np.float32)
    out = img.astype(np.float32) * (1 - t[..., None]) + (level + noise)[..., None] * t[..., None]
    img[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return t > 0.5


# ------------------------------------------------------------------------------------ compose
def composite(
    img: np.ndarray,
    rng=None,
    lesion_box=None,
    probs=(0.5, 0.3, 0.3, 0.3),
    severity=(1, 4),
    force: str | None = None,
):
    """Add a random set of artifacts to `img` (in place).

    Args:
        img: uint8 BGR image.
        lesion_box: (x1, y1, x2, y2) in pixels; rulers/markers avoid it, dark corners keep it visible.
        probs: probability of each type in TYPES order. At least one type is always applied.
        severity: int, or (lo, hi) inclusive range sampled per type.
        force: apply only this type (used for the severity-sweep test sets).

    Returns:
        masks (H, W, 4) bool core masks (label = present),
        touched (H, W, 4) bool pixels changed by each type (core + anti-aliased edges / soft falloff),
        applied: list of type names actually drawn.
    """
    rng = _rng(rng)
    H, W = img.shape[:2]
    masks = np.zeros((H, W, len(TYPES)), bool)
    touched = np.zeros_like(masks)
    if force is not None:
        chosen = [force]
    else:
        chosen = [t for t, p in zip(TYPES, probs) if rng.random() < p]
        if not chosen:
            chosen = [TYPES[int(rng.choice(len(TYPES), p=np.asarray(probs) / np.sum(probs)))]]

    def sev():
        return int(severity) if np.isscalar(severity) else int(rng.integers(severity[0], severity[1] + 1))

    applied = []
    # physical order: ink on skin -> hair -> ruler on top -> optical vignette last
    for name in ("marker", "hair", "ruler", "dark_corner"):
        if name not in chosen:
            continue
        before = img.copy()
        if name == "marker":
            m = add_marker(img, rng, sev(), lesion_box)
        elif name == "hair":
            m = add_hair(img, rng, sev())
        elif name == "ruler":
            m = add_ruler(img, rng, sev(), lesion_box)
        else:
            m = add_dark_corner(img, rng, sev(), lesion_box)
        if m is not None and m.any():
            k = TYPES.index(name)
            masks[..., k] = m
            touched[..., k] = np.any(img != before, axis=2) | m
            applied.append(name)
    return masks, touched, applied
