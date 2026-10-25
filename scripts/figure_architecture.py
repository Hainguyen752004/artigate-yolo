"""Paper Fig. 1: ArtiGate-YOLO architecture (overview, AGA block, artifact head and training targets).

Writes paper/figures/architecture.svg and, when Playwright with Chromium is available, a vector
paper/figures/architecture.pdf (fonts embedded) and a PNG preview. The visual language follows the
ICEEIS architecture figure: feature maps as blue cuboids, operation blocks with a dark offset shadow,
light panels with coloured titles. Orange marks what ArtiGate-YOLO adds.

The thumbnails are an illustration on a drawn lesion with procedurally drawn artifacts (no dataset
image, no model output). Regenerate them from an ISIC 2018 training image with --image/--mask.

    python scripts/figure_architecture.py
    python scripts/figure_architecture.py --image <ISIC_xxx.jpg> --mask <ISIC_xxx_segmentation.png>
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from artigate.data.synth_artifacts import TYPES, composite  # noqa: E402

# ------------------------------------------------------------------------------------ palette
INK = "#1D2738"        # outlines, arrows, text
MUTED = "#5A6578"
SHADOW = "#2A3345"
PANEL = "#F4F7FB"
PANEL_EDGE = "#C9D3E1"
CONV = "#5EC4CB"       # convolution
POOL = "#F4C653"       # pooling, sigmoid, max
HEAD = "#C9B8F0"       # Segment26 heads
NECK = "#E3E8F0"
WHITE = "#FFFFFF"
NEW = "#F39A3D"        # added by ArtiGate-YOLO
NEW_DARK = "#C4620E"
NEW_SOFT = "#FDE3C6"
GROUP = "#E06262"      # dashed grouping
FM_FRONT, FM_TOP, FM_SIDE, FM_EDGE = "#5B8ED8", "#9DBEEE", "#3F6DB5", "#1F3B66"
TITLE_A = "#1F4E8C"
FONT = "InterTTF, Inter, 'Inter Display', 'DejaVu Sans', Arial, sans-serif"
MATH = "'DejaVu Sans', Inter, sans-serif"


_FONT_FILES = {500: "Inter-Medium.otf", 600: "Inter-SemiBold.otf", 700: "Inter-Bold.otf", 800: "Inter-ExtraBold.otf"}


def measure(text, size, weight=500):
    """Rendered width of `text` in Inter; a rough estimate if the font file is not installed."""
    try:
        from PIL import ImageFont

        for folder in ("/usr/share/fonts/opentype/inter", "C:/Windows/Fonts", str(Path.home() / ".fonts")):
            path = Path(folder) / _FONT_FILES.get(weight, "Inter-Medium.otf")
            if path.exists():
                return ImageFont.truetype(str(path), size).getlength(text)
    except Exception:
        pass
    return len(text) * size * 0.58


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=22, weight=500, fill=INK, anchor="middle", italic=False, family=FONT, spacing=None):
        style = ' font-style="italic"' if italic else ""
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}"'
                 f' fill="{fill}" text-anchor="{anchor}"{style}{ls}>{s}</text>')

    def rect(self, x, y, w, h, fill, stroke=INK, sw=2, rx=6, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{o}'
                 f' stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def block(self, x, y, w, h, fill, label, sub=None, label_size=22, sub_size=19, label_fill=INK):
        """Operation block with a dark offset shadow (the ICEEIS figure style)."""
        self.add(f'<rect x="{x + 4:.1f}" y="{y + 4:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" fill="{SHADOW}"/>')
        self.rect(x, y, w, h, fill, rx=5)
        if sub:
            self.text(x + w / 2, y + h / 2 - 2, label, label_size, 700, label_fill)
            self.text(x + w / 2, y + h / 2 + sub_size + 1, sub, sub_size, 500, label_fill)
        else:
            self.text(x + w / 2, y + h / 2 + label_size * 0.36, label, label_size, 700, label_fill)

    def cuboid(self, x, y, w, h, d, front=FM_FRONT, top=FM_TOP, side=FM_SIDE, edge=FM_EDGE):
        """Feature map: front face at (x, y, w, h), receding by d up and to the right."""
        k = 0.75
        dx, dy = d, -d * k
        self.add(f'<polygon points="{x+w+4},{y+4} {x+w+dx+4},{y+dy+4} {x+w+dx+4},{y+h+dy+4} {x+w+4},{y+h+4}" fill="{SHADOW}"/>')
        self.add(f'<rect x="{x + 4}" y="{y + 4}" width="{w}" height="{h}" fill="{SHADOW}"/>')
        self.add(f'<polygon points="{x},{y} {x+dx},{y+dy} {x+w+dx},{y+dy} {x+w},{y}" fill="{top}" stroke="{edge}" stroke-width="2" stroke-linejoin="round"/>')
        self.add(f'<polygon points="{x+w},{y} {x+w+dx},{y+dy} {x+w+dx},{y+h+dy} {x+w},{y+h}" fill="{side}" stroke="{edge}" stroke-width="2" stroke-linejoin="round"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{front}" stroke="{edge}" stroke-width="2"/>')

    def line(self, pts, color=INK, sw=2.4, dash=None, arrow=True, marker=None):
        d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker or ("arrow-new" if color == NEW_DARK else "arrow")})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{da}{m}/>')

    def op(self, x, y, sym, r=15):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{WHITE}" stroke="{GROUP}" stroke-width="2.2"/>')
        if sym == "x":
            a = r * 0.42
            self.add(f'<path d="M{x-a},{y-a} L{x+a},{y+a} M{x+a},{y-a} L{x-a},{y+a}" stroke="{GROUP}" stroke-width="2.4" stroke-linecap="round"/>')
        else:
            a = r * 0.5
            self.add(f'<path d="M{x-a},{y} L{x+a},{y} M{x},{y-a} L{x},{y+a}" stroke="{GROUP}" stroke-width="2.4" stroke-linecap="round"/>')

    def dot(self, x, y, color=INK, r=4.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def image(self, x, y, w, h, png_b64, stroke=INK, sw=2):
        self.add(f'<rect x="{x + 4}" y="{y + 4}" width="{w}" height="{h}" fill="{SHADOW}"/>')
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" href="data:image/png;base64,{png_b64}"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>')

    def panel(self, x, y, w, h, title, subtitle, color):
        self.rect(x, y, w, h, PANEL, PANEL_EDGE, 2, rx=18)
        self.text(x + 24, y + 44, title, 34, 800, color, anchor="start")
        self.text(x + 24 + measure(title, 34, 800) + 18, y + 44, subtitle, 23, 500, color, anchor="start", italic=True)

    def svg(self):
        defs = f"""<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L9.5,5 L0,9.2 z" fill="{INK}"/></marker>
<marker id="arrow-new" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L9.5,5 L0,9.2 z" fill="{NEW_DARK}"/></marker>
<marker id="arrow-blue" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L9.5,5 L0,9.2 z" fill="{TITLE_A}"/></marker>
<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#9AA4B4"/><line x1="0" y1="0" x2="0" y2="8" stroke="#C9D0DB" stroke-width="4"/></pattern>
</defs>"""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>{defs}' + "".join(self.parts) + "</svg>")


# ------------------------------------------------------------------------------- thumbnails
def drawn_lesion(size=320, seed=0):
    from figure_synthetic_artifacts import drawn_lesion as dl

    return dl(size, seed)


def load_pair(image, mask, size):
    from figure_synthetic_artifacts import load_pair as lp

    return lp(image, mask, size)


def b64png(img_rgb_or_gray):
    ok, buf = cv2.imencode(".png", img_rgb_or_gray)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def make_thumbnails(base, gt, seed=11):
    img = base.copy()
    ys, xs = np.nonzero(gt)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    rng = np.random.default_rng(seed)
    masks = {}
    for name, sev in (("marker", 3), ("hair", 2), ("ruler", 2), ("dark_corner", 3)):
        m, _, applied = composite(img, rng, lesion_box=box, severity=sev, force=name)
        masks[name] = m[..., TYPES.index(name)]
    abar = np.maximum.reduce([masks[t] for t in TYPES]).astype(np.uint8)
    thumbs = {"input": b64png(img)}
    navy = np.array([68, 39, 29], np.uint8)  # BGR of a deep navy background for mask tiles
    kernel = np.ones((3, 3), np.uint8)
    for t in TYPES:
        tile = np.zeros(img.shape, np.uint8) + navy
        tile[cv2.dilate(masks[t].astype(np.uint8), kernel) > 0] = (255, 255, 255)
        thumbs[t] = b64png(tile)
    tile = np.zeros(img.shape, np.uint8) + navy
    tile[abar.astype(bool)] = (61, 154, 243)  # orange (BGR)
    thumbs["abar"] = b64png(tile)
    # outputs: lesion contour and box on the image, artifact map overlay
    out = img.copy()
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (110, 60, 20), 3, cv2.LINE_AA)
    x1, y1, x2, y2 = box
    for i in range(x1, x2, 14):
        cv2.line(out, (i, y1), (min(i + 7, x2), y1), (60, 60, 200), 2)
        cv2.line(out, (i, y2), (min(i + 7, x2), y2), (60, 60, 200), 2)
    for j in range(y1, y2, 14):
        cv2.line(out, (x1, j), (x1, min(j + 7, y2)), (60, 60, 200), 2)
        cv2.line(out, (x2, j), (x2, min(j + 7, y2)), (60, 60, 200), 2)
    thumbs["mask_out"] = b64png(out)
    over = img.copy().astype(np.float32)
    a = abar.astype(bool)
    over[a] = over[a] * 0.35 + np.array([61, 154, 243], np.float32) * 0.65
    thumbs["abar_out"] = b64png(np.clip(over, 0, 255).astype(np.uint8))
    return thumbs


# ----------------------------------------------------------------------------------- figure
def build(thumbs):
    W, H = 1800, 1330
    s = SVG(W, H)
    MK = "<tspan font-size='14' dy='5'>k</tspan>"

    # =========================================================== (a) overview
    s.panel(20, 18, 1760, 520, "ArtiGate-YOLO", "YOLO26n-seg with a P2 level · AGA replaces CBAM after B2 and B3", TITLE_A)
    yc = 262
    s.text(140, 124, "Dermoscopic input", 22, 700)
    s.image(56, 136, 168, 168, thumbs["input"])
    s.text(140, 330, "640 × 640 × 3", 20, 500, MUTED)
    s.text(140, 354, "hair · ruler · ink · corner", 18, 500, MUTED, italic=True)
    s.line([(228, yc), (262, yc)])
    cub = {}
    for name, x, f, d, dims in (("B2", 266, 108, 22, "64 × 160²"), ("B3", 548, 88, 28, "128 × 80²"),
                                ("B4", 812, 70, 28, "128 × 40²"), ("B5", 944, 54, 36, "256 × 20²")):
        s.cuboid(x, yc - f / 2, f, f, d)
        cub[name] = (x, f, d)
        s.text(x + f / 2 + d / 2, yc + 84, name, 22, 800)
        s.text(x + f / 2 + d / 2, yc + 108, dims, 20, 500, MUTED)
    aga = {"A2": 430, "A3": 690}
    for x in aga.values():
        s.block(x, yc - 32, 88, 64, NEW, "AGA")
    right = {k: x + f + d * 0.6 for k, (x, f, d) in cub.items()}
    s.line([(right["B2"], yc), (426, yc)])
    s.dot(406, yc)
    s.line([(522, yc), (546, yc)])
    s.line([(right["B3"], yc), (686, yc)])
    s.line([(782, yc), (810, yc)])
    s.line([(right["B4"], yc), (942, yc)])

    # neck and heads
    rows = {"N2": 150, "N3": 218, "N4": 286, "N5": 354}
    s.rect(1094, 112, 116, 290, WHITE, PANEL_EDGE, 2, rx=12, dash="7 6")
    s.text(1152, 100, "PAN neck", 21, 700, MUTED)
    for k, y in rows.items():
        s.block(1108, y - 23, 88, 46, NECK, k, label_size=21)
    a2x, a3x = aga["A2"] + 44, aga["A3"] + 44
    b4x, b4f, b4d = cub["B4"]
    b4c = b4x + b4f / 2
    s.line([(a2x, yc - 32), (a2x, 120), (1078, 120), (1078, rows["N2"]), (1104, rows["N2"])])
    s.line([(a3x, yc - 32), (a3x, 136), (1066, 136), (1066, rows["N3"]), (1104, rows["N3"])])
    s.line([(b4c, yc - b4f / 2 - b4d * 0.75), (b4c, 152), (1054, 152), (1054, rows["N4"]), (1104, rows["N4"])])
    s.line([(1036, yc), (1042, yc), (1042, rows["N5"]), (1104, rows["N5"])])
    scales = {"N2": "P2 · 1/4", "N3": "P3 · 1/8", "N4": "P4 · 1/16", "N5": "P5 · 1/32"}
    for k, y in rows.items():
        s.line([(1200, y), (1242, y)])
        s.block(1246, y - 25, 132, 50, HEAD, "Segment", scales[k], 19, 17)
        s.line([(1382, y), (1404, y)], arrow=False)
    s.line([(1404, rows["N2"]), (1404, rows["N5"])], arrow=False)
    s.line([(1404, 220), (1434, 220)])
    for i, t in enumerate(("boxes, classes and", "mask coefficients;", "prototypes from N3–N5")):
        s.text(1312, 404 + i * 21, t, 18, 500, MUTED, italic=True)
    s.text(1520, 126, "Lesion mask", 22, 700)
    s.image(1438, 138, 164, 164, thumbs["mask_out"])
    s.text(1520, 326, "top-1 instance", 19, 500, MUTED, italic=True)

    # artifact branch
    hb = (318, 414, 176, 62)
    s.line([(406, yc), (406, hb[1] - 4)])
    s.block(*hb, NEW, "Artifact head", "4 maps at stride 4", 21, 17)
    names = [("hair", "hair"), ("ruler", "ruler"), ("marker", "ink"), ("dark_corner", "corner")]
    tx = 530
    s.line([(hb[0] + hb[2] + 4, hb[1] + 31), (tx - 4, hb[1] + 31)])
    for i, (key, lab) in enumerate(names):
        x = tx + i * 72
        s.image(x, hb[1] + 1, 60, 60, thumbs[key], sw=1.6)
        s.text(x + 30, hb[1] + 88, lab, 19, 600, MUTED)
    mx = tx + 4 * 72 + 8
    s.line([(mx - 14, hb[1] + 31), (mx - 2, hb[1] + 31)])
    s.block(mx, hb[1] + 11, 58, 40, POOL, "max" + MK, label_size=19)
    abx = mx + 82
    s.line([(mx + 62, hb[1] + 31), (abx - 4, hb[1] + 31)])
    s.image(abx, hb[1] + 1, 60, 60, thumbs["abar"], stroke=NEW_DARK, sw=3)
    chy = 394
    s.line([(abx + 30, hb[1] - 2), (abx + 30, chy), (a2x, chy), (a2x, yc + 36)], color=NEW_DARK, dash="9 6", sw=2.6)
    s.line([(a3x, chy), (a3x, yc + 36)], color=NEW_DARK, dash="9 6", sw=2.6)
    s.dot(a3x, chy, NEW_DARK)
    lx = abx + 78
    s.text(lx, hb[1] + 22, "artifact map Ā", 20, 700, NEW_DARK, anchor="start")
    s.add(f'<text x="{lx}" y="{hb[1] + 46}" font-family="{FONT}" font-size="18" font-weight="500" fill="{NEW_DARK}" font-style="italic">detached, trained only</text>')
    s.add(f'<text x="{lx}" y="{hb[1] + 68}" font-family="{FONT}" font-size="18" font-weight="500" fill="{NEW_DARK}" font-style="italic">by ℒ<tspan font-size="12" dy="4">art</tspan><tspan dy="-4"> (panel below)</tspan></text>')
    s.line([(abx + 30, hb[1] + 66), (abx + 30, 512), (1520, 512), (1520, 494)], color=NEW_DARK, dash="9 6", sw=2.6)
    s.image(1438, 354, 164, 134, thumbs["abar_out"])
    s.text(1616, 410, "Artifact maps", 20, 700, NEW_DARK, anchor="start")
    s.text(1616, 434, "by-product at", 18, 500, NEW_DARK, anchor="start", italic=True)
    s.text(1616, 456, "inference", 18, 500, NEW_DARK, anchor="start", italic=True)

    # =========================================================== (b) AGA
    s.panel(20, 554, 1760, 372, "AGA", "artifact-gated attention · drop-in replacement for CBAM", NEW_DARK)
    yc = 716
    s.cuboid(56, yc - 60, 50, 120, 14)
    s.text(81, yc + 9, "F", 26, 800, WHITE)
    s.rect(156, 624, 434, 182, "none", GROUP, 2, rx=10, dash="7 5")
    s.text(373, 830, "channel attention (as in CBAM)", 19, 600, GROUP)
    s.block(176, 646, 128, 50, POOL, "AvgPool", "over H×W", 20, 17)
    s.block(176, 742, 128, 50, POOL, "MaxPool", "over H×W", 20, 17)
    # one MLP applied to both pooled vectors (CBAM shares its weights): drawn twice, boxed as shared
    s.rect(320, 637, 148, 164, "none", GROUP, 1.8, rx=8, dash="4 4")
    s.block(330, 646, 126, 50, CONV, "MLP", "C → C/16 → C", 20, 17)
    s.block(330, 742, 126, 50, CONV, "MLP", "C → C/16 → C", 20, 17)
    s.text(394, 725, "shared weights", 17, 700, GROUP)
    s.line([(110, yc), (136, yc), (136, 671), (172, 671)])
    s.line([(136, yc), (136, 767), (172, 767)])
    s.dot(136, yc)
    s.line([(308, 671), (326, 671)])
    s.line([(308, 767), (326, 767)])
    s.op(494, yc, "+")
    s.line([(460, 671), (494, 671), (494, yc - 18)])
    s.line([(460, 767), (494, 767), (494, yc + 18)])
    s.line([(511, yc), (519, yc)])
    s.block(522, yc - 20, 40, 40, POOL, "σ", label_size=22)
    s.op(616, yc, "x")
    s.line([(566, yc), (598, yc)])
    s.line([(81, yc + 66), (81, 856), (616, 856), (616, yc + 18)])
    s.text(96, 876, "C × H × W", 18, 500, MUTED, anchor="start")
    s.line([(633, yc), (654, yc)])
    s.cuboid(658, yc - 60, 50, 120, 14)
    s.text(683, yc + 9, "F′", 26, 800, WHITE)
    s.rect(748, 624, 538, 182, "none", GROUP, 2, rx=10, dash="7 5")
    s.text(1017, 830, "spatial attention with the artifact map as a third channel", 19, 600, GROUP)
    s.block(768, 636, 150, 42, POOL, "Mean", "over C", 19, 16)
    s.block(768, 690, 150, 42, POOL, "Max", "over C", 19, 16)
    s.block(768, 744, 150, 46, NEW, "Ā′", "max-pool resize", 19, 15)
    s.line([(726, yc), (744, yc), (744, 657), (764, 657)])
    s.line([(744, yc), (744, 711), (764, 711)])
    s.dot(744, yc)
    s.block(946, yc - 26, 92, 52, WHITE, "Concat", "3 channels", 19, 15)
    s.line([(922, 657), (932, 657), (932, yc - 14), (942, yc - 14)])
    s.line([(922, 711), (930, 711), (930, yc), (942, yc)])
    s.line([(922, 758), (936, 758), (936, yc + 14), (942, yc + 14)], color=NEW_DARK)
    s.block(1062, yc - 26, 112, 52, CONV, "Conv 7×7", "3 → 1", 20, 16)
    s.line([(1042, yc), (1058, yc)])
    s.block(1196, yc - 20, 40, 40, POOL, "σ", label_size=22)
    s.line([(1178, yc), (1192, yc)])
    s.op(1316, yc, "x")
    s.line([(1240, yc), (1298, yc)])
    s.line([(683, yc - 60 - 11), (683, 613), (1316, 613), (1316, yc - 18)])
    s.op(1404, yc, "x")
    s.line([(1333, yc), (1386, yc)])
    s.block(1320, 782, 168, 46, NEW, "max(0, 1 − βĀ′)", label_size=18)
    s.line([(1404, 778), (1404, yc + 20)], color=NEW_DARK)
    s.line([(922, 782), (1300, 782), (1300, 805), (1316, 805)], color=NEW_DARK, dash="9 6", sw=2.4)
    s.line([(1421, yc), (1446, yc)])
    s.cuboid(1450, yc - 60, 50, 120, 14)
    s.text(1475, yc + 9, "F″", 26, 800, WHITE)
    s.rect(1534, 640, 230, 152, NEW_SOFT, NEW_DARK, 2, rx=10)
    for i, (t, w) in enumerate((("At initialization", 800), ("Ā′ weights = 0, β = 0", 500), ("→ AGA ≡ CBAM", 700),
                                ("β learned, unclamped;", 500), ("weight decay → 0", 500))):
        s.text(1550, 672 + i * 26 - (0 if i == 0 else -2), t, 20 if i == 0 else 18, w, NEW_DARK if i in (0, 2) else INK, anchor="start")
    sub = lambda t: f'<tspan font-size="14" dy="5">{t}</tspan><tspan dy="-5">'
    s.add(f'<text x="176" y="904" font-family="{MATH}" font-size="21" fill="{INK}">M{sub("c")} = σ(MLP(Avg</tspan>{sub("HW")} F) + MLP(Max</tspan>{sub("HW")} F)),  F′ = M</tspan>{sub("c")} ⊙ F</tspan></text>')
    s.add(f'<text x="768" y="904" font-family="{MATH}" font-size="21" fill="{INK}">M{sub("s")} = σ(Conv</tspan>{sub("7×7")}[Mean</tspan>{sub("C")} F′; Max</tspan>{sub("C")} F′; </tspan><tspan fill="{NEW_DARK}">Ā′</tspan>]),  F″ = M{sub("s")} ⊙ F′ ⊙ </tspan><tspan fill="{NEW_DARK}">max(0, 1 − βĀ′)</tspan></text>')

    # =========================================================== (c) artifact head and targets
    s.panel(20, 942, 1760, 300, "Artifact head", "trained without manual masks: synthetic artifacts and partial labels", NEW_DARK)
    yc = 1086
    s.cuboid(52, yc - 42, 84, 84, 18)
    s.text(103, yc + 70, "B2 · 64 × 160²", 19, 700)
    x = 176
    s.line([(154, yc), (172, yc)])
    for i, (lab, sb) in enumerate((("Conv 1×1", "64 → 32"), ("Conv 3×3", "32 → 32"), ("Conv 3×3", "32 → 32"), ("Conv 1×1", "32 → 4"))):
        s.block(x, yc - 27, 106, 54, CONV if i < 3 else WHITE, lab, sb, 19, 16)
        s.line([(x + 110, yc), (x + 122, yc)]) if i < 3 else None
        x += 126
    s.text(176 + (3 * 126 - 20) / 2, yc + 58, "each followed by BN and SiLU", 18, 500, MUTED, italic=True)
    s.text(554 + 53, yc + 58, "bias log(0.01/0.99)", 18, 500, MUTED, italic=True)
    s.line([(664, yc), (680, yc)])
    s.block(684, yc - 20, 40, 40, POOL, "σ", label_size=22)
    tx = 752
    s.line([(728, yc), (tx - 4, yc)])
    for i, (key, lab) in enumerate(names):
        xx = tx + i * 68
        s.image(xx, yc - 28, 56, 56, thumbs[key], sw=1.6)
        s.text(xx + 28, yc + 50, lab, 18, 600, MUTED)
    s.add(f'<text x="{tx + 130}" y="{yc - 42}" font-family="{FONT}" font-size="19" font-weight="700" fill="{INK}" text-anchor="middle">A<tspan font-size="13" dy="5">k</tspan><tspan dy="-5">: one map per type, 4 × 160²</tspan></text>')
    s.line([(tx + 4 * 68 - 8, yc), (1026, yc)])
    s.block(1030, yc - 20, 58, 40, POOL, "max" + MK, label_size=19)
    s.line([(1092, yc), (1106, yc)])
    s.image(1110, yc - 28, 56, 56, thumbs["abar"], stroke=NEW_DARK, sw=3)
    s.text(1138, yc + 50, "Ā → AGA", 19, 800, NEW_DARK, family=MATH)
    gx = 1236
    s.rect(1214, 980, 548, 250, WHITE, PANEL_EDGE, 2, rx=12)
    s.text(gx, 1012, "Targets per type and pixel", 20, 800, INK, anchor="start")
    for i, (lab, src, fill) in enumerate((("present", "synthetic artifact masks", WHITE),
                                          ("absent", "image-level flags, colour/morphology rules", "#26324A"),
                                          ("unknown", "everything else, ignored by the loss", "url(#hatch)"))):
        y = 1028 + i * 44
        s.rect(gx, y, 30, 30, fill, INK, 1.6, rx=3)
        s.text(gx + 42, y + 21, lab, 19, 800, INK, anchor="start")
        s.text(gx + 136, y + 21, src, 18, 500, MUTED, anchor="start")
    s.add(f'<text x="{gx}" y="1186" font-family="{MATH}" font-size="20" fill="{INK}">ℒ<tspan font-size="13" dy="5">art</tspan><tspan dy="-5">: BCE + soft Dice on labelled cells,</tspan></text>')
    s.add(f'<text x="{gx}" y="1214" font-family="{MATH}" font-size="20" fill="{INK}">averaged over types;  ℒ = ℒ<tspan font-size="13" dy="5">seg</tspan><tspan dy="-5"> + λ ℒ</tspan><tspan font-size="13" dy="5">art</tspan></text>')
    s.line([(1212, 1198), (tx + 130, 1198), (tx + 130, yc + 60)], color=TITLE_A, sw=2.2, marker="arrow-blue")
    s.add(f'<text x="1200" y="1190" font-family="{FONT}" font-size="18" font-weight="600" fill="{TITLE_A}" font-style="italic" text-anchor="end">supervises A<tspan font-size="12" dy="4">k</tspan></text>')

    # =========================================================== legend
    ly = 1262
    s.rect(20, ly, 1760, 52, WHITE, PANEL_EDGE, 2, rx=12)
    items = [("box", CONV, "Conv"), ("box", POOL, "Pooling / σ / max"), ("box", HEAD, "Segment26 head"),
             ("box", NECK, "Neck"), ("box", NEW, "Added by ArtiGate-YOLO"), ("cub", None, "Feature map"),
             ("dash", None, "Artifact map Ā"), ("opx", None, "Element-wise product"), ("opp", None, "Sum")]
    widths = []
    for kind, _, lab in items:
        sw_ = {"box": 40, "cub": 40, "dash": 52}.get(kind, 34)
        widths.append(sw_ + measure(lab, 19, 600))
    gap = (1760 - 48 - sum(widths)) / (len(items) - 1)
    x = 44
    for (kind, col, lab), wdt in zip(items, widths):
        if kind == "box":
            s.rect(x, ly + 15, 30, 22, col, INK, 1.6, rx=3)
            x2 = x + 40
        elif kind == "cub":
            s.cuboid(x, ly + 18, 20, 20, 8)
            x2 = x + 40
        elif kind == "dash":
            s.line([(x, ly + 26), (x + 40, ly + 26)], color=NEW_DARK, dash="8 5", sw=2.6)
            x2 = x + 52
        else:
            s.op(x + 12, ly + 26, "x" if kind == "opx" else "+", r=12)
            x2 = x + 34
        s.text(x2, ly + 33, lab, 19, 600, INK, anchor="start")
        x += wdt + gap
    return s.svg()



_TTF_FACES = (("Inter-Medium", 500, "normal"), ("Inter-SemiBold", 600, "normal"), ("Inter-Bold", 700, "normal"),
              ("Inter-ExtraBold", 800, "normal"), ("Inter-MediumItalic", 500, "italic"),
              ("Inter-SemiBoldItalic", 600, "italic"))


def _otf_to_ttf(src, dst):
    """CFF (OTF) -> TrueType outlines, so Chromium embeds the font as TrueType instead of Type 3
    (IEEE PDF checks reject Type 3 fonts). Adapted from fontTools' otf2ttf snippet."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(str(src))
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in glyph_set.keys():
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    for name, glyph in glyf.glyphs.items():
        if hasattr(glyph, "xMin"):
            font["hmtx"][name] = (font["hmtx"][name][0], glyph.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for field in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                  "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, field, 1 if field == "maxZones" else 0)
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    font.sfntVersion = "\000\001\000\000"
    font.save(str(dst))


def font_face_css():
    """@font-face rules for TrueType copies of Inter (made once in the temp folder); '' if unavailable."""
    import tempfile

    src_dir = Path("/usr/share/fonts/opentype/inter")
    cache = Path(tempfile.gettempdir()) / "artigate_fonts"
    rules = []
    try:
        cache.mkdir(exist_ok=True)
        for stem, weight, style in _TTF_FACES:
            dst = cache / f"{stem}.ttf"
            if not dst.exists():
                _otf_to_ttf(src_dir / f"{stem}.otf", dst)
            rules.append(f"@font-face {{ font-family: 'InterTTF'; src: url('{dst.resolve().as_uri()}'); "
                         f"font-weight: {weight}; font-style: {style}; }}")
    except Exception as exc:  # noqa: BLE001 - fall back to the system font
        print(f"note: Inter TrueType conversion unavailable ({exc}); the PDF may use Type 3 fonts")
        return ""
    return "\n".join(rules)


def export(svg_path, pdf_path, png_path, w, h):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: wrote SVG only")
        return False
    faces = font_face_css()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
{faces}
@page {{ size: {w / 1800 * 7.16:.4f}in {h / 1800 * 7.16:.4f}in; margin: 0 }}
html, body {{ margin: 0; padding: 0 }} svg {{ display: block; width: 100%; height: auto }}
</style></head><body>{svg_path.read_text(encoding='utf-8')}</body></html>"""
    tmp = svg_path.with_suffix(".tmp.html")
    tmp.write_text(html, encoding="utf-8")
    import os

    exe = "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page(viewport={"width": int(w / 1800 * 7.16 * 96), "height": int(h / 1800 * 7.16 * 96)})
        page.goto(tmp.resolve().as_uri())
        page.wait_for_timeout(300)
        page.pdf(path=str(pdf_path), width=f"{w / 1800 * 7.16:.4f}in", height=f"{h / 1800 * 7.16:.4f}in",
                 print_background=True, page_ranges="1")
        page2 = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        page2.goto(tmp.resolve().as_uri())
        page2.wait_for_timeout(300)
        page2.screenshot(path=str(png_path), full_page=True)
        browser.close()
    tmp.unlink()
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--mask")
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures" / "architecture"))
    args = ap.parse_args(argv)
    if bool(args.image) != bool(args.mask):
        ap.error("--image and --mask go together")
    base, gt = load_pair(args.image, args.mask, 320) if args.image else drawn_lesion(320)
    svg = build(make_thumbnails(base, gt))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out.with_suffix(".svg")
    svg_path.write_text(svg, encoding="utf-8")
    w, h = 1800, 1330
    ok = export(svg_path, out.with_suffix(".pdf"), out.with_suffix(".png"), w, h)
    print(f"wrote {svg_path}" + (f", {out.with_suffix('.pdf')}, {out.with_suffix('.png')}" if ok else ""))


if __name__ == "__main__":
    main()
