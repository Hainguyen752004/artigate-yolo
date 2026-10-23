"""Fixed preprocessing views from the ICEEIS pipeline.

v1: letterbox only. v7: fixed-threshold ROI crop, DullRazor-inspired hair removal, Gray-World
color constancy, then letterbox. The names are legacy ICEEIS identifiers. The v7 chain is applied
to every image it is asked for; it does not detect artifacts.
"""
import cv2
import numpy as np

VIEWS = ("v1", "v7")
PAD_COLOR = (114, 114, 114)


def smart_roi_crop(img):
    """Crop to the bounding box of the largest region brighter than gray level 15."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, 0, 0, img.shape[1], img.shape[0]
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y + h, x:x + w], x, y, w, h


def dullrazor(img, return_mask=False):
    """17x17 cross black-hat, threshold 10, Telea inpainting with radius 3."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return (inpainted, mask) if return_mask else inpainted


def gray_world_color_constancy(img):
    b, g, r = cv2.split(img.astype(np.float32))
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        return img
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    b = np.clip(b * (avg_gray / avg_b), 0, 255)
    g = np.clip(g * (avg_gray / avg_g), 0, 255)
    r = np.clip(r * (avg_gray / avg_r), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def letterbox(img, new_size=640, color=PAD_COLOR, interpolation=cv2.INTER_LINEAR):
    """Resize to fit ``new_size`` and pad to a centered square; works for color images and masks."""
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
    canvas = np.full((new_size, new_size) + img.shape[2:], color, dtype=img.dtype)
    pad_x, pad_y = (new_size - nw) // 2, (new_size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def make_view(image, mask, view, size=640):
    """Apply one view to a BGR image and its binary lesion mask with identical geometry.

    Returns the view image, the view mask (bool) and a JSON-compatible transform record that
    maps canvas pixels back to the source image.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    height, width = image.shape[:2]
    if mask.shape[:2] != (height, width):
        raise ValueError(f"mask size {mask.shape[:2]} differs from image size {(height, width)}")
    x, y, w, h = 0, 0, width, height
    if view == "v7":
        image, x, y, w, h = smart_roi_crop(image)
        mask = mask[y:y + h, x:x + w]
        image = gray_world_color_constancy(dullrazor(image))
    image, scale, pad_x, pad_y = letterbox(image, size)
    mask, _, _, _ = letterbox(mask.astype(np.uint8) * 255, size, color=0, interpolation=cv2.INTER_NEAREST)
    transform = {
        "view": view,
        "source_size": [width, height],
        "crop_box": [int(x), int(y), int(w), int(h)],
        "scale": float(scale),
        "pad": [int(pad_x), int(pad_y)],
        "canvas_size": size,
    }
    return image, mask > 127, transform
