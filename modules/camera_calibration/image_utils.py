from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional fallback
    Image = None


def load_image_any_depth(path: str | Path) -> np.ndarray | None:
    """Load an image while preserving high bit depth when possible."""
    path = Path(path)

    if path.suffix.lower() in {".tif", ".tiff"}:
        pil_img = _load_tiff_with_pillow(path)
        if pil_img is not None:
            return pil_img

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img

    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    return _load_tiff_with_pillow(path)


def _load_tiff_with_pillow(path: Path) -> np.ndarray | None:
    if Image is None:
        return None

    try:
        with Image.open(path) as pil_img:
            arr = np.array(pil_img)
    except Exception:
        return None

    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

    return arr


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image to uint8 for display or classical CV routines."""
    if image.dtype == np.uint8:
        return image
    if image.size == 0:
        return image.astype(np.uint8)

    scaled = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return scaled.astype(np.uint8)


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a loaded image to an 8-bit grayscale image."""
    if image.ndim == 2:
        return to_uint8(image)

    if image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return to_uint8(gray)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a loaded image to an 8-bit RGB image for Qt display."""
    if image.ndim == 2:
        return cv2.cvtColor(to_uint8(image), cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_uint8(rgb)


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a loaded image to an 8-bit BGR image for OpenCV drawing."""
    if image.ndim == 2:
        return cv2.cvtColor(to_uint8(image), cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    return to_uint8(bgr)
