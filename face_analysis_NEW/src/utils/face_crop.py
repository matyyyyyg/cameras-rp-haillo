import cv2
import numpy as np
from typing import Tuple, Dict


def extract_face_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.5
) -> np.ndarray:
    """
    Extract a face crop from a frame with padding.

    Args:
        frame: Full BGR frame
        bbox: Bounding box as (x1, y1, x2, y2)
        padding: Fraction of box size to add as padding (default 50%)

    Returns:
        Cropped face region as BGR image
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Calculate padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)

    # Apply padding with boundary checks
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    # Extract crop
    face_crop = frame[y1:y2, x1:x2].copy()

    # Upscale small crops for better classification
    crop_h, crop_w = face_crop.shape[:2]
    min_size = 120
    if crop_h < min_size or crop_w < min_size:
        scale = max(min_size / crop_h, min_size / crop_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)
        face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return face_crop


def check_face_quality(face_crop: np.ndarray) -> Tuple[bool, Dict]:
    """
    Heuristic quality gate for face crops.

    Checks:
        - Blur via Laplacian variance (> 15)
        - Mean brightness between 30-240
        - Minimum size 40x40 pixels

    Args:
        face_crop: BGR face image

    Returns:
        Tuple of (passed, {"blur": val, "brightness": val, "size": (w, h)})
    """
    h, w = face_crop.shape[:2]

    # Convert to grayscale for quality checks
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # Blur check: Laplacian variance
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness check: mean pixel intensity
    brightness = float(gray.mean())

    metrics = {
        "blur": round(blur, 2),
        "brightness": round(brightness, 2),
        "size": (w, h)
    }

    passed = (
        blur > 15.0
        and 30.0 <= brightness <= 240.0
        and w >= 40
        and h >= 40
    )

    return passed, metrics


def apply_conditional_clahe(face_crop: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE only when image is dark (mean brightness < 100).

    Args:
        face_crop: BGR face image

    Returns:
        Enhanced (or original) BGR image
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= 100:
        return face_crop

    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
