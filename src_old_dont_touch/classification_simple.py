from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Same structure as classification.py for drop-in comparison."""
    age_bucket: str
    age_midpoint: float
    age_confidence: float
    gender: str
    gender_confidence: float
    raw_age: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self):
        result = {
            "age_bucket": self.age_bucket,
            "age_midpoint": self.age_midpoint,
            "age_confidence": round(self.age_confidence, 4),
            "gender": self.gender,
            "gender_confidence": round(self.gender_confidence, 4),
        }
        if self.raw_age is not None:
            result["raw_age"] = round(self.raw_age, 1)
        return result


AGE_BUCKETS = [
    "(0-2)", "(3-6)", "(7-12)", "(13-19)",
    "(20-29)", "(30-39)", "(40-49)", "(50-59)", "(60+)"
]


def get_age_bucket(age: int) -> Tuple[str, int]:
    if age <= 2:    return "(0-2)", 1
    elif age <= 6:  return "(3-6)", 5
    elif age <= 12: return "(7-12)", 10
    elif age <= 19: return "(13-19)", 16
    elif age <= 29: return "(20-29)", 25
    elif age <= 39: return "(30-39)", 35
    elif age <= 49: return "(40-49)", 45
    elif age <= 59: return "(50-59)", 55
    else:           return "(60+)", 70


def extract_face_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.3,
) -> np.ndarray:
    """
    Single-pass face crop with real context from the frame.

    No upscaling — InsightFace will resize to det_size anyway,
    so intermediate upscales just waste CPU and cancel out.

    Args:
        frame: Full BGR frame
        bbox: (x1, y1, x2, y2)
        padding: Fraction of box size to add as real context (default 0.3)

    Returns:
        Cropped face region
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    return frame[y1:y2, x1:x2].copy()


class SimpleInsightFaceClassifier:
    """
    Simplified classifier: one crop, one call to InsightFace.

    Key difference from the original:
    - No intermediate upscaling (cancels out at InsightFace resize)
    - No border padding (dilutes face in the 320x320 input)
    - No retry strategies (single clean pass)
    - Optional CLAHE (the one preprocessing step that adds real value)

    Face pixel formula:
        face_pixels = D / (1 + 2 * padding)
        With padding=0.3, D=320: face = 200px (62.5% of input)
        With padding=0.5, D=320: face = 160px (50.0% of input)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (320, 320),
        use_clahe: bool = True,
    ):
        self.model_name = model_name
        self.det_size = det_size
        self.use_clahe = use_clahe
        self._app = None
        self._initialized = False

        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=model_name,
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'genderage', 'recognition'],
            )
            self._app.prepare(ctx_id=0, det_size=det_size)
            self._initialized = True
            logger.info(
                f"SimpleInsightFace initialized: {model_name} "
                f"det_size={det_size[0]}x{det_size[1]} clahe={use_clahe}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize InsightFace: {e}")

    @property
    def is_available(self) -> bool:
        return self._initialized and self._app is not None

    def classify_face(self, face_crop: np.ndarray) -> Optional[ClassificationResult]:
        """
        Classify a face crop in a single pass.

        Args:
            face_crop: BGR face image (already cropped with padding from frame)

        Returns:
            ClassificationResult or None
        """
        if not self.is_available or face_crop is None or face_crop.size == 0:
            return None

        try:
            image = face_crop

            # CLAHE — the one preprocessing step that actually helps
            if self.use_clahe:
                try:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
                except Exception:
                    pass

            # Single call — InsightFace resizes to det_size internally
            faces = self._app.get(image)

            if not faces:
                return None

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            raw_age = float(face.age) if hasattr(face, 'age') else 25.0
            age_bucket, _ = get_age_bucket(int(raw_age))

            gender = "Male" if face.gender == 1 else "Female"
            gender_conf = 0.85 if hasattr(face, 'gender') else 0.5

            embedding = None
            if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                embedding = face.normed_embedding

            return ClassificationResult(
                age_bucket=age_bucket,
                age_midpoint=round(raw_age, 1),
                age_confidence=0.90,
                gender=gender,
                gender_confidence=gender_conf,
                raw_age=raw_age,
                embedding=embedding,
            )

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None


class SimpleAgeGenderClassifier:
    """
    Simplified wrapper matching AgeGenderClassifier interface.

    No Caffe fallback, no Hailo gender, no ensemble.
    Just InsightFace with a clean crop.
    """

    def __init__(
        self,
        insightface_model: str = "buffalo_l",
        det_size: Tuple[int, int] = (320, 320),
        use_clahe: bool = True,
        **kwargs,
    ):
        self._insightface = SimpleInsightFaceClassifier(
            model_name=insightface_model,
            det_size=det_size,
            use_clahe=use_clahe,
        )
        logger.info("SimpleAgeGenderClassifier initialized")

    def classify(self, face_crop: np.ndarray) -> ClassificationResult:
        if face_crop is None or face_crop.size == 0:
            return ClassificationResult(
                age_bucket="unknown", age_midpoint=-1,
                age_confidence=0.0, gender="unknown",
                gender_confidence=0.0,
            )

        result = self._insightface.classify_face(face_crop)
        if result is not None:
            return result

        return ClassificationResult(
            age_bucket="unknown", age_midpoint=-1,
            age_confidence=0.0, gender="unknown",
            gender_confidence=0.0,
        )

    def classify_batch(self, face_crops: List[np.ndarray]) -> List[ClassificationResult]:
        return [self.classify(crop) for crop in face_crops]
