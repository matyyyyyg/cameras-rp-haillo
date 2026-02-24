import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    gender: str
    gender_confidence: float
    age: float
    age_confidence: float
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "gender": self.gender,
            "gender_confidence": round(self.gender_confidence, 3),
            "age": round(self.age, 1),
            "age_confidence": round(self.age_confidence, 3)
        }


class InsightFaceClassifier:

    def __init__(self, model_name: str = "buffalo_l", min_face_size: int = 40):
        self.min_face_size = min_face_size
        self._initialized = False
        self._app = None

        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=model_name,
                providers=['CPUExecutionProvider'],
                allowed_modules=['genderage', 'recognition']
            )
            self._app.prepare(ctx_id=0, det_size=(160, 160))
            self._initialized = True
            logger.info(f"InsightFace initialized: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            self._initialized = False

    def preprocess_face(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        if face_crop is None or face_crop.size == 0:
            return None

        h, w = face_crop.shape[:2]

        if h < self.min_face_size or w < self.min_face_size:
            return None

        try:
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            face_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except:
            pass

        min_size = 112
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad = 20
        face_crop = cv2.copyMakeBorder(
            face_crop, pad, pad, pad, pad,
            cv2.BORDER_REPLICATE
        )

        return face_crop

    def classify(self, face_crop: np.ndarray) -> Optional[ClassificationResult]:
        if not self._initialized or self._app is None:
            return None

        processed = self.preprocess_face(face_crop)
        if processed is None:
            return None

        try:
            faces = self._app.get(processed)

            if not faces:
                return None

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            age = float(face.age) if hasattr(face, 'age') else 0.0
            gender = "male" if face.gender == 1 else "female"
            gender_conf = 0.85

            embedding = None
            if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                embedding = face.normed_embedding.copy()

            return ClassificationResult(
                gender=gender,
                gender_confidence=gender_conf,
                age=age,
                age_confidence=0.85,
                embedding=embedding
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return None

    def classify_batch(self, face_crops: List[np.ndarray]) -> List[Optional[ClassificationResult]]:
        return [self.classify(crop) for crop in face_crops]


def extract_face_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.3
) -> np.ndarray:
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