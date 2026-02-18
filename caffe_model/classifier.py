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

    def to_dict(self) -> Dict:
        return {
            "gender": self.gender,
            "gender_confidence": round(self.gender_confidence, 3),
            "age": round(self.age, 1),
            "age_confidence": round(self.age_confidence, 3)
        }


class LightweightClassifier:

    INPUT_SIZE = (227, 227)
    CAFFE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
    AGE_GROUPS = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

    def __init__(
        self,
        gender_model_path: Optional[str] = None,
        age_model_path: Optional[str] = None,
        min_face_size: int = 40,
    ):
        self.min_face_size = min_face_size
        self._initialized = False

        project_root = Path(__file__).parent.parent

        gender_proto = project_root / "models" / "gender_deploy.prototxt"
        gender_model = Path(gender_model_path) if gender_model_path else project_root / "models" / "gender_net.caffemodel"
        age_proto = project_root / "models" / "age_deploy.prototxt"
        age_model = Path(age_model_path) if age_model_path else project_root / "models" / "age_net.caffemodel"

        try:
            for f in [gender_proto, gender_model, age_proto, age_model]:
                if not f.exists():
                    raise FileNotFoundError(f"Model file not found: {f}")

            self.gender_net = cv2.dnn.readNetFromCaffe(str(gender_proto), str(gender_model))
            self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self.age_net = cv2.dnn.readNetFromCaffe(str(age_proto), str(age_model))
            self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self._initialized = True
            logger.info("Caffe gender/age models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Caffe models: {e}")
            self._initialized = False

    def preprocess_face(self, face_crop: np.ndarray) -> np.ndarray:
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

        blob = cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0,
            size=self.INPUT_SIZE,
            mean=self.CAFFE_MEAN,
            swapRB=False,
            crop=False
        )

        return blob

    def classify(self, face_crop: np.ndarray) -> Optional[ClassificationResult]:
        if not self._initialized:
            return None

        blob = self.preprocess_face(face_crop)
        if blob is None:
            return None

        try:
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()

            gender_probs = gender_preds[0]
            gender_idx = gender_probs.argmax()
            gender_conf = float(gender_probs[gender_idx])
            gender = "male" if gender_idx == 0 else "female"

            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()

            age_probs = age_preds[0]
            age_idx = age_probs.argmax()
            age_conf = float(age_probs[age_idx])

            age_range = self.AGE_GROUPS[age_idx]
            age_midpoint = (age_range[0] + age_range[1]) / 2

            return ClassificationResult(
                gender=gender,
                gender_confidence=gender_conf,
                age=age_midpoint,
                age_confidence=age_conf
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
