import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import onnxruntime as ort
from ..utils.types import ClassificationResult, get_age_bucket
from ..utils.face_crop import apply_conditional_clahe

logger = logging.getLogger(__name__)

# VGGFace2 mean subtraction (BGR order, raw 0-255 pixel space)
VGGFACE2_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)

# HSE model constants
MIN_AGE = 1       # Model classes represent ages MIN_AGE .. MIN_AGE+N-1
MALE_THRESHOLD = 0.6  # Sigmoid value >= this → male


class AgeGenderClassifier:
    INPUT_SIZE = (224, 224)

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the HSE MobileNet ONNX age/gender classifier.

        The model expects 224x224 BGR input with VGGFace2 mean subtraction and
        produces two heads:
          - age_pred/Softmax  : softmax over ~101 age classes (ages 1-101)
          - gender_pred/Sigmoid: single sigmoid (>=0.6 → male)

        Args:
            model_path: Path to the ONNX model file. If None, looks for
                        models/age_gender.onnx relative to project root.
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. Install: pip install onnxruntime"
            )

        project_root = Path(__file__).parent.parent.parent
        if model_path is None:
            self.model_path = project_root / "models" / "age_gender.onnx"
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}\n"
                f"Please download the age/gender ONNX model to: {self.model_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )

        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]

        logger.info(f"AgeGenderClassifier loaded: {self.model_path.name}")
        logger.info(f"   Input: {self._input_name}")
        logger.info(f"   Outputs: {self._output_names}")

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Preprocess a BGR face crop for the HSE MobileNet model."""
        img = apply_conditional_clahe(face_crop)
        img = cv2.resize(img, self.INPUT_SIZE)

        # Keep BGR, raw 0-255 range, subtract VGGFace2 mean
        img = img.astype(np.float32) - VGGFACE2_MEAN

        # HWC -> CHW -> NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def classify(self, face_crop: np.ndarray) -> Optional[ClassificationResult]:

        if face_crop is None or face_crop.size == 0:
            return None

        try:
            input_tensor = self._preprocess(face_crop)
            outputs = self.session.run(self._output_names, {self._input_name: input_tensor})

            return self._parse_outputs(outputs)

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return None

    def _parse_outputs(self, outputs) -> ClassificationResult:
        """Parse HSE MobileNet outputs (age softmax + gender sigmoid)."""

        # Identify which output is age (long array) vs gender (single value)
        # by shape, since ONNX export may reorder them.
        flat = [o.flatten() for o in outputs]

        age_probs = None
        gender_val = None

        for arr in flat:
            if len(arr) > 2:
                age_probs = arr
            else:
                gender_val = float(arr[0])

        if age_probs is None or gender_val is None:
            raise ValueError(
                f"Unexpected output shapes: {[o.shape for o in outputs]}"
            )

        # --- Age: weighted average of top-2 softmax classes ---
        top2_idx = np.argsort(age_probs)[-2:]
        top2_probs = age_probs[top2_idx]
        top2_probs = top2_probs / top2_probs.sum()
        age = float(np.sum((top2_idx + MIN_AGE) * top2_probs))
        age = max(0.0, min(age, 100.0))

        # --- Gender: sigmoid value, threshold at MALE_THRESHOLD ---
        if gender_val >= MALE_THRESHOLD:
            gender = "male"
            gender_confidence = float(gender_val)
        else:
            gender = "female"
            gender_confidence = float(1.0 - gender_val)

        age_bucket = get_age_bucket(age)

        return ClassificationResult(
            age=round(age, 1),
            age_bucket=age_bucket,
            gender=gender,
            gender_confidence=round(gender_confidence, 4)
        )
