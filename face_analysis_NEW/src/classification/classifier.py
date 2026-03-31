import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import onnxruntime as ort
from ..utils.types import ClassificationResult, get_age_bucket
from ..utils.face_crop import apply_conditional_clahe

logger = logging.getLogger(__name__)

# Mean subtraction applied AFTER BGR→RGB conversion (matches facial_analysis.py)
CHANNEL_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)

# HSE model constants
MIN_AGE = 1           # Model classes represent ages MIN_AGE .. MIN_AGE+N-1
MALE_THRESHOLD = 0.6  # Sigmoid value >= this → male (from is_male() in source)


class AgeGenderClassifier:
    INPUT_SIZE = (224, 224)

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the HSE MobileNet ONNX age/gender classifier.
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
        self._first_run = True

        logger.info(f"AgeGenderClassifier loaded: {self.model_path.name}")
        logger.info(f"   Input: {self._input_name}")
        logger.info(f"   Outputs: {self._output_names}")

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Preprocess a BGR face crop matching facial_analysis.py exactly."""
        img = apply_conditional_clahe(face_crop)
        img = cv2.resize(img, self.INPUT_SIZE)

        # BGR → RGB (as in source: x = x[..., ::-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Subtract channel means on RGB (raw 0-255, no /255 scaling)
        img = img.astype(np.float32)
        img[..., 0] -= CHANNEL_MEAN[0]  # 103.939
        img[..., 1] -= CHANNEL_MEAN[1]  # 116.779
        img[..., 2] -= CHANNEL_MEAN[2]  # 123.68

        # NHWC (TF convention, kept by tf2onnx conversion)
        img = np.expand_dims(img, axis=0)

        return img

    def classify(self, face_crop: np.ndarray) -> Optional[ClassificationResult]:

        if face_crop is None or face_crop.size == 0:
            return None

        input_tensor = self._preprocess(face_crop)

        try:
            outputs = self.session.run(self._output_names, {self._input_name: input_tensor})
        except RuntimeError as e:
            logger.warning(f"ORT inference error (transient): {e}")
            return None

        return self._parse_outputs(outputs)

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

        # Debug: log raw outputs on first run
        if self._first_run:
            self._first_run = False
            logger.debug(f"age_probs shape={age_probs.shape}, top5={age_probs[np.argsort(age_probs)[-5:]]}")
            logger.debug(f"top5 indices={np.argsort(age_probs)[-5:]}")
            logger.debug(f"gender_val={gender_val:.4f}")

        # --- Age: expected value over full softmax distribution ---
        ages = np.arange(MIN_AGE, MIN_AGE + len(age_probs), dtype=np.float32)
        age = float(np.sum(ages * age_probs))
        age = max(0.0, min(age, 100.0))

        # --- Gender: sigmoid >= 0.6 → male (from source is_male()) ---
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
