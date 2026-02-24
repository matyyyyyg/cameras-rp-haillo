from .classification import AgeGenderClassifier, ClassificationResult, extract_face_crop
from .kalman_tracker import KalmanPersonTracker, TrackedPerson, format_output_json
from .unified_hailo_face import UnifiedHailoFaceDetector, is_hailo_available

__version__ = "0.2.0"
__all__ = [
    "AgeGenderClassifier",
    "ClassificationResult",
    "extract_face_crop",
    "KalmanPersonTracker",
    "TrackedPerson",
    "format_output_json",
    "UnifiedHailoFaceDetector",
    "is_hailo_available",
]
