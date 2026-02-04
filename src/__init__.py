"""
CCTV Analytics System

A modular face detection and analytics system for CCTV applications.
Supports CPU-based inference (OpenCV DNN) with architecture ready for
hardware acceleration (Hailo-8L).

Modules:
    - detectors: Face detection backends
    - classification: Age and gender estimation
    - tracking: Multi-person tracking
    - logging_utils: CSV/JSONL logging utilities
    - main: Main pipeline entrypoint
"""

from .detectors import BaseFaceDetector, OpenCVDNNFaceDetector, create_detector
from .classification import AgeGenderClassifier, ClassificationResult, extract_face_crop
from .tracking import SimpleTracker, TrackedObject
from .logging_utils import DetectionLogger, DetectionEvent, setup_logging

__version__ = "0.1.0"
__all__ = [
    "BaseFaceDetector",
    "OpenCVDNNFaceDetector", 
    "create_detector",
    "AgeGenderClassifier",
    "ClassificationResult",
    "extract_face_crop",
    "SimpleTracker",
    "TrackedObject",
    "DetectionLogger",
    "DetectionEvent",
    "setup_logging",
]
