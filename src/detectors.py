"""
Face Detection Backends

This module provides a common interface for face detectors and implements
both CPU-based detection using OpenCV DNN (SSD face detector) and 
RetinaFace for better detection of small, blurry, and side faces.

The architecture is designed to easily add new backends (e.g., Hailo-8L)
by implementing the BaseFaceDetector interface.


Created: 2024
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

import cv2
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class BaseFaceDetector(ABC):
    """
    Abstract base class for face detection backends.
    
    All face detector implementations should inherit from this class
    and implement the detect_faces method.
    
    This abstraction allows swapping detection backends (CPU/Hailo/etc.)
    without changing the rest of the pipeline.
    """
    
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a BGR frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            
        Returns:
            List of detection dictionaries, each containing:
                - "bbox": Tuple[int, int, int, int] as (x1, y1, x2, y2)
                - "confidence": float between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement detect_faces()")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Optional preprocessing step. Override in subclasses if needed.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed frame
        """
        return frame
    
    def get_backend_name(self) -> str:
        """
        Return the name of this detection backend.
        
        Returns:
            String identifier for the backend
        """
        return self.__class__.__name__


def enhance_image_for_detection(frame: np.ndarray, 
                                 apply_clahe: bool = True,
                                 apply_denoise: bool = False) -> np.ndarray:
    """
    Enhance image quality for better face detection in CCTV footage.
    
    Args:
        frame: Input BGR frame
        apply_clahe: Apply CLAHE for contrast enhancement
        apply_denoise: Apply denoising (slower but helps with noise)
        
    Returns:
        Enhanced BGR frame
    """
    enhanced = frame.copy()
    
    # Convert to LAB color space for CLAHE on luminance only
    if apply_clahe:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply denoising (optional, more expensive)
    if apply_denoise:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    
    return enhanced


class OpenCVDNNFaceDetector(BaseFaceDetector):
    """
    Face detector using OpenCV DNN with SSD (Single Shot Detector).
    
    Uses the res10_300x300_ssd model which provides good accuracy
    and reasonable performance on CPU.
    
    Attributes:
        confidence_threshold: Minimum confidence to accept a detection
        input_size: Input dimensions for the neural network
    """
    
    # Default model paths relative to project root
    DEFAULT_PROTOTXT = "models/deploy.prototxt"
    DEFAULT_CAFFEMODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    def __init__(
        self,
        prototxt_path: Optional[str] = None,
        caffemodel_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (300, 300),
        scale_factor: float = 1.0,
        mean_values: Tuple[float, float, float] = (104.0, 177.0, 123.0),
        enhance_image: bool = False
    ):
        """
        Initialize the OpenCV DNN face detector.
        
        Args:
            prototxt_path: Path to deploy.prototxt file
            caffemodel_path: Path to .caffemodel file
            confidence_threshold: Minimum confidence (0-1) to accept detection
            input_size: Network input dimensions (width, height)
            scale_factor: Scale factor for blob creation
            mean_values: Mean subtraction values (BGR order)
            enhance_image: Whether to enhance images before detection
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.mean_values = mean_values
        self.enhance_image = enhance_image
        
        # Resolve model paths
        project_root = Path(__file__).parent.parent
        self.prototxt_path = Path(prototxt_path) if prototxt_path else project_root / self.DEFAULT_PROTOTXT
        self.caffemodel_path = Path(caffemodel_path) if caffemodel_path else project_root / self.DEFAULT_CAFFEMODEL
        
        # Load the network
        self._load_network()
        
        logger.info(f"Initialized {self.get_backend_name()} with confidence threshold {confidence_threshold}")
    
    def _load_network(self) -> None:
        """Load the Caffe model into OpenCV DNN."""
        if not self.prototxt_path.exists():
            raise FileNotFoundError(f"Prototxt not found: {self.prototxt_path}")
        if not self.caffemodel_path.exists():
            raise FileNotFoundError(f"Caffemodel not found: {self.caffemodel_path}")
        
        logger.info(f"Loading face detection model from {self.caffemodel_path}")
        
        self.net = cv2.dnn.readNetFromCaffe(
            str(self.prototxt_path),
            str(self.caffemodel_path)
        )
        
        # Set preferable backend and target for CPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        logger.info("Face detection model loaded successfully")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in the input frame using SSD.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detection dictionaries with 'bbox' and 'confidence'
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received for face detection")
            return []
        
        # Optionally enhance image for better detection
        if self.enhance_image:
            frame = enhance_image_for_detection(frame)
        
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=self.scale_factor,
            size=self.input_size,
            mean=self.mean_values,
            swapRB=False,
            crop=False
        )
        
        # Forward pass through the network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Parse detections
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box coordinates (normalized 0-1)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confidence)
            })
        
        logger.debug(f"Detected {len(results)} faces in frame")
        return results
    
    def get_backend_name(self) -> str:
        """Return backend identifier."""
        return "OpenCV_DNN_SSD"


class MultiScaleOpenCVDetector(BaseFaceDetector):
    """
    Multi-scale face detector that processes the image at multiple resolutions.
    
    This helps detect small faces in wide-angle CCTV footage by upscaling
    the image before detection, then mapping coordinates back.
    """
    
    DEFAULT_PROTOTXT = "models/deploy.prototxt"
    DEFAULT_CAFFEMODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    def __init__(
        self,
        prototxt_path: Optional[str] = None,
        caffemodel_path: Optional[str] = None,
        confidence_threshold: float = 0.3,  # Lower threshold for multi-scale
        scales: Tuple[float, ...] = (1.0, 1.5, 2.0),  # Multiple scales
        min_face_size: int = 20,  # Minimum face size in pixels
        nms_threshold: float = 0.3,  # Non-maximum suppression threshold
        enhance_image: bool = True
    ):
        """
        Initialize multi-scale detector.
        
        Args:
            scales: Tuple of scale factors to use
            min_face_size: Minimum face size to detect
            nms_threshold: IoU threshold for NMS
        """
        self.confidence_threshold = confidence_threshold
        self.scales = scales
        self.min_face_size = min_face_size
        self.nms_threshold = nms_threshold
        self.enhance_image = enhance_image
        
        # Resolve model paths
        project_root = Path(__file__).parent.parent
        self.prototxt_path = Path(prototxt_path) if prototxt_path else project_root / self.DEFAULT_PROTOTXT
        self.caffemodel_path = Path(caffemodel_path) if caffemodel_path else project_root / self.DEFAULT_CAFFEMODEL
        
        # Load the network
        self._load_network()
        
        logger.info(f"Initialized MultiScaleOpenCVDetector with scales {scales}")
    
    def _load_network(self) -> None:
        """Load the Caffe model."""
        if not self.prototxt_path.exists():
            raise FileNotFoundError(f"Prototxt not found: {self.prototxt_path}")
        if not self.caffemodel_path.exists():
            raise FileNotFoundError(f"Caffemodel not found: {self.caffemodel_path}")
        
        self.net = cv2.dnn.readNetFromCaffe(
            str(self.prototxt_path),
            str(self.caffemodel_path)
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def _detect_at_scale(self, frame: np.ndarray, scale: float) -> List[Dict]:
        """Detect faces at a specific scale."""
        if scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_frame = frame
            scale = 1.0
        
        h, w = scaled_frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            scaled_frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < self.confidence_threshold:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Scale back to original image coordinates
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            
            # Check minimum size
            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue
            
            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confidence),
                "scale": scale
            })
        
        return results
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicates."""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Handle different OpenCV versions
        if isinstance(indices, tuple):
            indices = indices[0]
        indices = indices.flatten()
        
        return [detections[i] for i in indices]
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces at multiple scales."""
        if frame is None or frame.size == 0:
            return []
        
        # Optionally enhance image
        if self.enhance_image:
            frame = enhance_image_for_detection(frame)
        
        # Collect detections from all scales
        all_detections = []
        for scale in self.scales:
            detections = self._detect_at_scale(frame, scale)
            all_detections.extend(detections)
        
        # Apply NMS to remove duplicates
        final_detections = self._nms(all_detections)
        
        # Clamp to frame boundaries
        h, w = frame.shape[:2]
        for det in final_detections:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = (
                max(0, x1),
                max(0, y1),
                min(w, x2),
                min(h, y2)
            )
        
        logger.debug(f"Multi-scale detected {len(final_detections)} faces")
        return final_detections
    
    def get_backend_name(self) -> str:
        return "MultiScale_OpenCV_DNN"


class MTCNNDetector(BaseFaceDetector):
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks).
    
    MTCNN is excellent for:
    - Detecting side/profile faces
    - Detecting small faces in wide-angle footage
    - Handling partially occluded faces
    - Robust detection in varying lighting conditions
    
    Uses mtcnn-opencv which doesn't require TensorFlow.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_face_size: int = 20,  # Minimum face size in pixels
        scale_factor: float = 0.709,  # Image pyramid scale factor
        enhance_image: bool = True
    ):
        """
        Initialize MTCNN detector.
        
        Args:
            confidence_threshold: Minimum confidence for detection
            min_face_size: Minimum face size to detect (pixels)
            scale_factor: Scale factor for image pyramid
            enhance_image: Apply image enhancement for CCTV
        """
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.enhance_image = enhance_image
        
        self._mtcnn = None
        self._use_mtcnn = False
        self._fallback_detector = None
        
        # Try to import mtcnn-opencv
        try:
            from mtcnn_cv2 import MTCNN
            self._mtcnn = MTCNN(
                min_face_size=min_face_size,
                scale_factor=scale_factor
            )
            self._use_mtcnn = True
            logger.info(f"MTCNN-OpenCV loaded successfully (min_face={min_face_size}px)")
        except ImportError:
            logger.warning("mtcnn-opencv not installed. Install with: pip install mtcnn-opencv")
            logger.warning("Falling back to multi-scale OpenCV detector")
            self._use_mtcnn = False
            self._fallback_detector = MultiScaleOpenCVDetector(
                confidence_threshold=confidence_threshold * 0.6,
                enhance_image=enhance_image
            )
        
        logger.info(f"Initialized MTCNNDetector with threshold {confidence_threshold}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using MTCNN.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if frame is None or frame.size == 0:
            return []
        
        # Use fallback if MTCNN not available
        if not self._use_mtcnn:
            return self._fallback_detector.detect_faces(frame)
        
        # Optionally enhance image
        if self.enhance_image:
            frame = enhance_image_for_detection(frame)
        
        h, w = frame.shape[:2]
        
        try:
            # MTCNN expects RGB, convert from BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self._mtcnn.detect_faces(rgb_frame)
            
            if not faces:
                return []
            
            results = []
            for face in faces:
                # MTCNN returns bounding box as [x, y, width, height]
                bbox = face.get("box", [])
                confidence = face.get("confidence", 0.0)
                
                if confidence < self.confidence_threshold:
                    continue
                
                if len(bbox) == 4:
                    x, y, bw, bh = bbox
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w, x + bw)
                    y2 = min(h, y + bh)
                    
                    if x2 > x1 and y2 > y1:
                        results.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": float(confidence)
                        })
            
            logger.debug(f"MTCNN detected {len(results)} faces")
            return results
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            if self._fallback_detector:
                return self._fallback_detector.detect_faces(frame)
            return []
    
    def get_backend_name(self) -> str:
        return "MTCNN"


# Alias for backward compatibility
RetinaFaceDetector = MTCNNDetector


class HailoFaceDetector(BaseFaceDetector):
    """
    Hardware-accelerated face detector using Hailo-8L.
    
    Provides 25-30+ FPS on Raspberry Pi 5 with Hailo-8L AI accelerator.
    Falls back to MTCNN if Hailo is not available.
    
    Requirements:
    - Raspberry Pi 5 with Hailo-8L HAT
    - HailoRT and hailo-platform installed
    - Pre-compiled HEF model file
    """
    
    def __init__(
        self,
        hef_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        enhance_image: bool = False  # Hailo doesn't need enhancement
    ):
        """
        Initialize Hailo face detector.
        
        Args:
            hef_path: Path to compiled Hailo Executable Format file
            confidence_threshold: Minimum confidence threshold
            enhance_image: Ignored for Hailo (preprocessing done on device)
        """
        self.confidence_threshold = confidence_threshold
        self.hef_path = hef_path
        self._hailo_detector = None
        self._fallback_detector = None
        self._use_hailo = False
        
        # Try to import and initialize Hailo detector
        try:
            from .hailo_detector import HailoFaceDetector as HailoImpl, is_hailo_available
            
            if is_hailo_available():
                self._hailo_detector = HailoImpl(
                    face_hef_path=hef_path,
                    confidence_threshold=confidence_threshold
                )
                self._use_hailo = True
                logger.info("Hailo-8L detector initialized successfully!")
            else:
                logger.warning("Hailo-8L not available on this system")
                self._setup_fallback(confidence_threshold)
                
        except ImportError as e:
            logger.warning(f"Hailo module not available: {e}")
            self._setup_fallback(confidence_threshold)
        except Exception as e:
            logger.warning(f"Failed to initialize Hailo: {e}")
            self._setup_fallback(confidence_threshold)
    
    def _setup_fallback(self, confidence_threshold: float) -> None:
        """Setup fallback detector when Hailo is not available."""
        logger.info("Falling back to MTCNN detector (CPU)")
        self._fallback_detector = MTCNNDetector(
            confidence_threshold=confidence_threshold,
            enhance_image=True
        )
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using Hailo-8L accelerator or fallback.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if self._use_hailo and self._hailo_detector:
            return self._hailo_detector.detect_faces(frame)
        elif self._fallback_detector:
            return self._fallback_detector.detect_faces(frame)
        else:
            logger.error("No detector available")
            return []
    
    def get_backend_name(self) -> str:
        """Return backend identifier."""
        if self._use_hailo:
            return "Hailo_8L"
        elif self._fallback_detector:
            return f"Fallback_{self._fallback_detector.get_backend_name()}"
        return "None"


def create_detector(backend: str = "mtcnn", **kwargs) -> BaseFaceDetector:
    """
    Factory function to create a face detector instance.
    
    Args:
        backend: Detection backend to use:
            - "opencv": Standard OpenCV DNN SSD detector (fast, frontal faces only)
            - "multiscale": Multi-scale OpenCV detector (better for small faces)
            - "mtcnn": MTCNN detector (best for side faces, small faces, CCTV)
            - "retinaface": Alias for MTCNN detector
            - "hailo": Hailo-8L accelerated detector (Raspberry Pi 5)
        **kwargs: Additional arguments passed to the detector constructor
        
    Returns:
        Configured face detector instance
        
    Raises:
        ValueError: If unknown backend specified
    """
    backends = {
        "opencv": OpenCVDNNFaceDetector,
        "multiscale": MultiScaleOpenCVDetector,
        "mtcnn": MTCNNDetector,
        "retinaface": MTCNNDetector,  # Alias - uses MTCNN
        "hailo": HailoFaceDetector,
    }
    
    backend_lower = backend.lower()
    if backend_lower not in backends:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(backends.keys())}")
    
    logger.info(f"Creating face detector with backend: {backend}")
    return backends[backend_lower](**kwargs)
