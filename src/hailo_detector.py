"""
Hailo-8L Face Detection Backend for Raspberry Pi 5

This module provides hardware-accelerated face detection using the Hailo-8L
AI accelerator on Raspberry Pi 5. Expected performance: 25-30+ FPS.

Requirements:
- Raspberry Pi 5 with Hailo-8L AI HAT/Kit
- HailoRT installed (hailo_platform Python bindings)
- Pre-compiled .hef model files

Model Download:
Download RetinaFace and Age/Gender HEF models from Hailo Model Zoo:
https://github.com/hailo-ai/hailo_model_zoo


Created: 2024
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


# Check if running on Raspberry Pi with Hailo
def is_hailo_available() -> bool:
    """Check if Hailo-8L is available on this system."""
    try:
        from hailo_platform import HailoRTDevice, Device
        devices = Device.scan()
        return len(devices) > 0
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Hailo check failed: {e}")
        return False


class HailoFaceDetector:
    """
    Face detector using Hailo-8L hardware acceleration.
    
    This provides 10-30x speedup over CPU-based detection on Raspberry Pi.
    
    Typical performance on Raspberry Pi 5 + Hailo-8L:
    - RetinaFace: ~25-30 FPS at 640x640
    - SCRFD: ~40+ FPS at 640x480
    """
    
    # Default model paths
    DEFAULT_FACE_HEF = "models/hailo/retinaface_mobilenet_v1.hef"
    DEFAULT_AGE_GENDER_HEF = "models/hailo/age_gender_resnet18.hef"
    
    def __init__(
        self,
        face_hef_path: Optional[str] = None,
        age_gender_hef_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        batch_size: int = 1
    ):
        """
        Initialize Hailo face detector.
        
        Args:
            face_hef_path: Path to face detection HEF model
            age_gender_hef_path: Path to age/gender HEF model (optional)
            confidence_threshold: Minimum confidence threshold
            nms_threshold: NMS threshold for overlapping detections
            batch_size: Inference batch size
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size
        
        # Resolve paths
        project_root = Path(__file__).parent.parent
        self.face_hef_path = Path(face_hef_path) if face_hef_path else project_root / self.DEFAULT_FACE_HEF
        self.age_gender_hef_path = Path(age_gender_hef_path) if age_gender_hef_path else project_root / self.DEFAULT_AGE_GENDER_HEF
        
        # Initialize Hailo
        self._device = None
        self._face_model = None
        self._age_gender_model = None
        self._initialized = False
        
        self._initialize_hailo()
    
    def _initialize_hailo(self) -> None:
        """Initialize Hailo device and load models."""
        try:
            from hailo_platform import (
                HailoRTDevice, 
                Device, 
                HEF, 
                ConfigureParams,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType
            )
            
            # Find Hailo device
            devices = Device.scan()
            if not devices:
                raise RuntimeError("No Hailo devices found. Make sure Hailo-8L is connected.")
            
            self._device = Device(devices[0])
            logger.info(f"Connected to Hailo device: {devices[0]}")
            
            # Load face detection model
            if self.face_hef_path.exists():
                self._load_face_model()
            else:
                logger.warning(f"Face detection HEF not found: {self.face_hef_path}")
                logger.info("Download from: https://github.com/hailo-ai/hailo_model_zoo")
            
            # Load age/gender model (optional)
            if self.age_gender_hef_path.exists():
                self._load_age_gender_model()
            
            self._initialized = True
            logger.info("Hailo face detector initialized successfully")
            
        except ImportError:
            logger.error("Hailo SDK not installed. Install with: pip install hailo-platform")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise
    
    def _load_face_model(self) -> None:
        """Load the face detection HEF model."""
        from hailo_platform import HEF, ConfigureParams
        
        logger.info(f"Loading face detection model: {self.face_hef_path}")
        hef = HEF(str(self.face_hef_path))
        
        # Configure network
        configure_params = ConfigureParams.create_from_hef(hef, interface="ASYNC")
        self._face_network_group = self._device.configure(hef, configure_params)[0]
        
        # Get input/output info
        self._face_input_vstream_info = hef.get_input_vstream_infos()[0]
        self._face_output_vstream_infos = hef.get_output_vstream_infos()
        
        # Get input shape
        self._face_input_shape = self._face_input_vstream_info.shape
        logger.info(f"Face model input shape: {self._face_input_shape}")
    
    def _load_age_gender_model(self) -> None:
        """Load the age/gender estimation HEF model."""
        from hailo_platform import HEF, ConfigureParams
        
        logger.info(f"Loading age/gender model: {self.age_gender_hef_path}")
        hef = HEF(str(self.age_gender_hef_path))
        
        configure_params = ConfigureParams.create_from_hef(hef, interface="ASYNC")
        self._age_gender_network_group = self._device.configure(hef, configure_params)[0]
        
        self._age_gender_input_info = hef.get_input_vstream_infos()[0]
        self._age_gender_output_infos = hef.get_output_vstream_infos()
        
        logger.info("Age/gender model loaded successfully")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Hailo inference."""
        h, w = self._face_input_shape[1], self._face_input_shape[2]
        
        # Resize to model input size
        resized = cv2.resize(frame, (w, h))
        
        # Convert BGR to RGB if needed
        # rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize (model-specific)
        # normalized = resized.astype(np.float32) / 255.0
        
        # For RetinaFace, usually expects uint8
        return resized
    
    def _postprocess_detections(
        self, 
        outputs: Dict,
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess face detection outputs.
        
        This is model-specific. Adjust based on your HEF model's output format.
        """
        results = []
        orig_h, orig_w = original_shape
        model_h, model_w = self._face_input_shape[1], self._face_input_shape[2]
        
        # Scale factors
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        
        # Parse outputs (this varies by model)
        # RetinaFace typically outputs: scores, bboxes, landmarks
        for output_name, output_data in outputs.items():
            if 'bbox' in output_name.lower() or 'box' in output_name.lower():
                # Parse bounding boxes
                # Format depends on model, typically [N, 4] or [N, 5]
                pass
        
        # Simplified parsing for common RetinaFace format
        # Adjust based on actual model output format
        try:
            # Find detection outputs
            scores = None
            boxes = None
            
            for name, data in outputs.items():
                data = np.array(data)
                if len(data.shape) >= 2:
                    if data.shape[-1] == 1 or 'conf' in name.lower() or 'score' in name.lower():
                        scores = data.flatten()
                    elif data.shape[-1] >= 4:
                        boxes = data.reshape(-1, data.shape[-1])
            
            if scores is not None and boxes is not None:
                for i, (score, box) in enumerate(zip(scores, boxes)):
                    if score < self.confidence_threshold:
                        continue
                    
                    # Scale to original image size
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    
                    # Clamp to bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(orig_w, x2)
                    y2 = min(orig_h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        results.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": float(score)
                        })
            
        except Exception as e:
            logger.error(f"Postprocess error: {e}")
        
        return results
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using Hailo-8L accelerator.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detection dictionaries with 'bbox' and 'confidence'
        """
        if not self._initialized or self._face_network_group is None:
            logger.warning("Hailo detector not initialized")
            return []
        
        if frame is None or frame.size == 0:
            return []
        
        original_shape = frame.shape[:2]
        
        try:
            from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams
            
            # Preprocess
            input_data = self._preprocess_frame(frame)
            
            # Create vstream params
            input_params = InputVStreamParams.make_from_network_group(
                self._face_network_group
            )
            output_params = OutputVStreamParams.make_from_network_group(
                self._face_network_group
            )
            
            # Run inference
            with InferVStreams(
                self._face_network_group,
                input_params,
                output_params
            ) as vstreams:
                input_name = self._face_input_vstream_info.name
                inputs = {input_name: input_data[np.newaxis, ...]}
                
                outputs = vstreams.infer(inputs)
            
            # Postprocess
            detections = self._postprocess_detections(outputs, original_shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"Hailo inference error: {e}")
            return []
    
    def get_backend_name(self) -> str:
        return "Hailo_8L"
    
    def __del__(self):
        """Cleanup Hailo resources."""
        if self._device:
            try:
                self._device.release()
            except:
                pass


class HailoAgeGenderClassifier:
    """
    Age and gender classifier using Hailo-8L acceleration.
    
    This provides fast age/gender estimation to pair with face detection.
    """
    
    def __init__(self, hef_path: Optional[str] = None):
        """Initialize Hailo age/gender classifier."""
        self.hef_path = hef_path
        self._initialized = False
        
        # TODO: Implement similar to HailoFaceDetector
        logger.info("HailoAgeGenderClassifier initialized (placeholder)")
    
    def classify(self, face_crop: np.ndarray) -> Dict:
        """Classify age and gender from face crop."""
        # Placeholder - implement with actual Hailo inference
        return {
            "age": 25,
            "gender": "Unknown",
            "age_confidence": 0.0,
            "gender_confidence": 0.0
        }


def download_hailo_models(output_dir: str = "models/hailo") -> None:
    """
    Download pre-compiled HEF models for face detection and age/gender.
    
    Models are from Hailo Model Zoo:
    https://github.com/hailo-ai/hailo_model_zoo
    """
    import urllib.request
    import os
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Hailo Model Zoo URLs (these may need updating)
    models = {
        "retinaface_mobilenet_v1.hef": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.8.0/retinaface_mobilenet_v1.hef",
        "scrfd_2.5g.hef": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.8.0/scrfd_2.5g.hef",
    }
    
    print("Downloading Hailo models...")
    for filename, url in models.items():
        filepath = output_path / filename
        if filepath.exists():
            print(f"  {filename} already exists, skipping")
            continue
        
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ {filename} downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
    
    print("\nNote: Age/gender HEF models may require custom compilation.")
    print("See: https://github.com/hailo-ai/hailo_model_zoo for instructions.")


# Utility function to check system readiness
def check_hailo_setup() -> Dict:
    """
    Check if the system is ready for Hailo acceleration.
    
    Returns:
        Dictionary with setup status and recommendations
    """
    status = {
        "hailo_available": False,
        "hailo_sdk_installed": False,
        "device_found": False,
        "models_available": False,
        "recommendations": []
    }
    
    # Check SDK
    try:
        import hailo_platform
        status["hailo_sdk_installed"] = True
    except ImportError:
        status["recommendations"].append(
            "Install Hailo SDK: pip install hailo-platform"
        )
    
    # Check device
    if status["hailo_sdk_installed"]:
        try:
            from hailo_platform import Device
            devices = Device.scan()
            if devices:
                status["device_found"] = True
                status["hailo_available"] = True
            else:
                status["recommendations"].append(
                    "No Hailo device found. Make sure Hailo-8L HAT is connected."
                )
        except Exception as e:
            status["recommendations"].append(f"Device check failed: {e}")
    
    # Check models
    models_dir = Path(__file__).parent.parent / "models" / "hailo"
    if models_dir.exists() and any(models_dir.glob("*.hef")):
        status["models_available"] = True
    else:
        status["recommendations"].append(
            "Download HEF models using: python -c 'from src.hailo_detector import download_hailo_models; download_hailo_models()'"
        )
    
    return status


if __name__ == "__main__":
    # Test Hailo setup
    print("Checking Hailo-8L setup...")
    status = check_hailo_setup()
    
    print("\n=== Hailo-8L Status ===")
    print(f"SDK Installed: {'✓' if status['hailo_sdk_installed'] else '✗'}")
    print(f"Device Found: {'✓' if status['device_found'] else '✗'}")
    print(f"Models Available: {'✓' if status['models_available'] else '✗'}")
    print(f"Ready: {'✓' if status['hailo_available'] else '✗'}")
    
    if status["recommendations"]:
        print("\nRecommendations:")
        for rec in status["recommendations"]:
            print(f"  - {rec}")
