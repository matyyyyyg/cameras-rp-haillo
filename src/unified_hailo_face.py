"""
Hailo-8 hardware accelerated face detector.
Supports RetinaFace and SCRFD models on Raspberry Pi 5.
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

def is_hailo_available() -> bool:
    """Check if Hailo-8 is available on Raspberry Pi 5."""
    try:
        from hailo_platform import VDevice, Device

        devices = Device.scan()
        if len(devices) == 0:
            logger.warning("No Hailo devices detected")
            return False

        logger.info(f"Detected {len(devices)} Hailo device(s): {devices}")

        # Verify we can actually open the device
        vdevice = VDevice()
        del vdevice

        return True

    except ImportError as e:
        logger.warning(f"Hailo platform not installed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Hailo check failed: {e}")
        return False


class UnifiedHailoFaceDetector:
    """
    Unified face detector using Hailo-8 hardware acceleration.

    This detector:
    1. Uses RetinaFace or SCRFD on Hailo-8 for face detection
    2. Returns bounding boxes of detected faces
    3. Can be paired with ANY gender classifier for fair comparison

    Typical Performance (Raspberry Pi 5 + Hailo-8):
    - RetinaFace: 25-30 FPS (best accuracy for CCTV)
    - SCRFD: 40+ FPS (faster, good for real-time)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device_id: str = "0"
    ):
        """
        Initialize Hailo face detector.

        Args:
            model_path: Path to .hef model file
                       Default: models/hailo/retinaface_mobilenet_v1.hef
            confidence_threshold: Minimum confidence (0.0-1.0)
            nms_threshold: NMS threshold for overlapping boxes
            device_id: Hailo device ID (usually "0")
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device_id = device_id

        # Resolve model path
        project_root = Path(__file__).parent.parent
        if model_path is None:
            self.model_path = project_root / "models" / "hailo" / "retinaface_mobilenet_v1.hef"
        else:
            self.model_path = Path(model_path)

        # Hailo components
        self._vdevice = None
        self._hef = None
        self._network_group = None
        self._network_group_params = None
        self._input_vstream_info = None
        self._output_vstream_infos = None
        self._initialized = False

        # Model info
        self._input_height = None
        self._input_width = None
        self._input_channels = None
        self._input_name = None

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Hailo device and load face detection model."""
        if not is_hailo_available():
            raise RuntimeError(
                "Hailo-8 device not found on this system. "
                "Please ensure:\n"
                "1. Raspberry Pi 5 with Hailo-8 AI HAT is connected\n"
                "2. HailoRT is installed: sudo apt install hailo-all\n"
                "3. Driver is loaded: lsmod | grep hailo\n"
            )

        try:
            from hailo_platform import (
                HEF, VDevice, HailoStreamInterface,
                ConfigureParams, InputVStreamParams,
                OutputVStreamParams, FormatType,
                InferVStreams
            )

            logger.info("Initializing Hailo-8 face detector...")

            # Check if HEF file exists
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"HEF model not found: {self.model_path}\n"
                    f"Please run: bash download_hailo_models.sh\n"
                    f"This will download RetinaFace and SCRFD models from Hailo Model Zoo"
                )

            logger.info(f"Loading HEF: {self.model_path}")
            self._hef = HEF(str(self.model_path))

            # Create VDevice
            self._vdevice = VDevice()

            # Configure network group from HEF
            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            )
            network_groups = self._vdevice.configure(self._hef, configure_params)
            self._network_group = network_groups[0]

            # Get stream info from HEF
            self._input_vstream_info = self._hef.get_input_vstream_infos()[0]
            self._output_vstream_infos = self._hef.get_output_vstream_infos()

            # Build vstream params with float32 output format
            self._input_vstream_params = InputVStreamParams.make(self._network_group)
            self._output_vstream_params = OutputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )

            # Activate network group (must stay active for inference)
            self._network_group_context = self._network_group.activate()
            self._network_group_context.__enter__()

            # Create persistent InferVStreams pipeline (same pattern as network group context)
            self._infer_pipeline = InferVStreams(
                self._network_group,
                self._input_vstream_params,
                self._output_vstream_params
            )
            self._infer_pipeline.__enter__()

            # Parse input shape
            self._input_name = self._input_vstream_info.name
            shape = self._input_vstream_info.shape
            if len(shape) == 3:
                self._input_height, self._input_width, self._input_channels = shape
            elif len(shape) == 4:
                _, self._input_height, self._input_width, self._input_channels = shape
            else:
                raise ValueError(f"Unexpected input shape: {shape}")

            self._initialized = True

            logger.info(f"Hailo face detector initialized successfully")
            logger.info(f"   Model: {self.model_path.name}")
            logger.info(f"   Input: {self._input_name} shape: {self._input_height}x{self._input_width}x{self._input_channels}")
            logger.info(f"   Outputs: {[info.name for info in self._output_vstream_infos]}")
            logger.info(f"   Confidence threshold: {self.confidence_threshold}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import Hailo platform: {e}\n"
                f"Please install HailoRT: sudo apt install hailo-all"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Hailo face detector: {e}")

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using Hailo-8 acceleration.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            List of detected faces, each dict containing:
            {
                'box': (x, y, w, h),
                'confidence': float,
                'landmarks': [(x,y), ...] (if available)
            }
        """
        if not self._initialized:
            raise RuntimeError("Hailo detector not initialized")

        if cv2 is None:
            raise ImportError("OpenCV not installed. Install: pip install opencv-python")

        # Preprocess image
        input_data = self._preprocess(image)

        # Run inference on Hailo
        outputs = self._infer(input_data)

        # Postprocess results
        detections = self._postprocess(outputs, image.shape)

        logger.debug(f"Detected {len(detections)} faces")
        return detections

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for Hailo inference.

        Args:
            image: BGR image from OpenCV

        Returns:
            Preprocessed image ready for Hailo
        """
        # Resize to model input size
        resized = cv2.resize(image, (self._input_width, self._input_height))

        # Convert BGR to RGB if needed
        if self._input_channels == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Models expect uint8 input (0-255)
        # No normalization needed - Hailo models handle this internally
        return resized

    def _infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on Hailo device using persistent VStreams pipeline.

        Args:
            input_data: Preprocessed image (H, W, C) uint8

        Returns:
            Dictionary of output tensors (float32)
        """
        # Add batch dimension: (H,W,C) -> (1,H,W,C)
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        input_dict = {self._input_name: input_data}

        try:
            return self._infer_pipeline.infer(input_dict)
        except Exception as e:
            logger.warning(f"Hailo inference failed, recreating pipeline: {e}")
            # Tear down and recreate the pipeline
            try:
                self._infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
            try:
                from hailo_platform import InferVStreams
                self._infer_pipeline = InferVStreams(
                    self._network_group,
                    self._input_vstream_params,
                    self._output_vstream_params
                )
                self._infer_pipeline.__enter__()
                return self._infer_pipeline.infer(input_dict)
            except Exception as e2:
                logger.error(f"Hailo inference retry failed: {e2}")
                logger.error(f"Input shape: {input_data.shape}, dtype: {input_data.dtype}")
                raise

    def _postprocess(self, outputs: Dict[str, np.ndarray], original_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Postprocess face detection outputs with auto-detection of model type.

        Both RetinaFace and SCRFD output 9 tensors: 3 FPN levels x 3 types each.
        Auto-detection uses classification channel count:
          - 4 channels = RetinaFace (2 anchors * 2 classes, softmax)
          - 2 channels = SCRFD (2 anchors * 1 class, sigmoid)
        Common tensors:
          - 8 channels  = bbox regression (2 anchors * 4 coords)
          - 20 channels = landmarks (2 anchors * 5 points * 2 coords)

        Args:
            outputs: Raw output tensors from Hailo
            original_shape: Original image shape (H, W, C)

        Returns:
            List of face detections
        """
        try:
            # Shared anchor configuration
            min_sizes_per_level = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
            variance = [0.1, 0.2]  # Used by RetinaFace only

            # Log all tensor shapes on first call for diagnostics
            if not hasattr(self, '_postprocess_logged'):
                self._postprocess_logged = True
                for name, tensor in outputs.items():
                    t = tensor.squeeze(0) if tensor.ndim == 4 else tensor
                    logger.info(f"  Output tensor: {name} shape={t.shape} "
                                f"min={t.min():.4f} max={t.max():.4f}")
                    # For cls tensors, show what sigmoid would give
                    if t.shape[-1] in (2, 4):
                        raw_max = t.max()
                        sig_max = 1.0 / (1.0 + np.exp(-raw_max))
                        logger.info(f"    ^ cls tensor: raw_max={raw_max:.4f}, sigmoid(raw_max)={sig_max:.4f}")

            # Group output tensors by spatial dimensions (each FPN level)
            levels = {}
            cls_channels = None
            for name, tensor in outputs.items():
                if tensor.ndim == 4:
                    tensor = tensor.squeeze(0)
                h, w, c = tensor.shape
                key = (h, w)
                if key not in levels:
                    levels[key] = {}
                # Identify tensor type by channel count
                if c in (2, 4):
                    levels[key]['cls'] = tensor
                    cls_channels = c
                elif c == 8:
                    levels[key]['bbox'] = tensor
                elif c == 20:
                    levels[key]['lm'] = tensor

            # Auto-detect model type from classification channels
            is_scrfd = (cls_channels == 2)
            if is_scrfd:
                logger.info("Auto-detected SCRFD model (cls channels=2, sigmoid)")
            else:
                logger.info("Auto-detected RetinaFace model (cls channels=4, softmax)")

            # Sort levels largest-first (stride 8 first, stride 32 last)
            sorted_levels = sorted(levels.items(), key=lambda x: x[0][0] * x[0][1], reverse=True)

            all_boxes = []
            all_scores = []
            all_landmarks = []

            for level_idx, ((feat_h, feat_w), tensors) in enumerate(sorted_levels):
                if 'cls' not in tensors or 'bbox' not in tensors:
                    continue

                step = steps[level_idx]
                min_sizes = min_sizes_per_level[level_idx]

                cls_data = tensors['cls']
                bbox_data = tensors['bbox']
                lm_data = tensors.get('lm')

                # Generate anchor grid centers
                cols = np.arange(feat_w, dtype=np.float32)
                rows = np.arange(feat_h, dtype=np.float32)
                col_grid, row_grid = np.meshgrid(cols, rows)
                cx = (col_grid + 0.5) * step  # (H, W)
                cy = (row_grid + 0.5) * step  # (H, W)

                for anchor_idx, anchor_size in enumerate(min_sizes):
                    # --- Score extraction ---
                    if is_scrfd:
                        # SCRFD: 1 class per anchor
                        # Hailo HEF outputs are already sigmoid-activated
                        scores = cls_data[:, :, anchor_idx]
                    else:
                        # RetinaFace: 2 classes per anchor, softmax activation
                        bg = cls_data[:, :, anchor_idx * 2]
                        fg = cls_data[:, :, anchor_idx * 2 + 1]
                        max_val = np.maximum(bg, fg)
                        exp_bg = np.exp(bg - max_val)
                        exp_fg = np.exp(fg - max_val)
                        scores = exp_fg / (exp_bg + exp_fg)

                    # Early filter by confidence
                    mask = scores >= self.confidence_threshold
                    if not np.any(mask):
                        continue

                    scores_f = scores[mask]
                    cx_f = cx[mask]
                    cy_f = cy[mask]

                    # --- Bounding box decoding ---
                    off = anchor_idx * 4
                    d0 = bbox_data[:, :, off][mask]
                    d1 = bbox_data[:, :, off + 1][mask]
                    d2 = bbox_data[:, :, off + 2][mask]
                    d3 = bbox_data[:, :, off + 3][mask]

                    if is_scrfd:
                        # SCRFD: distance-to-boundary (FCOS-style)
                        x1 = cx_f - d0 * step
                        y1 = cy_f - d1 * step
                        x2 = cx_f + d2 * step
                        y2 = cy_f + d3 * step
                    else:
                        # RetinaFace: center-size with variance
                        pred_cx = cx_f + d0 * variance[0] * anchor_size
                        pred_cy = cy_f + d1 * variance[0] * anchor_size
                        pred_w = anchor_size * np.exp(d2 * variance[1])
                        pred_h = anchor_size * np.exp(d3 * variance[1])
                        x1 = pred_cx - pred_w / 2
                        y1 = pred_cy - pred_h / 2
                        x2 = pred_cx + pred_w / 2
                        y2 = pred_cy + pred_h / 2

                    boxes = np.stack([x1, y1, x2, y2], axis=1)
                    all_boxes.append(boxes)
                    all_scores.append(scores_f)

                    # --- Landmark decoding ---
                    if lm_data is not None:
                        lm_off = anchor_idx * 10
                        landmarks = np.zeros((len(scores_f), 10), dtype=np.float32)
                        for pt in range(5):
                            lx = lm_data[:, :, lm_off + pt * 2][mask]
                            ly = lm_data[:, :, lm_off + pt * 2 + 1][mask]
                            if is_scrfd:
                                landmarks[:, pt * 2] = cx_f + lx * step
                                landmarks[:, pt * 2 + 1] = cy_f + ly * step
                            else:
                                landmarks[:, pt * 2] = cx_f + lx * variance[0] * anchor_size
                                landmarks[:, pt * 2 + 1] = cy_f + ly * variance[0] * anchor_size
                        all_landmarks.append(landmarks)

            if not all_boxes:
                return []

            all_boxes = np.concatenate(all_boxes, axis=0)
            all_scores = np.concatenate(all_scores, axis=0)
            if all_landmarks:
                all_landmarks = np.concatenate(all_landmarks, axis=0)
            else:
                all_landmarks = None

            logger.debug(f"Pre-NMS candidates: {len(all_scores)}")

            # Scale from model input coords to original image coords
            orig_h, orig_w = original_shape[:2]
            scale_x = orig_w / self._input_width
            scale_y = orig_h / self._input_height

            all_boxes[:, 0] *= scale_x
            all_boxes[:, 1] *= scale_y
            all_boxes[:, 2] *= scale_x
            all_boxes[:, 3] *= scale_y

            if all_landmarks is not None:
                all_landmarks[:, 0::2] *= scale_x
                all_landmarks[:, 1::2] *= scale_y

            # Filter out degenerate boxes (x1 >= x2 or y1 >= y2)
            valid = (all_boxes[:, 2] > all_boxes[:, 0]) & (all_boxes[:, 3] > all_boxes[:, 1])
            if not np.any(valid):
                return []
            all_boxes = all_boxes[valid]
            all_scores = all_scores[valid]
            if all_landmarks is not None:
                all_landmarks = all_landmarks[valid]

            # NMS
            keep = self._nms(all_boxes, all_scores, self.nms_threshold)

            detections = []
            for i in keep:
                x1, y1, x2, y2 = all_boxes[i]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                det = {
                    'box': (x, y, w, h),
                    'confidence': float(all_scores[i])
                }
                if all_landmarks is not None:
                    lm = all_landmarks[i]
                    det['landmarks'] = [(int(lm[j]), int(lm[j+1])) for j in range(0, 10, 2)]
                detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            logger.debug(f"Output keys: {list(outputs.keys())}")
            logger.debug(f"Output shapes: {[(k, v.shape) for k, v in outputs.items()]}")
            return []

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Vectorized Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def cleanup(self) -> None:
        """Release Hailo resources."""
        try:
            if getattr(self, '_infer_pipeline', None):
                try:
                    self._infer_pipeline.__exit__(None, None, None)
                except:
                    pass
            if getattr(self, '_network_group_context', None):
                try:
                    self._network_group_context.__exit__(None, None, None)
                except:
                    pass
            if getattr(self, '_network_group', None):
                try:
                    del self._network_group
                except:
                    pass
            if getattr(self, '_hef', None):
                try:
                    del self._hef
                except:
                    pass
            if getattr(self, '_vdevice', None):
                try:
                    del self._vdevice
                except:
                    pass

            self._initialized = False
            logger.info("Hailo face detector cleaned up")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Convenience function
def create_hailo_face_detector(
    model_path: Optional[str] = None,
    confidence: float = 0.5
) -> UnifiedHailoFaceDetector:
    """
    Create a Hailo face detector instance.

    Args:
        model_path: Path to .hef file (optional)
        confidence: Detection confidence threshold

    Returns:
        UnifiedHailoFaceDetector instance
    """
    return UnifiedHailoFaceDetector(
        model_path=model_path,
        confidence_threshold=confidence
    )


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Unified Hailo Face Detector Test")
    print("=" * 60)

    # Check Hailo availability
    if not is_hailo_available():
        print("Hailo-8 not available on this system")
        print("\nThis test requires:")
        print("- Raspberry Pi 5 with Hailo-8 AI HAT")
        print("- HailoRT installed")
        exit(1)

    print("Hailo-8 detected")

    # Initialize detector
    try:
        detector = create_hailo_face_detector()
        print(f"\nDetector initialized: {detector.model_path.name}")
        print(f"   Input size: {detector._input_height}x{detector._input_width}")

    except Exception as e:
        print(f"\nFailed to initialize detector: {e}")
        exit(1)

    # Test with sample image (if available)
    test_image_path = Path(__file__).parent.parent / "test_image.jpg"

    if test_image_path.exists() and cv2 is not None:
        print(f"\nTesting with image: {test_image_path}")

        image = cv2.imread(str(test_image_path))
        if image is not None:
            faces = detector.detect_faces(image)
            print(f"   Detected {len(faces)} faces")

            for i, face in enumerate(faces):
                x, y, w, h = face['box']
                conf = face['confidence']
                print(f"   Face {i+1}: ({x}, {y}, {w}, {h}) confidence={conf:.3f}")
        else:
            print("   Failed to load test image")
    else:
        print(f"\nNo test image found at {test_image_path}")
        print("   Detector initialized successfully but not tested")

    # Cleanup
    detector.cleanup()
    print("\nTest complete")
