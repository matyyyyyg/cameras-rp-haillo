from typing import List, Dict, Tuple, Optional
from pathlib import Path
import contextlib
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def is_hailo_available() -> bool:
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

    # Valid aspect ratio range for faces (width/height)
    MIN_ASPECT_RATIO = 0.5
    MAX_ASPECT_RATIO = 1.5

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        device_id: str = "0",
        min_face_size: int = 45 #changed from 20 to 35 because we want to detect only faces that are closer than 5m
    ):
        """
        Initialize Hailo face detector.

        Args:
            model_path: Path to .hef model file
            confidence_threshold: Minimum confidence (0.0-1.0)
            nms_threshold: NMS threshold for overlapping boxes
            device_id: Hailo device ID (usually "0")
            min_face_size: Minimum face width/height in pixels
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device_id = device_id
        self.min_face_size = min_face_size

        # Resolve model path relative to face_analysis_NEW root
        project_root = Path(__file__).parent.parent.parent
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
        self._model_type_logged = False

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

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"HEF model not found: {self.model_path}\n"
                    f"Please download the model to: {self.model_path}"
                )

            logger.info(f"Loading HEF: {self.model_path}")
            self._hef = HEF(str(self.model_path))

            self._vdevice = VDevice()

            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            )
            network_groups = self._vdevice.configure(self._hef, configure_params)
            self._network_group = network_groups[0]

            self._input_vstream_info = self._hef.get_input_vstream_infos()[0]
            self._output_vstream_infos = self._hef.get_output_vstream_infos()

            self._input_vstream_params = InputVStreamParams.make(self._network_group)
            self._output_vstream_params = OutputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )

            self._exit_stack = contextlib.ExitStack()
            try:
                network_group_context = self._network_group.activate()
                self._exit_stack.enter_context(network_group_context)

                self._infer_pipeline = InferVStreams(
                    self._network_group,
                    self._input_vstream_params,
                    self._output_vstream_params
                )
                self._exit_stack.enter_context(self._infer_pipeline)
            except Exception:
                self._exit_stack.close()
                raise

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
        if not self._initialized:
            raise RuntimeError("Hailo detector not initialized")

        input_data = self._preprocess(image)
        outputs = self._infer(input_data)
        detections = self._postprocess(outputs, image.shape)

        logger.debug(f"Detected {len(detections)} faces")
        return detections

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Hailo inference."""
        resized = cv2.resize(image, (self._input_width, self._input_height))
        if self._input_channels == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return resized

    def _infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on Hailo device using persistent VStreams pipeline."""
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        input_dict = {self._input_name: input_data}

        try:
            return self._infer_pipeline.infer(input_dict)
        except Exception as e:
            logger.warning(f"Hailo inference failed, recreating pipeline: {e}")
            try:
                from hailo_platform import InferVStreams
                # Close old exit stack and create fresh one for new pipeline
                if getattr(self, '_exit_stack', None):
                    self._exit_stack.close()
                self._exit_stack = contextlib.ExitStack()
                network_group_context = self._network_group.activate()
                self._exit_stack.enter_context(network_group_context)
                self._infer_pipeline = InferVStreams(
                    self._network_group,
                    self._input_vstream_params,
                    self._output_vstream_params
                )
                self._exit_stack.enter_context(self._infer_pipeline)
                return self._infer_pipeline.infer(input_dict)
            except Exception as e2:
                logger.error(f"Hailo inference retry failed: {e2}")
                raise

    def _postprocess(self, outputs: Dict[str, np.ndarray], original_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Postprocess face detection outputs with auto-detection of model type.

        Both RetinaFace and SCRFD output 9 tensors: 3 FPN levels x 3 types each.
        Auto-detection uses classification channel count:
          - 4 channels = RetinaFace (2 anchors * 2 classes, softmax)
        """
        min_sizes_per_level = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        variance = [0.1, 0.2]

        if not self._model_type_logged:
            for name, tensor in outputs.items():
                t = tensor.squeeze(0) if tensor.ndim == 4 else tensor
                logger.info(f"  Output tensor: {name} shape={t.shape} "
                            f"min={t.min():.4f} max={t.max():.4f}")
                if t.shape[-1] in (2, 4):
                    raw_max = t.max()
                    sig_max = 1.0 / (1.0 + np.exp(-raw_max))
                    logger.info(f"    ^ cls tensor: raw_max={raw_max:.4f}, sigmoid(raw_max)={sig_max:.4f}")

        levels = {}
        cls_channels = None
        for name, tensor in outputs.items():
            if tensor.ndim == 4:
                tensor = tensor.squeeze(0)
            h, w, c = tensor.shape
            key = (h, w)
            if key not in levels:
                levels[key] = {}
            if c in (2, 4):
                levels[key]['cls'] = tensor
                cls_channels = c
            elif c == 8:
                levels[key]['bbox'] = tensor
            elif c == 20:
                levels[key]['lm'] = tensor

        is_scrfd = (cls_channels == 2)
        if not self._model_type_logged:
            self._model_type_logged = True
            if is_scrfd:
                logger.info("Auto-detected SCRFD model (cls channels=2, sigmoid)")
            else:
                logger.info("Auto-detected RetinaFace model (cls channels=4, softmax)")

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

            cols = np.arange(feat_w, dtype=np.float32)
            rows = np.arange(feat_h, dtype=np.float32)
            col_grid, row_grid = np.meshgrid(cols, rows)
            cx = (col_grid + 0.5) * step
            cy = (row_grid + 0.5) * step

            for anchor_idx, anchor_size in enumerate(min_sizes):
                if is_scrfd:
                    raw_scores = cls_data[:, :, anchor_idx]
                    if raw_scores.max() > 1.0 or raw_scores.min() < 0.0:
                        scores = 1.0 / (1.0 + np.exp(-raw_scores))
                    else:
                        scores = raw_scores
                else:
                    bg = cls_data[:, :, anchor_idx * 2]
                    fg = cls_data[:, :, anchor_idx * 2 + 1]
                    max_val = np.maximum(bg, fg)
                    exp_bg = np.exp(bg - max_val)
                    exp_fg = np.exp(fg - max_val)
                    scores = exp_fg / (exp_bg + exp_fg)

                mask = scores >= self.confidence_threshold
                if not np.any(mask):
                    continue

                scores_f = scores[mask]
                cx_f = cx[mask]
                cy_f = cy[mask]

                off = anchor_idx * 4
                d0 = bbox_data[:, :, off][mask]
                d1 = bbox_data[:, :, off + 1][mask]
                d2 = bbox_data[:, :, off + 2][mask]
                d3 = bbox_data[:, :, off + 3][mask]

                if is_scrfd:
                    x1 = cx_f - d0 * step
                    y1 = cy_f - d1 * step
                    x2 = cx_f + d2 * step
                    y2 = cy_f + d3 * step
                else:
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

        valid = (all_boxes[:, 2] > all_boxes[:, 0]) & (all_boxes[:, 3] > all_boxes[:, 1])
        if not np.any(valid):
            return []
        all_boxes = all_boxes[valid]
        all_scores = all_scores[valid]
        if all_landmarks is not None:
            all_landmarks = all_landmarks[valid]

        keep = self._nms(all_boxes, all_scores, self.nms_threshold)

        detections = []
        filtered_count = 0
        for i in keep:
            x1, y1, x2, y2 = all_boxes[i]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            if w <= 0 or h <= 0:
                continue

            if w < self.min_face_size or h < self.min_face_size:
                filtered_count += 1
                continue

            aspect_ratio = w / max(h, 1)
            if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
                filtered_count += 1
                continue

            det = {
                'box': (x, y, w, h),
                'confidence': float(all_scores[i])
            }
            if all_landmarks is not None:
                lm = all_landmarks[i]
                det['landmarks'] = [(int(lm[j]), int(lm[j+1])) for j in range(0, 10, 2)]
            detections.append(det)

        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} detections (size/aspect ratio)")

        if len(detections) == 0 and len(all_scores) > 0:
            logger.debug(f"No detections passed filters (pre-NMS: {len(all_scores)}, post-NMS: {len(keep)}, filtered: {filtered_count})")

        return detections

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
        """Release Hailo resources safely via ExitStack."""
        if getattr(self, '_exit_stack', None):
            self._exit_stack.close()
            self._exit_stack = None

        self._network_group = None
        self._hef = None
        self._vdevice = None
        self._initialized = False
        logger.info("Hailo face detector cleaned up")

    def __del__(self):
        self.cleanup()


def create_hailo_face_detector(
    model_path: Optional[str] = None,
    confidence: float = 0.4
) -> UnifiedHailoFaceDetector:
    """Create a Hailo face detector instance."""
    return UnifiedHailoFaceDetector(
        model_path=model_path,
        confidence_threshold=confidence
    )
