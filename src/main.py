"""
CCTV Analytics Pipeline - Main Entrypoint

This module provides the main video processing pipeline that:
1. Captures video from webcam, file, RTSP stream, or Pi Camera
2. Detects faces using OpenCV DNN
3. Classifies age and gender for each face
4. Tracks persons across frames
5. Logs all detections to CSV/JSONL

Usage:
    python -m src.main                          # Use Pi Camera (default)
    python -m src.main --source 0               # Use USB webcam
    python -m src.main --source video.mp4       # Use video file
    python -m src.main --source rtsp://...      # Use RTSP stream
    python -m src.main --source picam           # Explicitly use Pi Camera


Created: 2024
"""

import argparse
import time
import sys
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .detectors import create_detector, BaseFaceDetector
from .classification import AgeGenderClassifier, extract_face_crop
from .tracking import SimpleTracker
from .logging_utils import DetectionLogger, DetectionEvent, setup_logging

logger = logging.getLogger(__name__)

# Try to import picamera2
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    logger.info("picamera2 available")
except ImportError:
    logger.info("picamera2 not available, Pi Camera support disabled")


class CCTVAnalyticsPipeline:
    """
    Main CCTV analytics pipeline.
    
    Orchestrates video capture, face detection, classification,
    tracking, and logging into a unified processing loop.
    """
    
    def __init__(
        self,
        source: Union[int, str] = "picam",
        camera_id: str = "cam_01",
        detection_backend: str = "retinaface",
        detection_confidence: float = 0.3,
        log_path: str = "logs/detections.csv",
        enable_jsonl: bool = False,
        display: bool = True,
        log_interval: int = 1,
        enhance_image: bool = True
    ):
        """
        Initialize the analytics pipeline.
        
        Args:
            source: Video source ("picam" for Pi Camera, 0 for webcam, path for file/RTSP)
            camera_id: Identifier for this camera
            detection_backend: Face detection backend ("opencv", "multiscale", "retinaface", "hailo")
            detection_confidence: Minimum confidence for face detection
            log_path: Path to CSV log file
            enable_jsonl: Also log to JSONL format
            display: Show video with annotations
            log_interval: Log every N frames (1 = every frame)
            enhance_image: Apply image enhancement for better CCTV detection
        """
        self.source = source
        self.camera_id = camera_id
        self.display = display
        self.log_interval = log_interval
        
        # Initialize components
        logger.info(f"Initializing pipeline for camera: {camera_id}")
        
        # Face detector with enhancement options
        self.detector = create_detector(
            backend=detection_backend,
            confidence_threshold=detection_confidence,
            enhance_image=enhance_image
        )
        
        # Age/Gender classifier
        self.classifier = AgeGenderClassifier()
        
        # Tracker
        self.tracker = SimpleTracker(camera_id=camera_id)
        
        # Logger
        self.logger = DetectionLogger(
            csv_path=log_path,
            enable_jsonl=enable_jsonl
        )
        
        # Video capture (initialized on run)
        self.cap: Optional[cv2.VideoCapture] = None
        self.picam: Optional[object] = None  # Picamera2 instance
        self.use_picamera = False
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time: Optional[float] = None
        
        logger.info("Pipeline initialized successfully")
    
    def _init_capture(self) -> bool:
        """Initialize video capture (OpenCV or Pi Camera)."""
        
        # Check if Pi Camera requested
        if self.source == "picam" or self.source == "picamera":
            return self._init_picamera()
        
        # Try OpenCV VideoCapture for other sources
        logger.info(f"Opening video source: {self.source}")
        
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")
            # Fallback to Pi Camera if available
            if PICAMERA2_AVAILABLE:
                logger.info("Falling back to Pi Camera...")
                return self._init_picamera()
            return False
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video source opened: {width}x{height} @ {fps:.1f} FPS")
        return True
    
    def _init_picamera(self) -> bool:
        """Initialize Pi Camera using picamera2."""
        if not PICAMERA2_AVAILABLE:
            logger.error("picamera2 not available. Install with: sudo apt install python3-picamera2")
            return False
        
        try:
            logger.info("Initializing Pi Camera...")
            self.picam = Picamera2()
            
            # Configure for video capture (BGR format for OpenCV compatibility)
            config = self.picam.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
            self.picam.configure(config)
            self.picam.start()
            
            self.use_picamera = True
            logger.info("Pi Camera initialized: 1280x720 @ 30 FPS")
            
            # Give camera time to warm up
            time.sleep(1)
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pi Camera: {e}")
            return False
    
    def _read_frame(self):
        """Read a frame from the appropriate source."""
        if self.use_picamera and self.picam:
            # Read from Pi Camera
            frame = self.picam.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        elif self.cap:
            # Read from OpenCV VideoCapture
            return self.cap.read()
        else:
            return False, None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: BGR frame from video capture
            
        Returns:
            Annotated frame for display
        """
        self.frame_count += 1
        
        # 1. Detect faces
        detections = self.detector.detect_faces(frame)
        
        # 2. Track faces
        tracked = self.tracker.update(detections)
        
        # 3. Classify and log each tracked face
        events = []
        for det in tracked:
            bbox = det["bbox"]
            person_id = det["person_id"]
            det_conf = det["confidence"]
            
            # Extract face crop
            face_crop = extract_face_crop(frame, bbox, padding=0.2)
            
            # Classify age and gender
            result = self.classifier.classify(face_crop)
            
            # Create detection event
            event = DetectionEvent.create(
                person_id=person_id,
                camera_id=self.camera_id,
                age_midpoint=result.age_midpoint,
                age_bucket=result.age_bucket,
                age_confidence=result.age_confidence,
                gender=result.gender,
                gender_confidence=result.gender_confidence,
                bbox=bbox,
                detection_confidence=det_conf
            )
            events.append(event)
            
            # Draw annotations on frame
            if self.display:
                frame = self._draw_annotation(
                    frame, bbox, person_id,
                    result.age_bucket, result.gender
                )
        
        # Log events (respecting log_interval)
        if events and self.frame_count % self.log_interval == 0:
            self.logger.log_batch(events)
            self.detection_count += len(events)
        
        # Draw FPS on frame
        if self.display:
            frame = self._draw_fps(frame)
        
        return frame
    
    def _draw_annotation(
        self,
        frame: np.ndarray,
        bbox: tuple,
        person_id: str,
        age: str,
        gender: str
    ) -> np.ndarray:
        """Draw bounding box and labels on frame."""
        x1, y1, x2, y2 = bbox
        
        # Colors
        color = (0, 255, 0)  # Green
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{gender}, {age}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Draw person ID below box
        id_label = person_id.split("_")[-1]  # Just show number
        cv2.putText(frame, f"ID: {id_label}", (x1, y2 + 15), font, 0.4, color, 1)
        
        return frame
    
    def _draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter on frame."""
        if self.start_time and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
    
    def run(self) -> None:
        """Run the main processing loop."""
        if not self._init_capture():
            return
        
        self.start_time = time.time()
        logger.info("Starting processing loop. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = self._read_frame()
                
                if not ret or frame is None:
                    if isinstance(self.source, str) and self.source not in ["picam", "picamera"]:
                        logger.info("End of video file reached")
                    else:
                        logger.error("Failed to read frame")
                    break
                
                # Process frame
                annotated = self.process_frame(frame)
                
                # Display
                if self.display:
                    cv2.imshow(f"CCTV Analytics - {self.camera_id}", annotated)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit requested by user")
                        break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        
        if self.picam:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception as e:
                logger.debug(f"Pi Camera cleanup: {e}")
        
        if self.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass  # Ignore GUI errors in headless mode
        
        # Print statistics
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        logger.info("=" * 50)
        logger.info("Pipeline Statistics:")
        logger.info(f"  Frames processed: {self.frame_count}")
        logger.info(f"  Detections logged: {self.detection_count}")
        logger.info(f"  Average FPS: {fps:.2f}")
        logger.info(f"  Total time: {elapsed:.2f}s")
        logger.info("=" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CCTV Analytics Pipeline - Face Detection, Age/Gender Estimation"
    )
    
    parser.add_argument(
        "--source", "-s",
        default="picam",
        help="Video source: 'picam' for Pi Camera, 0 for USB webcam, path for file, rtsp:// for stream"
    )
    parser.add_argument(
        "--camera-id", "-c",
        default="cam_01",
        help="Camera identifier (default: cam_01)"
    )
    parser.add_argument(
        "--backend", "-b",
        default="mtcnn",
        choices=["opencv", "multiscale", "mtcnn", "retinaface", "hailo"],
        help="Face detection backend (default: mtcnn). Use 'mtcnn' for CCTV/side faces, 'multiscale' for small faces"
    )
    parser.add_argument(
        "--confidence", "-conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3 - lower for CCTV footage)"
    )
    parser.add_argument(
        "--enhance-image",
        action="store_true",
        default=True,
        help="Enhance image before detection (better for blurry CCTV)"
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable image enhancement"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: Use OpenCV SSD (10-15 FPS) instead of MTCNN (3-4 FPS). Less accurate for side faces."
    )
    parser.add_argument(
        "--log-path", "-l",
        default="logs/detections.csv",
        help="Path to CSV log file (default: logs/detections.csv)"
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also log to JSONL format"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display (headless mode)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log every N frames (default: 1)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Parse source (convert to int if numeric)
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Create and run pipeline
    enhance = args.enhance_image and not args.no_enhance
    
    # Fast mode uses OpenCV SSD for ~3x faster processing
    backend = args.backend
    confidence = args.confidence
    if args.fast:
        backend = "opencv"
        confidence = max(confidence, 0.5)  # Higher confidence for faster backend
        enhance = False  # Skip enhancement for speed
        logger.info("Fast mode enabled: Using OpenCV SSD backend")
    
    pipeline = CCTVAnalyticsPipeline(
        source=source,
        camera_id=args.camera_id,
        detection_backend=backend,
        detection_confidence=confidence,
        log_path=args.log_path,
        enable_jsonl=args.jsonl,
        display=not args.no_display,
        log_interval=args.log_interval,
        enhance_image=enhance
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()




























