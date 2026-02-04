#!/usr/bin/env python3
"""
Hailo SCRFD-10g face detection + classification.py gender/age + Kalman Tracking.
Outputs detections in client JSON format with persistent person IDs.

Requires Raspberry Pi 5 with Hailo-8 accelerator.

Output format:
{
    "sensor_id": "XXXXXXXXXXXXX",
    "timestamp": "2025-12-10 14:35:12.123456",
    "detections": [
        {"id": 1, "age": 28.4, "gender": "male", "confidence": 0.95, "bbox": {...}}
    ]
}
"""

import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple


class PiCameraCapture:
    """Wrapper around Picamera2 that mimics cv2.VideoCapture interface."""

    def __init__(self, width=640, height=480, framerate=30):
        from picamera2 import Picamera2
        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": framerate},
        )
        self._picam.configure(config)
        self._picam.start()
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        frame = self._picam.capture_array()
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        return 0

    def release(self):
        if self._opened:
            self._picam.stop()
            self._opened = False


# Import unified Hailo face detector
from src.unified_hailo_face import UnifiedHailoFaceDetector, is_hailo_available

# Import classification module (handles InsightFace + Caffe fallback + CLAHE + upscaling)
from src.classification import AgeGenderClassifier, extract_face_crop

# Import Kalman tracker
from src.kalman_tracker import KalmanPersonTracker, format_output_json, TrackedPerson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SCRFDPipeline:
    """
    Complete pipeline: Hailo SCRFD-10g face detection + AgeGenderClassifier + Kalman tracking.
    """

    def __init__(
        self,
        sensor_id: str = "SENSOR_001",
        hailo_model_path: str = None,
        face_confidence: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.25
    ):
        """Initialize pipeline with tracking."""
        logger.info("Initializing SCRFD + Classification + Tracking Pipeline...")

        self.sensor_id = sensor_id

        # Default to SCRFD-10g model
        if hailo_model_path is None:
            project_root = Path(__file__).parent
            hailo_model_path = str(project_root / "models" / "hailo" / "scrfd_10g.hef")

        # 1. Hailo face detector
        logger.info("1/3: Initializing Hailo SCRFD-10g detector...")
        self.face_detector = UnifiedHailoFaceDetector(
            model_path=hailo_model_path,
            confidence_threshold=face_confidence
        )

        # 2. Age/gender classifier (auto-selects best available backend)
        logger.info("2/3: Initializing age/gender classifier...")
        self.classifier = AgeGenderClassifier()

        # 3. Kalman tracker
        logger.info("3/3: Initializing Kalman person tracker...")
        self.tracker = KalmanPersonTracker(
            sensor_id=sensor_id,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

        logger.info("Pipeline initialized successfully")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[TrackedPerson], Dict]:
        """
        Process a single frame: detect faces + classify gender/age + track persons.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (tracked_persons, client_json_output)
        """
        # Detect faces using Hailo
        faces = self.face_detector.detect_faces(frame)

        if not hasattr(self, '_detect_log_count'):
            self._detect_log_count = 0
        self._detect_log_count += 1
        if self._detect_log_count <= 5 or (len(faces) > 0 and self._detect_log_count % 30 == 0):
            logger.info(f"SCRFD raw detections: {len(faces)} faces")

        if len(faces) > 10:
            logger.warning(f"Too many detections ({len(faces)}), keeping top 10 by confidence")
            faces = sorted(faces, key=lambda f: f['confidence'], reverse=True)[:10]

        # Classify gender/age for each face
        detections = []
        for face in faces:
            x, y, w, h = face['box']
            bbox = (x, y, x + w, y + h)

            # Use extract_face_crop which handles padding + upscaling
            face_crop = extract_face_crop(frame, bbox)

            if face_crop is None or face_crop.size == 0:
                continue

            # Classify using the unified classifier (InsightFace + CLAHE + Caffe fallback)
            result = self.classifier.classify(face_crop)

            detection = {
                'box': (x, y, w, h),
                'bbox': bbox,
                'confidence': face['confidence'],
                'gender': result.gender.lower(),
                'gender_confidence': result.gender_confidence,
                'age': result.raw_age if result.raw_age is not None else float(result.age_midpoint)
            }

            detections.append(detection)

        # Update tracker with detections
        tracked_persons = self.tracker.update(detections)

        # Format output as client JSON
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        client_output = format_output_json(tracked_persons, self.sensor_id, timestamp)

        return tracked_persons, client_output

    def get_tracker_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            "active_tracks": self.tracker.get_active_count(),
            "total_unique_persons": self.tracker.get_total_count()
        }

    def cleanup(self):
        """Release resources."""
        self.face_detector.cleanup()


def draw_annotations(
    frame: np.ndarray,
    tracked_persons: List[TrackedPerson],
    show_ids: bool = True
) -> np.ndarray:
    """Draw bounding boxes and labels with tracking IDs."""
    annotated = frame.copy()

    for person in tracked_persons:
        x1, y1, x2, y2 = person.bbox
        gender = person.gender
        age = person.age
        person_id = person.id

        # Color based on gender
        if gender.lower() == 'male':
            color = (255, 150, 0)  # Blue-ish for male
        elif gender.lower() == 'female':
            color = (147, 20, 255)  # Pink for female
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Build label
        if show_ids:
            label = f"ID:{person_id}"
            if gender != 'unknown':
                label += f" {gender.capitalize()}"
            if age > 0:
                label += f" {age:.0f}y"
            label += f" ({person.confidence:.2f})"
        else:
            label = f"{gender.capitalize()} {age:.0f}y" if age > 0 else gender.capitalize()

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)

        # Draw label text
        cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


class JSONLogger:
    """Logger for client JSON format output."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            pass
        self.frame_count = 0
        logger.info(f"JSON logger initialized: {output_path}")

    def log(self, client_output: Dict) -> None:
        with open(self.output_path, 'a') as f:
            json.dump(client_output, f)
            f.write('\n')
        self.frame_count += 1

    def get_stats(self) -> Dict:
        return {"frames_logged": self.frame_count, "output_path": str(self.output_path)}


def main():
    parser = argparse.ArgumentParser(
        description="Hailo SCRFD-10g + Age/Gender Classification with Kalman Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_scrfd_insightface.py --display
  python test_scrfd_insightface.py --input video.mp4 --log detections.jsonl
  python test_scrfd_insightface.py --input camera --output-video test_scrfd.mp4
        """
    )

    # Input/output options
    parser.add_argument('--input', type=str, default='camera',
                        help='Input: camera, video file path, or image')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to save JSON detections (JSONL format)')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Save annotated output video')

    # Sensor configuration
    parser.add_argument('--sensor-id', type=str, default='SENSOR_001',
                        help='Unique sensor/camera identifier')

    # Detection parameters
    parser.add_argument('--hailo-model', type=str, default=None,
                        help='Path to Hailo face detection HEF (default: models/hailo/scrfd_10g.hef)')
    parser.add_argument('--face-conf', type=float, default=0.3,
                        help='Face detection confidence threshold (default: 0.3)')

    # Tracking parameters
    parser.add_argument('--max-age', type=int, default=30,
                        help='Max frames to keep track without detection')
    parser.add_argument('--min-hits', type=int, default=3,
                        help='Min detections to confirm track')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Min IoU for track association')

    # Logging frequency
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='Log detections every N seconds (default: 1.0)')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                        help='Minimum confidence to log detection (default: 0.3)')

    # Display options
    parser.add_argument('--display', action='store_true',
                        help='Display output video window')
    parser.add_argument('--no-ids', action='store_true',
                        help='Hide tracking IDs in display')

    # Remote monitoring
    parser.add_argument('--snapshot-dir', type=str, default=None,
                        help='Directory to save periodic snapshots for remote monitoring')
    parser.add_argument('--snapshot-interval', type=float, default=60.0,
                        help='Save snapshot every N seconds (default: 60)')

    args = parser.parse_args()

    # Check Hailo availability
    if not is_hailo_available():
        logger.error("Hailo-8 not available. This script requires:")
        logger.error("   - Raspberry Pi 5 with Hailo-8 AI HAT")
        logger.error("   - HailoRT installed: sudo apt install hailo-all")
        return 1

    # Initialize pipeline
    try:
        pipeline = SCRFDPipeline(
            sensor_id=args.sensor_id,
            hailo_model_path=args.hailo_model,
            face_confidence=args.face_conf,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1

    # Initialize JSON logger
    json_logger = None
    if args.log:
        json_logger = JSONLogger(Path(args.log))

    # Open input source
    if args.input == 'camera':
        try:
            cap = PiCameraCapture(width=640, height=480, framerate=30)
            logger.info("Using Pi Camera via Picamera2")
        except Exception as e:
            logger.warning(f"Picamera2 not available ({e}), trying OpenCV...")
            cap = cv2.VideoCapture(0)
            logger.info("Using camera input via OpenCV")
    else:
        cap = cv2.VideoCapture(args.input)
        logger.info(f"Processing: {args.input}")

    if not cap.isOpened():
        logger.error(f"Failed to open input: {args.input}")
        return 1

    # Video writer
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
        logger.info(f"Saving video to: {args.output_video}")

    # Snapshot directory
    snapshot_dir = None
    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Snapshots will be saved to: {snapshot_dir}")

    # Processing loop
    frame_count = 0
    start_time = time.time()
    last_log_time = time.time()
    last_snapshot_time = time.time()
    fps_display = 0.0

    logger.info("Starting processing... Press 'q' to quit")
    print("\n" + "=" * 60)
    print(f"SCRFD-10g + AgeGenderClassifier + Kalman Tracking")
    print(f"Sensor ID: {args.sensor_id}")
    print(f"Confidence threshold: {args.face_conf}")
    print(f"Tracker: Kalman Filter + Hungarian Algorithm")
    print(f"Max Age: {args.max_age} frames | Min Hits: {args.min_hits}")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_start = time.time()

            # Process frame
            tracked_persons, client_output = pipeline.process_frame(frame)
            frame_time = time.time() - frame_start

            # Log to JSON file at specified interval
            current_time = time.time()
            if json_logger and len(client_output['detections']) > 0:
                if current_time - last_log_time >= args.log_interval:
                    filtered_output = client_output.copy()
                    filtered_output['detections'] = [
                        d for d in client_output['detections']
                        if d.get('confidence', 0) >= args.min_confidence
                    ]
                    if len(filtered_output['detections']) > 0:
                        json_logger.log(filtered_output)
                        last_log_time = current_time

            # Calculate FPS
            fps_display = 1.0 / frame_time if frame_time > 0 else 0

            # Draw annotations
            annotated = draw_annotations(frame, tracked_persons, show_ids=not args.no_ids)

            # Add overlay info
            stats = pipeline.get_tracker_stats()
            cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Tracking: {stats['active_tracks']} persons", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Total unique: {stats['total_unique_persons']}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(annotated, f"SCRFD-10g | {args.sensor_id}", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Display
            if args.display:
                cv2.imshow('SCRFD-10g + Kalman Tracking', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save video
            if video_writer:
                video_writer.write(annotated)

            # Save periodic snapshots
            if snapshot_dir and (current_time - last_snapshot_time >= args.snapshot_interval):
                snapshot_name = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
                snapshot_path = snapshot_dir / snapshot_name
                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Snapshot saved: {snapshot_path}")
                last_snapshot_time = current_time

            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                logger.info(f"Frame {frame_count}: {len(tracked_persons)} tracked, FPS: {avg_fps:.1f}")

                if tracked_persons:
                    print(f"\n--- Frame {frame_count} JSON Output ---")
                    print(json.dumps(client_output, indent=2))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()

        pipeline.cleanup()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        stats = pipeline.get_tracker_stats()

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Model: SCRFD-10g")
        print(f"  Sensor ID: {args.sensor_id}")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Total unique persons tracked: {stats['total_unique_persons']}")

        if json_logger:
            log_stats = json_logger.get_stats()
            print(f"  Detections logged: {log_stats['frames_logged']} entries")
            print(f"  Log file: {log_stats['output_path']}")

        print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
