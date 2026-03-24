#!/usr/bin/env python3
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple

from dotenv import load_dotenv

from src.detection.hailo_detector import UnifiedHailoFaceDetector, is_hailo_available
from src.classification.classifier import AgeGenderClassifier
from src.kalman_tracking.tracker import KalmanPersonTracker, format_output_json, calculate_iou
from src.utils.types import TrackedPerson, get_age_bucket
from src.utils.face_crop import extract_face_crop, check_face_quality
from src.utils.json_logger import JSONLogger
from src.api.client import create_client_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PiCameraCapture:
    """Wrapper around Picamera2 that mimics cv2.VideoCapture interface."""

    def __init__(self, width=1280, height=960, framerate=30):
        from picamera2 import Picamera2
        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": framerate},
        )
        self._picam.configure(config)
        self._picam.start()
        self._opened = True
        self._width = width
        self._height = height
        self._framerate = framerate

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        frame = self._picam.capture_array()
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._framerate
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        return 0

    def release(self):
        if self._opened:
            self._picam.stop()
            self._opened = False


class FaceAnalysisPipeline:
    """
    Complete pipeline: Hailo face detection + ONNX age/gender + Kalman tracking.
    """

    # Classification skip settings
    RECLASS_INTERVAL = 90
    GENDER_CONF_THRESHOLD = 0.8
    SKIP_IOU_THRESHOLD = 0.25
    MIN_GENDER_VOTES = 2

    def __init__(
        self,
        sensor_id: str = "SENSOR_001",
        hailo_model_path: str = None,
        face_confidence: float = 0.5,
        age_gender_model: str = None,
        max_age: int = 60,
        min_hits: int = 3,
        iou_threshold: float = 0.15
    ):
        logger.info("Initializing Face Analysis Pipeline...")

        self.sensor_id = sensor_id
        self._frame_number = 0

        # 1/3: Hailo face detector
        logger.info("1/3: Initializing Hailo RetinaFace detector...")
        self.face_detector = UnifiedHailoFaceDetector(
            model_path=hailo_model_path,
            confidence_threshold=face_confidence
        )

        # 2/3: ONNX age/gender classifier
        logger.info("2/3: Initializing ONNX age/gender classifier...")
        self.classifier = AgeGenderClassifier(model_path=age_gender_model)

        # 3/3: Kalman tracker
        logger.info("3/3: Initializing Kalman person tracker...")
        self.tracker = KalmanPersonTracker(
            sensor_id=sensor_id,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

        # Per-stage timing accumulators (ms)
        self.timing = {'detect': 0.0, 'classify': 0.0, 'track': 0.0}
        self.timing_count = 0

        logger.info("Pipeline initialized successfully")

    def _match_detection_to_track(self, det_bbox: Tuple[int, int, int, int]) -> int:
        """Pre-match a detection to existing tracks using IoU."""
        best_iou = 0.0
        best_id = -1
        for tracker in self.tracker.trackers:
            track_bbox = tracker.get_state()
            iou = calculate_iou(det_bbox, track_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = tracker.id
        if best_iou >= self.SKIP_IOU_THRESHOLD:
            return best_id
        return -1

    def _can_skip_classification(self, track_id: int) -> bool:
        """Check if a track has confident enough attributes to skip classification."""
        if track_id < 0:
            return False
        attrs = self.tracker.track_attributes.get(track_id)
        if attrs is None:
            return False
        if self._frame_number % self.RECLASS_INTERVAL == 0:
            return False
        total_votes = attrs.get('gender_votes_male', 0) + attrs.get('gender_votes_female', 0)
        if total_votes < self.MIN_GENDER_VOTES:
            return False
        return (
            attrs.get('gender_confidence', 0) > self.GENDER_CONF_THRESHOLD
            and attrs.get('age', 0) > 0
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[List[TrackedPerson], Dict]:
        """
        Process a single frame: detect -> quality gate -> classify -> track.

        Returns:
            Tuple of (tracked_persons, client_json_output)
        """
        self._frame_number += 1

        # --- Stage 1: Detect faces ---
        t0 = time.time()
        faces = self.face_detector.detect_faces(frame)
        t1 = time.time()

        # --- Stage 2: Classify (with skip + quality gate) ---
        detections = []
        skipped = 0
        quality_rejected = 0
        total = len(faces)

        for face in faces:
            x, y, w, h = face['box']
            bbox = (x, y, x + w, y + h)

            matched_track_id = self._match_detection_to_track(bbox)

            if self._can_skip_classification(matched_track_id):
                attrs = self.tracker.track_attributes[matched_track_id]
                detection = {
                    'box': (x, y, w, h),
                    'bbox': bbox,
                    'confidence': face['confidence'],
                    'gender': attrs['gender'],
                    'gender_confidence': attrs['gender_confidence'],
                    'age': attrs['age'],
                }
                skipped += 1
            else:
                face_crop = extract_face_crop(frame, bbox)
                if face_crop.size == 0:
                    continue

                # Quality gate
                passed, metrics = check_face_quality(face_crop)
                if not passed:
                    quality_rejected += 1
                    # Still add to tracker with unknown attributes so tracking continues
                    detection = {
                        'box': (x, y, w, h),
                        'bbox': bbox,
                        'confidence': face['confidence'],
                        'gender': 'unknown',
                        'gender_confidence': 0.0,
                        'age': 0.0,
                    }
                    detections.append(detection)
                    continue

                result = self.classifier.classify(face_crop)
                if result is None:
                    detection = {
                        'box': (x, y, w, h),
                        'bbox': bbox,
                        'confidence': face['confidence'],
                        'gender': 'unknown',
                        'gender_confidence': 0.0,
                        'age': 0.0,
                    }
                else:
                    detection = {
                        'box': (x, y, w, h),
                        'bbox': bbox,
                        'confidence': face['confidence'],
                        'gender': result.gender.lower(),
                        'gender_confidence': result.gender_confidence,
                        'age': result.age,
                    }

            detections.append(detection)

        t2 = time.time()

        # --- Stage 3: Update tracker ---
        tracked_persons = self.tracker.update(detections)
        t3 = time.time()

        # Accumulate timing (ms)
        self.timing['detect'] += (t1 - t0) * 1000
        self.timing['classify'] += (t2 - t1) * 1000
        self.timing['track'] += (t3 - t2) * 1000
        self.timing_count += 1

        # Log per-stage timing every 30 frames
        if self._frame_number % 30 == 0 and self.timing_count > 0:
            avg_d = self.timing['detect'] / self.timing_count
            avg_c = self.timing['classify'] / self.timing_count
            avg_t = self.timing['track'] / self.timing_count
            logger.info(
                f"detect={avg_d:.1f}ms  classify={avg_c:.1f}ms  "
                f"track={avg_t:.1f}ms  skipped={skipped}/{total}  "
                f"quality_rejected={quality_rejected}/{total}"
            )
            self.timing = {'detect': 0.0, 'classify': 0.0, 'track': 0.0}
            self.timing_count = 0

        # Format output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        client_output = format_output_json(tracked_persons, self.sensor_id, timestamp)

        return tracked_persons, client_output

    def get_tracker_stats(self) -> Dict:
        return {
            "active_tracks": self.tracker.get_active_count(),
            "total_unique_persons": self.tracker.get_total_count()
        }

    def cleanup(self):
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
        gender_conf = person.gender_confidence
        age = person.age
        person_id = person.id

        if gender.lower() == 'male':
            color = (255, 150, 0)
        elif gender.lower() == 'female':
            color = (147, 20, 255)
        else:
            color = (128, 128, 128)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if show_ids:
            label = f"ID:{person_id}"
            if gender != 'unknown':
                label += f" {gender.capitalize()}"
            if age > 0:
                label += f" {age:.0f}y {get_age_bucket(age)}"
            label += f" ({gender_conf:.2f})"
        else:
            label = f"{gender.capitalize()} {age:.0f}y {get_age_bucket(age)}" if age > 0 else gender.capitalize()

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Face Analysis: Hailo Detection + ONNX Age/Gender + Kalman Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --display
  python main.py --input video.mp4 --log detections.jsonl
  python main.py --sensor-id CAM_ENTRANCE --log logs/entrance.jsonl
        """
    )

    # Input/output
    parser.add_argument('--input', type=str, default='camera',
                        help='Input: camera, video file path, or image')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to save JSON detections (JSONL format)')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Save annotated output video')

    # Sensor
    parser.add_argument('--sensor-id', type=str, default='SENSOR_001',
                        help='Unique sensor/camera identifier')

    # Detection
    parser.add_argument('--hailo-model', type=str, default=None,
                        help='Path to Hailo face detection HEF')
    parser.add_argument('--face-conf', type=float, default=0.5,
                        help='Face detection confidence threshold')
    parser.add_argument('--age-gender-model', type=str, default=None,
                        help='Path to ONNX age/gender model')

    # Tracking
    parser.add_argument('--max-age', type=int, default=60,
                        help='Max frames to keep track without detection')
    parser.add_argument('--min-hits', type=int, default=3,
                        help='Min detections to confirm track')
    parser.add_argument('--iou-threshold', type=float, default=0.15,
                        help='Min IoU for track association')

    # Logging frequency
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='Log detections every N seconds (default: 1.0)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                        help='Minimum confidence to log detection (default: 0.5)')

    # Camera resolution
    parser.add_argument('--resolution', type=str, default='1280x960',
                        help='Camera resolution as WxH (default: 1280x960)')

    # Display
    parser.add_argument('--display', action='store_true',
                        help='Display output video window')
    parser.add_argument('--no-ids', action='store_true',
                        help='Hide tracking IDs in display')

    # Snapshots
    parser.add_argument('--snapshot-dir', type=str, default=None,
                        help='Directory to save periodic snapshots')
    parser.add_argument('--snapshot-interval', type=float, default=60.0,
                        help='Save snapshot every N seconds (default: 60)')

    args = parser.parse_args()

    # Load .env file (from face_analysis_NEW/ directory)
    load_dotenv(Path(__file__).parent / ".env")

    # Parse resolution
    try:
        res_w, res_h = (int(v) for v in args.resolution.split('x'))
    except ValueError:
        logger.error(f"Invalid resolution format '{args.resolution}', expected WxH (e.g. 1280x960)")
        return 1

    # Check Hailo availability
    if not is_hailo_available():
        logger.error("Hailo-8 not available. This script requires:")
        logger.error("   - Raspberry Pi 5 with Hailo-8 AI HAT")
        logger.error("   - HailoRT installed: sudo apt install hailo-all")
        return 1

    # Initialize pipeline
    try:
        pipeline = FaceAnalysisPipeline(
            sensor_id=args.sensor_id,
            hailo_model_path=args.hailo_model,
            face_confidence=args.face_conf,
            age_gender_model=args.age_gender_model,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    # Initialize API client (reads from .env)
    api_client = create_client_from_env()

    # Initialize JSON logger
    json_logger = None
    if args.log:
        json_logger = JSONLogger(Path(args.log))

    # Open input source
    if args.input == 'camera':
        try:
            cap = PiCameraCapture(width=res_w, height=res_h, framerate=30)
            logger.info(f"Using Pi Camera via Picamera2 ({res_w}x{res_h})")
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

    # Snapshots
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
    print(f"Sensor ID: {args.sensor_id}")
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

            tracked_persons, client_output = pipeline.process_frame(frame)
            frame_time = time.time() - frame_start

            # Log/send at specified interval
            current_time = time.time()
            should_report = current_time - last_log_time >= args.log_interval

            if should_report and len(client_output['detections']) > 0:
                # Filter by min confidence
                confident_detections = [
                    d for d in client_output['detections']
                    if d.get('confidence', 0) >= args.min_confidence
                ]
                confident_persons = [
                    p for p in tracked_persons
                    if p.confidence >= args.min_confidence
                ]

                if confident_detections:
                    # JSONL log
                    if json_logger:
                        filtered_output = client_output.copy()
                        filtered_output['detections'] = confident_detections
                        json_logger.log(filtered_output)

                    # VisionCraft API
                    if api_client and confident_persons:
                        api_client.send_detections_async(confident_persons)

                    last_log_time = current_time

            # FPS
            fps_display = 1.0 / frame_time if frame_time > 0 else 0

            # Annotate
            annotated = draw_annotations(frame, tracked_persons, show_ids=not args.no_ids)

            stats = pipeline.get_tracker_stats()
            cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Tracking: {stats['active_tracks']} persons", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Total unique: {stats['total_unique_persons']}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(annotated, f"Sensor: {args.sensor_id}", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Display
            if args.display:
                cv2.imshow('Face Analysis Pipeline', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save video
            if video_writer:
                video_writer.write(annotated)

            # Snapshots
            if snapshot_dir and (current_time - last_snapshot_time >= args.snapshot_interval):
                snapshot_name = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
                snapshot_path = snapshot_dir / snapshot_name
                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Snapshot saved: {snapshot_path}")
                last_snapshot_time = current_time

            # Progress
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
