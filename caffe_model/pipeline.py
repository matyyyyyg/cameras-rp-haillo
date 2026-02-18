#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unified_hailo_face import UnifiedHailoFaceDetector, is_hailo_available
from src.kalman_tracker import KalmanPersonTracker, TrackedPerson, format_output_json
from caffe_model.classifier import LightweightClassifier, extract_face_crop

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePipeline:

    def __init__(
        self,
        face_confidence: float = 0.5,
        sensor_id: str = "SENSOR_001",
        max_age: int = 30,
        min_hits: int = 3,
    ):
        logger.info("Initializing Pipeline (Hailo + Caffe + Kalman)...")

        self.sensor_id = sensor_id

        logger.info("1/3: Loading Hailo face detector...")
        self.detector = UnifiedHailoFaceDetector(
            confidence_threshold=face_confidence
        )

        logger.info("2/3: Loading Caffe classifier...")
        self.classifier = LightweightClassifier()

        logger.info("3/3: Loading Kalman tracker...")
        self.tracker = KalmanPersonTracker(
            sensor_id=sensor_id,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=0.25
        )
        self.tracker.USE_REID = False

        self.frame_count = 0
        self.timing = {'detect': 0.0, 'classify': 0.0, 'track': 0.0, 'count': 0}

        logger.info("Pipeline ready!")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[TrackedPerson], Dict]:
        self.frame_count += 1

        t0 = time.time()
        faces = self.detector.detect_faces(frame)
        t1 = time.time()

        detections = []
        for face in faces:
            x, y, w, h = face['box']
            bbox = (x, y, x + w, y + h)

            crop = extract_face_crop(frame, bbox, padding=0.3)
            result = self.classifier.classify(crop)

            if result:
                detections.append({
                    'bbox': bbox,
                    'box': (x, y, w, h),
                    'gender': result.gender,
                    'gender_confidence': result.gender_confidence,
                    'age': result.age,
                    'confidence': face['confidence']
                })
            else:
                detections.append({
                    'bbox': bbox,
                    'box': (x, y, w, h),
                    'gender': 'unknown',
                    'gender_confidence': 0.0,
                    'age': 0.0,
                    'confidence': face['confidence']
                })

        t2 = time.time()

        tracked = self.tracker.update(detections)
        t3 = time.time()

        self.timing['detect'] += (t1 - t0) * 1000
        self.timing['classify'] += (t2 - t1) * 1000
        self.timing['track'] += (t3 - t2) * 1000
        self.timing['count'] += 1

        timing_info = {
            'detect_ms': (t1 - t0) * 1000,
            'classify_ms': (t2 - t1) * 1000,
            'track_ms': (t3 - t2) * 1000,
            'total_ms': (t3 - t0) * 1000,
            'faces': len(faces)
        }

        return tracked, timing_info

    def get_avg_timing(self) -> Dict:
        n = self.timing['count']
        if n == 0:
            return {}
        return {
            'detect_ms': self.timing['detect'] / n,
            'classify_ms': self.timing['classify'] / n,
            'track_ms': self.timing['track'] / n,
        }

    def get_tracker_stats(self) -> Dict:
        return {
            "active_tracks": self.tracker.get_active_count(),
            "total_unique": self.tracker.get_total_count()
        }

    def cleanup(self):
        self.detector.cleanup()
        self.tracker.reset()


def draw_annotations(frame: np.ndarray, persons: List[TrackedPerson]) -> np.ndarray:
    result = frame.copy()

    for person in persons:
        x1, y1, x2, y2 = person.bbox

        if person.gender == 'male':
            color = (255, 150, 0)
        elif person.gender == 'female':
            color = (147, 20, 255)
        else:
            color = (128, 128, 128)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{person.id}"
        if person.gender != 'unknown':
            label += f" {person.gender[0].upper()}"
        if person.age > 0:
            label += f" {int(person.age)}y"
        label += f" ({person.gender_confidence:.0%})"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Caffe Model Pipeline")
    parser.add_argument('--input', type=str, default='camera', help='Video file or "camera"')
    parser.add_argument('--display', action='store_true', help='Show video window')
    parser.add_argument('--log', type=str, help='Output JSON log file')
    parser.add_argument('--face-conf', type=float, default=0.5, help='Face detection threshold')
    parser.add_argument('--sensor-id', type=str, default='SENSOR_001', help='Sensor identifier')
    args = parser.parse_args()

    if not is_hailo_available():
        logger.error("Hailo-8 not detected!")
        return 1

    pipeline = SimplePipeline(
        face_confidence=args.face_conf,
        sensor_id=args.sensor_id
    )

    if args.input == 'camera':
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": (640, 480), "format": "BGR888"}
            )
            picam.configure(config)
            picam.start()
            use_picam = True
            logger.info("Using Pi Camera")
        except:
            cap = cv2.VideoCapture(0)
            use_picam = False
            logger.info("Using USB camera")
    else:
        cap = cv2.VideoCapture(args.input)
        use_picam = False
        logger.info(f"Processing: {args.input}")

    log_file = None
    if args.log:
        log_file = open(args.log, 'w')

    frame_count = 0
    start_time = time.time()

    print("\n" + "=" * 50)
    print("CAFFE MODEL PIPELINE - Press 'q' to quit")
    print("=" * 50 + "\n")

    try:
        while True:
            if use_picam:
                frame = picam.capture_array()
                ret = True
            else:
                ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            persons, timing = pipeline.process_frame(frame)

            if log_file and persons:
                entry = {
                    "sensor_id": args.sensor_id,
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame_count,
                    "detections": [p.to_client_format() for p in persons],
                    "timing_ms": timing
                }
                log_file.write(json.dumps(entry) + '\n')

            if args.display:
                annotated = draw_annotations(frame, persons)

                fps = 1000 / timing['total_ms'] if timing['total_ms'] > 0 else 0
                stats = pipeline.get_tracker_stats()
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Tracking: {stats['active_tracks']} (Kalman)", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated, f"det:{timing['detect_ms']:.0f}ms cls:{timing['classify_ms']:.0f}ms trk:{timing['track_ms']:.0f}ms",
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.imshow('Caffe Model Pipeline', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                avg_timing = pipeline.get_avg_timing()
                logger.info(
                    f"Frame {frame_count}: {len(persons)} tracked, "
                    f"FPS: {avg_fps:.1f}, "
                    f"det: {avg_timing.get('detect_ms', 0):.1f}ms, "
                    f"cls: {avg_timing.get('classify_ms', 0):.1f}ms"
                )

    except KeyboardInterrupt:
        logger.info("Stopped by user")

    finally:
        if use_picam:
            picam.stop()
        else:
            cap.release()

        if args.display:
            cv2.destroyAllWindows()

        if log_file:
            log_file.close()

        pipeline.cleanup()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        stats = pipeline.get_tracker_stats()
        avg_timing = pipeline.get_avg_timing()

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total unique persons: {stats['total_unique']}")
        print(f"Avg timing: detect={avg_timing.get('detect_ms', 0):.1f}ms, "
              f"classify={avg_timing.get('classify_ms', 0):.1f}ms, "
              f"track={avg_timing.get('track_ms', 0):.1f}ms")
        if args.log:
            print(f"Log saved to: {args.log}")
        print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
