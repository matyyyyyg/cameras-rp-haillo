"""
Web Server for CCTV Analytics Dashboard

Provides:
- Live video streaming with face detection
- REST API for analytics data
- WebSocket for real-time updates


"""

from flask import Flask, Response, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
import cv2
import numpy as np
import json
import csv
import threading
import queue
import time
import os
import tempfile
import uuid
from pathlib import Path
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from werkzeug.utils import secure_filename

from .detectors import create_detector
from .classification import AgeGenderClassifier, ClassificationResult, extract_face_crop, get_age_bucket
from .kalman_tracker import KalmanPersonTracker
from .logging_utils import DetectionLogger, DetectionEvent
from .client_formatter import format_for_client, ClientOutputFormatter

# Pi Camera support (optional - for Raspberry Pi)
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    print("✅ picamera2 available for web server")
except ImportError:
    print("ℹ️  picamera2 not available, will use USB webcam for live feed")


class ThreadedVideoReader:
    """
    Non-blocking video reader using a background thread.
    Pre-buffers frames for smoother video processing on Mac.
    """
    
    def __init__(self, video_path: str, buffer_size: int = 30):
        self.cap = cv2.VideoCapture(video_path)
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread = None
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = 0
        
    def start(self):
        """Start the background frame reading thread."""
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        return self
    
    def _read_frames(self):
        """Background thread that reads frames into buffer."""
        while not self.stopped:
            if not self.buffer.full():
                ret, frame = self.cap.read()
                self.frame_count += 1
                
                if not ret:
                    self.stopped = True
                    # Signal end of video with None
                    self.buffer.put(None)
                    break
                    
                self.buffer.put((frame, self.frame_count))
            else:
                # Buffer full, wait a bit
                time.sleep(0.001)
    
    def read(self) -> tuple:
        """
        Read next frame from buffer.
        Returns (frame, frame_number) or (None, -1) if video ended.
        """
        try:
            result = self.buffer.get(timeout=0.5)
            if result is None:
                return None, -1
            return result
        except queue.Empty:
            return None, -1
    
    def stop(self):
        """Stop the reader thread."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()
    
    def is_opened(self) -> bool:
        return self.cap.isOpened()
    
    @property
    def progress(self) -> int:
        """Return current progress as percentage."""
        if self.total_frames == 0:
            return 0
        return int((self.frame_count / self.total_frames) * 100)


def enhance_for_cctv(frame: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better face detection in CCTV footage.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve visibility in low-light or low-contrast images.
    """
    if frame is None or frame.size == 0:
        return frame
    
    try:
        # Convert to LAB color space for CLAHE on luminance only
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (luminance)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    except Exception:
        # If enhancement fails, return original
        return frame


# Custom JSON encoder for numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.json_encoder = NumpyJSONEncoder  # Use custom encoder
CORS(app)

# Global state
class AnalyticsState:
    def __init__(self):
        self.detector = None
        self.classifier = None
        self.tracker = None
        self.logger = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.stats = {
            'fps': 0,
            'frame_count': 0,
            'detection_count': 0,
            'active_persons': 0,
            'current_detections': []
        }
        self.recent_detections = deque(maxlen=100)
        self.client_formatter = ClientOutputFormatter(sensor_id="SENSOR_001")
        # Keep last detections for UI persistence
        self.last_detections = []
        self.last_detection_time = 0
        # Video upload processing
        self.video_processing = False
        self.video_progress = 0
        self.video_results = []
        # Video file playback
        self.video_file_path = None
        self.video_is_playing = False
        self.video_tracker = None
        # Performance optimization settings - PRODUCTION (Raspberry Pi 5 + Hailo-8L)
        # Full resolution, every frame - Hailo can handle 30+ FPS
        self.frame_skip = 1  # Process every frame on Hailo hardware
        self.scale_factor = 1.0  # Full resolution for best accuracy
        self.classification_cache = {}  # Cache age/gender results per person_id
        self.cache_ttl = 5  # Seconds to cache classification results (short to gather voting samples)
        # VIDEO-SPECIFIC settings (optimized for production)
        self.video_frame_skip = 1  # Process every frame for best accuracy
        self.video_scale_factor = 1.0  # Full resolution for best detection
        self.video_cache_ttl = 45  # Moderate cache for video processing
        self.video_playback_speed = 1.0  # Real-time playback
        self.video_detector = None  # Will use same Hailo detector
        # Per-person majority voting for gender/age accuracy
        self.person_votes = {}  # {person_id: {'genders': [], 'ages': [], 'locked': False, 'result': None}}

state = AnalyticsState()

MIN_VOTES_TO_LOCK = 5  # Number of classification samples before locking result


def get_voted_result(person_id: str, result: ClassificationResult) -> ClassificationResult:
    """
    Accumulate gender/age predictions per person and return majority-voted result.

    Collects predictions across frames. After MIN_VOTES_TO_LOCK samples,
    locks in the consensus (majority gender, median age) for that person.
    """
    if result.gender == "unknown":
        # Don't count failed classifications
        votes = state.person_votes.get(person_id)
        if votes and votes['locked']:
            return votes['result']
        return result

    if person_id not in state.person_votes:
        state.person_votes[person_id] = {
            'genders': [],
            'ages': [],
            'locked': False,
            'result': None,
        }

    votes = state.person_votes[person_id]

    # If already locked, return the locked result
    if votes['locked']:
        return votes['result']

    # Accumulate votes
    votes['genders'].append(result.gender)
    raw_age = result.raw_age if result.raw_age else result.age_midpoint
    if raw_age and raw_age > 0:
        votes['ages'].append(float(raw_age))

    # Check if we have enough samples to lock
    if len(votes['genders']) >= MIN_VOTES_TO_LOCK:
        # Gender: majority vote
        gender_counts = Counter(votes['genders'])
        voted_gender = gender_counts.most_common(1)[0][0]
        gender_ratio = gender_counts[voted_gender] / len(votes['genders'])

        # Age: median (resistant to outliers)
        if votes['ages']:
            sorted_ages = sorted(votes['ages'])
            mid = len(sorted_ages) // 2
            if len(sorted_ages) % 2 == 0:
                voted_age = (sorted_ages[mid - 1] + sorted_ages[mid]) / 2
            else:
                voted_age = sorted_ages[mid]
        else:
            voted_age = result.raw_age or result.age_midpoint

        age_bucket, age_midpoint = get_age_bucket(int(voted_age))

        locked_result = ClassificationResult(
            age_bucket=age_bucket,
            age_midpoint=round(voted_age, 1),
            age_confidence=result.age_confidence,
            gender=voted_gender,
            gender_confidence=round(gender_ratio, 2),
            raw_age=round(voted_age, 1),
        )
        votes['locked'] = True
        votes['result'] = locked_result
        return locked_result

    # Not yet locked — return current best guess
    gender_counts = Counter(votes['genders'])
    best_gender = gender_counts.most_common(1)[0][0]
    if votes['ages']:
        sorted_ages = sorted(votes['ages'])
        mid = len(sorted_ages) // 2
        best_age = sorted_ages[mid]
    else:
        best_age = result.raw_age or result.age_midpoint

    age_bucket, _ = get_age_bucket(int(best_age))
    return ClassificationResult(
        age_bucket=age_bucket,
        age_midpoint=round(float(best_age), 1),
        age_confidence=result.age_confidence,
        gender=best_gender,
        gender_confidence=result.gender_confidence,
        raw_age=round(float(best_age), 1),
    )


def initialize_pipeline(camera_id: str = "web_cam"):
    """
    Initialize detection pipeline optimized for Raspberry Pi 5 + Hailo-8.
    
    Production configuration:
    - Hailo RetinaFace for face detection (30-40 FPS)
    - Confidence 0.6 (eliminates false positives)
    - CPU fallback only for development/testing
    """
    
    print("Initializing for Raspberry Pi 5 + Hailo-8...")
    
    # PRIMARY: Hailo-8L hardware accelerated detection (26 TFLOPS)
    try:
        state.detector = create_detector(
            'hailo', 
            confidence_threshold=0.5,  # Lower threshold to catch more faces, filter by classification
            hef_path='models/hailo/retinaface_mobilenet_v1.hef'
        )
        print("✅ Hailo-8L RetinaFace loaded (30-40 FPS @ full resolution)")
        hailo_available = True
    except Exception as e:
        print(f"⚠️  Hailo not available: {e}")
        print("   Falling back to CPU (development only)")
        # Fallback to MTCNN for dev/testing
        state.detector = create_detector('mtcnn', confidence_threshold=0.5)
        hailo_available = False
    
    # Video detector: same as live (Hailo performs well on all sources)
    state.video_detector = state.detector
    
    # Age/Gender classifier (runs on CPU, optimized separately)
    state.classifier = AgeGenderClassifier()
    
    # Tracker
    state.tracker = KalmanPersonTracker(sensor_id=camera_id)
    
    # Logger
    state.logger = DetectionLogger(
        csv_path='logs/detections.csv',
        enable_jsonl=True  # Enable JSONL for production logging
    )
    
    print("Pipeline initialized successfully")
    if hailo_available:
        print("  Mode: PRODUCTION (Hailo-8 accelerated)")
        print("  Expected FPS: 30-40+")
    else:
        print("  Mode: DEVELOPMENT (CPU fallback)")
        print("  Expected FPS: 3-5")
    print("Pipeline initialized (Optimized for Mac)")
    print(f"  Live camera: OpenCV detector, skip={state.frame_skip}, scale={state.scale_factor}")
    print(f"  Video upload: MTCNN detector, skip={state.video_frame_skip}, scale={state.video_scale_factor}")


def process_frame(frame: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """Process a single frame and return annotated version (optimized)."""
    h, w = frame.shape[:2]
    
    # Scale down for faster detection
    if state.scale_factor < 1.0:
        small_frame = cv2.resize(frame, (int(w * state.scale_factor), int(h * state.scale_factor)))
    else:
        small_frame = frame
    
    # Detect faces on smaller frame
    detections = state.detector.detect_faces(small_frame)
    
    # Scale bounding boxes back to original size
    if state.scale_factor < 1.0:
        scale = 1.0 / state.scale_factor
        for det in detections:
            det['bbox'] = tuple(int(x * scale) for x in det['bbox'])
    
    # Track faces
    tracked = state.tracker.update(detections)

    current_dets = []
    current_time = time.time()

    for person in tracked:
        bbox = person.bbox
        person_id = f"person_{person.id:04d}"
        det_conf = person.confidence

        # Check classification cache first (big performance boost!)
        cached = state.classification_cache.get(person_id)
        if use_cache and cached and (current_time - cached['time']) < state.cache_ttl:
            result = cached['result']
        else:
            # Extract and classify (only if not cached)
            face_crop = extract_face_crop(frame, bbox, padding=0.2)
            result = state.classifier.classify(face_crop)
            # Cache the result
            state.classification_cache[person_id] = {'result': result, 'time': current_time}

        # Apply majority voting for stable gender/age
        result = get_voted_result(person_id, result)

        # Log detection
        event = DetectionEvent.create(
            person_id=person_id,
            camera_id=state.tracker.sensor_id,
            age_midpoint=result.age_midpoint,
            age_bucket=result.age_bucket,
            age_confidence=result.age_confidence,
            gender=result.gender,
            gender_confidence=result.gender_confidence,
            bbox=bbox,
            detection_confidence=det_conf
        )
        state.logger.log(event)
        state.stats['detection_count'] += 1

        # Store for API (convert numpy types to Python native for JSON)
        current_dets.append({
            'person_id': person_id,
            'bbox': tuple(int(x) for x in bbox),
            'gender': result.gender,
            'age_bucket': result.age_bucket,
            'age_midpoint': float(result.age_midpoint),
            'raw_age': float(result.raw_age) if result.raw_age else float(result.age_midpoint),
            'confidence': round(float(det_conf), 2)
        })

        state.recent_detections.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'person_id': person_id,
            'gender': result.gender,
            'age_bucket': result.age_bucket
        })

        # Draw on frame
        x1, y1, x2, y2 = bbox
        color = (0, 255, 100)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{result.gender}, {result.age_bucket}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (x1, y1-25), (x1 + len(label)*10, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-7), font, 0.5, (0, 0, 0), 1)

        # ID
        cv2.putText(frame, f"ID:{person.id}", (x1, y2+15), font, 0.4, color, 1)

    state.stats['current_detections'] = current_dets
    state.stats['active_persons'] = state.tracker.get_active_count()
    
    # Persist detections for UI (keep for 2 seconds)
    if current_dets:
        state.last_detections = current_dets
        state.last_detection_time = time.time()
    elif time.time() - state.last_detection_time < 2.0:
        state.stats['current_detections'] = state.last_detections
    
    # Clean old cache entries periodically
    if len(state.classification_cache) > 50:
        old_ids = [k for k, v in state.classification_cache.items() 
                   if current_time - v['time'] > state.cache_ttl * 2]
        for old_id in old_ids:
            del state.classification_cache[old_id]
    
    return frame


def generate_frames():
    """
    Generate video frames for streaming.
    Uses Pi Camera on Raspberry Pi, falls back to USB webcam otherwise.
    """
    picam = None
    use_picamera = False
    
    # Try Pi Camera first (Raspberry Pi)
    if PICAMERA2_AVAILABLE:
        try:
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
            picam.configure(config)
            picam.start()
            use_picamera = True
            print("✅ Pi Camera initialized for live feed: 1280x720 @ 30 FPS")
            time.sleep(1)  # Camera warm-up
        except Exception as e:
            print(f"⚠️ Pi Camera failed: {e}, falling back to USB webcam")
            use_picamera = False
            if picam:
                try:
                    picam.close()
                except:
                    pass
                picam = None
    
    # Fallback to USB webcam
    if not use_picamera:
        state.cap = cv2.VideoCapture(0)
        # Set lower resolution for faster processing
        state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("📹 Using USB webcam for live feed: 640x480")
    
    state.is_running = True
    
    fps_start = time.time()
    frame_count = 0
    skip_counter = 0
    last_processed = None
    
    try:
        while state.is_running:
            # Read frame from appropriate source
            if use_picamera and picam:
                frame = picam.capture_array()
                # Convert RGB888 to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret = True
            else:
                ret, frame = state.cap.read()
            
            if not ret or frame is None:
                break
            
            skip_counter += 1
            state.stats['frame_count'] += 1
            frame_count += 1
            
            # Only process every Nth frame for speed
            if skip_counter >= state.frame_skip:
                skip_counter = 0
                processed = process_frame(frame)
                last_processed = processed.copy()
            else:
                # Reuse last processed frame with current frame's faces
                if last_processed is not None:
                    processed = frame.copy()
                    # Draw cached bounding boxes on current frame
                    for det in state.stats.get('current_detections', []):
                        bbox = det['bbox']
                        x1, y1, x2, y2 = bbox
                        color = (0, 255, 100)
                        cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
                        label = f"{det['gender']}, {det['age_bucket']}"
                        cv2.rectangle(processed, (x1, y1-25), (x1 + len(label)*10, y1), color, -1)
                        cv2.putText(processed, label, (x1+2, y1-7), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                else:
                    processed = frame
            
            # Calculate FPS
            elapsed = time.time() - fps_start
            if elapsed > 1.0:
                state.stats['fps'] = round(frame_count / elapsed, 1)
                fps_start = time.time()
                frame_count = 0
            
            # Draw FPS and source info
            source_label = "PiCam" if use_picamera else "USB"
            cv2.putText(processed, f"FPS: {state.stats['fps']} ({source_label})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed, f"Skip: {state.frame_skip} | Scale: {int(state.scale_factor*100)}%", 
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Encode frame (reduced quality for speed)
            _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        # Proper cleanup
        if use_picamera and picam:
            try:
                picam.stop()
                picam.close()
                print("✅ Pi Camera released")
            except:
                pass
        elif state.cap:
            state.cap.release()
            print("✅ USB webcam released")


def process_video_frame(frame: np.ndarray, tracker, video_cache: dict) -> tuple:
    """Process a video file frame and return annotated frame + detections (optimized)."""
    h, w = frame.shape[:2]
    
    # Scale down for faster detection
    if state.scale_factor < 1.0:
        small_frame = cv2.resize(frame, (int(w * state.scale_factor), int(h * state.scale_factor)))
    else:
        small_frame = frame
    
    # Detect faces on smaller frame
    detections = state.detector.detect_faces(small_frame)
    
    # Scale bounding boxes back to original size
    if state.scale_factor < 1.0:
        scale = 1.0 / state.scale_factor
        for det in detections:
            det['bbox'] = tuple(int(x * scale) for x in det['bbox'])
    
    # Track faces
    tracked = tracker.update(detections)

    current_dets = []
    current_time = time.time()

    for person in tracked:
        bbox = person.bbox
        person_id = f"person_{person.id:04d}"
        det_conf = person.confidence

        # Check video-specific cache first
        cached = video_cache.get(person_id)
        if cached and (current_time - cached['time']) < state.cache_ttl:
            result = cached['result']
        else:
            # Classify face
            face_crop = extract_face_crop(frame, bbox, padding=0.2)
            result = state.classifier.classify(face_crop)
            # Cache result
            video_cache[person_id] = {'result': result, 'time': current_time}

        # Apply majority voting for stable gender/age
        result = get_voted_result(person_id, result)

        # Store detection
        current_dets.append({
            'person_id': person_id,
            'bbox': tuple(int(x) for x in bbox),
            'gender': result.gender,
            'age_bucket': result.age_bucket,
            'age_midpoint': float(result.age_midpoint),
            'raw_age': float(result.raw_age) if result.raw_age else float(result.age_midpoint),
            'confidence': round(float(det_conf), 2)
        })

        # Draw on frame
        x1, y1, x2, y2 = bbox
        color = (0, 255, 100)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{result.gender}, {result.age_bucket}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (x1, y1-25), (x1 + len(label)*10, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-7), font, 0.5, (0, 0, 0), 1)

        # ID
        cv2.putText(frame, f"ID:{person.id}", (x1, y2+15), font, 0.4, color, 1)

    return frame, current_dets


def generate_video_file_frames():
    """
    Generate frames from uploaded video file for streaming.
    OPTIMIZED for Mac with:
    - Threaded frame reading (non-blocking)
    - Aggressive frame skipping (every 5th frame)
    - Lower resolution processing (35% scale)
    - Faster playback (3x speed)
    - Longer classification cache (90s)
    """
    if not state.video_file_path or not Path(state.video_file_path).exists():
        return
    
    # Use threaded video reader for non-blocking frame access
    reader = ThreadedVideoReader(state.video_file_path, buffer_size=30)
    if not reader.is_opened():
        return
    
    reader.start()  # Start background frame reading thread
    
    state.video_is_playing = True
    state.video_tracker = KalmanPersonTracker(sensor_id='video_upload')
    
    total_frames = reader.total_frames
    fps = reader.fps
    
    # Use VIDEO-SPECIFIC settings (more aggressive than live camera)
    video_frame_skip = state.video_frame_skip  # 5 frames
    playback_speed = state.video_playback_speed  # 3x
    frame_delay = max(0.001, 1.0 / (fps * playback_speed))  # Minimum delay
    
    frame_count = 0
    skip_counter = 0
    state.video_results = []
    video_cache = {}  # Video-specific cache (longer TTL)
    last_dets = []
    last_processed_frame = None
    
    try:
        while state.video_is_playing:
            start_time = time.time()
            frame, frame_num = reader.read()
            
            if frame is None:
                break
            
            frame_count = frame_num
            skip_counter += 1
            state.video_progress = reader.progress
            
            # Process only every Nth frame (VIDEO skip is 5, not 2)
            if skip_counter >= video_frame_skip:
                skip_counter = 0
                processed, dets = process_video_frame_fast(frame, state.video_tracker, video_cache)
                last_dets = dets
                last_processed_frame = processed
                
                # Store detections for results
                for det in dets:
                    det_copy = det.copy()
                    det_copy['frame'] = frame_count
                    det_copy['video_timestamp'] = round(frame_count / fps, 2)
                    state.video_results.append(det_copy)
            else:
                # Fast path: just draw boxes on skipped frames (no detection)
                if last_dets:
                    processed = draw_cached_detections_fast(frame, last_dets)
                else:
                    processed = frame
                dets = last_dets
            
            # Update stats for UI
            state.stats['current_detections'] = dets
            state.stats['active_persons'] = len(dets)
            
            # Add compact frame info overlay
            cv2.putText(processed, f"{frame_count}/{total_frames} ({state.video_progress}%)", 
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(processed, f"Faces: {len(dets)} | {playback_speed}x", 
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Encode with lower quality for speed
            _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Minimal delay - let the processing pace itself
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
                
    finally:
        reader.stop()
        state.video_is_playing = False
        state.video_progress = 100


def draw_cached_detections_fast(frame: np.ndarray, detections: list) -> np.ndarray:
    """Fast drawing of cached detections (no processing, just draw boxes)."""
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        label = f"{det['gender']}, {det['age_bucket']}"
        cv2.rectangle(frame, (x1, y1-20), (x1 + len(label)*8, y1), (0, 255, 100), -1)
        cv2.putText(frame, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return frame


def process_video_frame_fast(frame: np.ndarray, tracker, video_cache: dict) -> tuple:
    """
    Process a video frame with CCTV-optimized detection.
    Uses MTCNN detector with image enhancement for better small face detection.
    """
    h, w = frame.shape[:2]
    
    # Apply image enhancement for low-quality CCTV footage
    enhanced = enhance_for_cctv(frame)
    
    # VIDEO-specific scale factor (0.8 = 80% resolution to keep small faces)
    scale = state.video_scale_factor
    
    if scale < 1.0:
        proc_frame = cv2.resize(enhanced, (int(w * scale), int(h * scale)), 
                                interpolation=cv2.INTER_LINEAR)
    else:
        proc_frame = enhanced
    
    # Use dedicated VIDEO detector (MTCNN - better for CCTV)
    # Falls back to OpenCV if video_detector not initialized
    detector = state.video_detector if state.video_detector else state.detector
    detections = detector.detect_faces(proc_frame)
    
    # Scale bounding boxes back to original size
    if scale < 1.0:
        inv_scale = 1.0 / scale
        for det in detections:
            det['bbox'] = tuple(int(x * inv_scale) for x in det['bbox'])
    
    # Track faces
    tracked = tracker.update(detections)

    current_dets = []
    current_time = time.time()

    for person in tracked:
        bbox = person.bbox
        person_id = f"person_{person.id:04d}"
        det_conf = person.confidence

        # Check video-specific cache (longer TTL: 90s)
        cached = video_cache.get(person_id)
        if cached and (current_time - cached['time']) < state.video_cache_ttl:
            result = cached['result']
        else:
            # Classify face (most expensive operation)
            face_crop = extract_face_crop(frame, bbox, padding=0.2)
            result = state.classifier.classify(face_crop)
            video_cache[person_id] = {'result': result, 'time': current_time}

        # Apply majority voting for stable gender/age
        result = get_voted_result(person_id, result)

        # Store detection data
        current_dets.append({
            'person_id': person_id,
            'bbox': tuple(int(x) for x in bbox),
            'gender': result.gender,
            'age_bucket': result.age_bucket,
            'age_midpoint': float(result.age_midpoint),
            'raw_age': float(result.raw_age) if result.raw_age else float(result.age_midpoint),
            'confidence': round(float(det_conf), 2)
        })

        # Draw on frame
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        label = f"{result.gender}, {result.age_bucket}"
        cv2.rectangle(frame, (x1, y1-20), (x1 + len(label)*8, y1), (0, 255, 100), -1)
        cv2.putText(frame, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(frame, f"ID:{person.id}", (x1, y2+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 100), 1)

    return frame, current_dets


@app.route('/')
def index():
    """Serve main dashboard."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route for live camera."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_file_feed')
def video_file_feed():
    """Video streaming route for uploaded video files with detection."""
    return Response(generate_video_file_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    """Get current statistics."""
    return jsonify(state.stats)


@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update optimization settings (including video-specific settings)."""
    if request.method == 'POST':
        data = request.get_json() or {}
        # Live camera settings
        if 'frame_skip' in data:
            state.frame_skip = max(1, min(10, int(data['frame_skip'])))
        if 'scale_factor' in data:
            state.scale_factor = max(0.25, min(1.0, float(data['scale_factor'])))
        if 'cache_ttl' in data:
            state.cache_ttl = max(5, min(120, int(data['cache_ttl'])))
        # VIDEO-specific settings
        if 'video_frame_skip' in data:
            state.video_frame_skip = max(1, min(15, int(data['video_frame_skip'])))
        if 'video_scale_factor' in data:
            state.video_scale_factor = max(0.2, min(1.0, float(data['video_scale_factor'])))
        if 'video_cache_ttl' in data:
            state.video_cache_ttl = max(10, min(300, int(data['video_cache_ttl'])))
        if 'video_playback_speed' in data:
            state.video_playback_speed = max(1.0, min(10.0, float(data['video_playback_speed'])))
    
    return jsonify({
        # Live camera settings
        'frame_skip': state.frame_skip,
        'scale_factor': state.scale_factor,
        'cache_ttl': state.cache_ttl,
        'cache_size': len(state.classification_cache),
        # Video-specific settings (Mac optimized defaults)
        'video_frame_skip': state.video_frame_skip,
        'video_scale_factor': state.video_scale_factor,
        'video_cache_ttl': state.video_cache_ttl,
        'video_playback_speed': state.video_playback_speed
    })


@app.route('/api/history')
def get_history():
    """Get detection history from CSV."""
    csv_path = Path('logs/detections.csv')
    if not csv_path.exists():
        return jsonify({'data': [], 'summary': {}})
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Calculate summary
    if data:
        genders = Counter(row['gender'] for row in data)
        ages = Counter(row['age_bucket'] for row in data)
        persons = len(set(row['person_id'] for row in data))
        
        summary = {
            'total_detections': len(data),
            'unique_persons': persons,
            'gender_distribution': dict(genders),
            'age_distribution': dict(ages)
        }
    else:
        summary = {}
    
    # Return last 100 records
    return jsonify({
        'data': data[-100:],
        'summary': summary
    })


@app.route('/api/recent')
def get_recent():
    """Get recent detections."""
    return jsonify(list(state.recent_detections))


@app.route('/api/stop')
def stop_stream():
    """Stop video stream."""
    state.is_running = False
    return jsonify({'status': 'stopped'})


@app.route('/api/client_data')
def get_client_data():
    """
    Get current detections in client-specified JSON format.
    
    Returns JSON matching client specification:
    {
        "sensor_id": "SENSOR_001",
        "timestamp": "YYYY-MM-DD HH:MM:SS.microseconds",
        "detections": [{"id": 1, "age": 28.4, "gender": "male", ...}]
    }
    """
    return jsonify(state.client_formatter.format(state.stats['current_detections']))


# Video upload configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_uploaded_video(video_path: str, output_log: str) -> Dict:
    """
    Process an uploaded video file and generate detection logs.
    
    Returns:
        Dictionary with processing results
    """
    results = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {'error': 'Could not open video file', 'results': []}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Create temporary tracker for this video
    video_id = Path(video_path).stem
    tracker = KalmanPersonTracker(sensor_id=f'video_{video_id}')
    
    frame_count = 0
    unique_persons = set()
    
    # Process every 3rd frame for speed
    frame_skip = 3
    
    state.video_processing = True
    state.video_progress = 0
    state.video_results = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for speed
            if frame_count % frame_skip != 0:
                continue
            
            # Update progress
            state.video_progress = int((frame_count / total_frames) * 100)
            
            # Detect faces
            detections = state.detector.detect_faces(frame)
            tracked = tracker.update(detections)
            
            # Calculate timestamp in video
            video_time = frame_count / fps
            
            for person in tracked:
                bbox = person.bbox
                person_id = f"person_{person.id:04d}"
                confidence = person.confidence
                unique_persons.add(person_id)

                # Classify
                face_crop = extract_face_crop(frame, bbox, padding=0.2)
                result = state.classifier.classify(face_crop)

                # Format for client output
                detection_data = {
                    'video_timestamp': round(video_time, 2),
                    'frame': frame_count,
                    'person_id': person_id,
                    'id': person.id,
                    'age': round(result.raw_age or result.age_midpoint, 1),
                    'gender': result.gender.lower(),
                    'confidence': round(confidence, 2),
                    'bbox': {
                        'xc': (bbox[0] + bbox[2]) // 2,
                        'yc': (bbox[1] + bbox[3]) // 2,
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1],
                        'top': bbox[1],
                        'left': bbox[0]
                    }
                }
                results.append(detection_data)
                state.video_results.append(detection_data)
        
        # Save results to JSON log
        log_data = {
            'sensor_id': 'SENSOR_001',
            'video_file': Path(video_path).name,
            'processed_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f'),
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': round(total_frames / fps, 2),
            'unique_persons': len(unique_persons),
            'total_detections': len(results),
            'detections': results
        }
        
        with open(output_log, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        state.video_progress = 100
        
        return {
            'success': True,
            'video_file': Path(video_path).name,
            'total_frames': total_frames,
            'unique_persons': len(unique_persons),
            'total_detections': len(results),
            'log_file': output_log
        }
        
    except Exception as e:
        return {'error': str(e), 'results': []}
    finally:
        cap.release()
        state.video_processing = False


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """
    Upload a video file for analysis with visual playback.
    Returns the video stream URL to watch the video with detection boxes.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    # Stop any currently playing video
    state.video_is_playing = False
    time.sleep(0.5)  # Give time for stream to stop
    
    # Clean up previous video file if exists
    if state.video_file_path and Path(state.video_file_path).exists():
        try:
            os.remove(state.video_file_path)
        except:
            pass
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    video_path = UPLOAD_FOLDER / unique_name
    file.save(str(video_path))
    
    # Store path for streaming
    state.video_file_path = str(video_path)
    state.video_progress = 0
    state.video_results = []
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    return jsonify({
        'success': True,
        'video_file': filename,
        'total_frames': total_frames,
        'fps': round(fps, 2),
        'duration_seconds': round(duration, 2),
        'stream_url': '/video_file_feed'
    })


@app.route('/api/video_progress')
def video_progress():
    """Get current video processing progress."""
    return jsonify({
        'processing': state.video_processing,
        'progress': state.video_progress,
        'detections_so_far': len(state.video_results)
    })


@app.route('/api/video_results')
def video_results():
    """Get results from last video processing."""
    return jsonify({
        'results': state.video_results[-100:],  # Last 100 for preview
        'total': len(state.video_results)
    })


@app.route('/logs/<path:filename>')
def download_log(filename):
    """Download a log file."""
    return send_from_directory('logs', filename, as_attachment=True)


def run_server(host='0.0.0.0', port=5000, camera_id='web_cam'):
    """Run the web server."""
    initialize_pipeline(camera_id)
    print(f"Starting server at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == '__main__':
    run_server()
