import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time
import logging

# Try to import scipy for Hungarian algorithm, fallback to greedy if not available
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class KalmanBoxTracker:
 

    count = 0  # Class variable for unique ID generation

    def __init__(self, bbox: Tuple[int, int, int, int]):
        """
        Initialize tracker with bounding box.

        Args:
            bbox: (x1, y1, x2, y2) format
        """
        # State: [x, y, area, ratio, vx, vy, va]
        self.dim_x = 7
        self.dim_z = 4

        # State vector
        self.x = np.zeros((self.dim_x, 1))

        # State covariance matrix
        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.P *= 10.0

        # State transition matrix
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # area += va

        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # area
        self.H[3, 3] = 1  # ratio

        # Measurement noise
        self.R = np.eye(self.dim_z)
        self.R[2, 2] *= 10.0  # area measurement noise
        self.R[3, 3] *= 10.0  # ratio measurement noise

        # Process noise
        self.Q = np.eye(self.dim_x)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01
        self.Q[3, 3] *= 0.01

        # Initialize state from bbox
        self.x[:4] = self._bbox_to_z(bbox)

        # Tracking metadata
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

        # Store history for smoothing
        self.history: deque = deque(maxlen=30)

    def _bbox_to_z(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert bbox (x1,y1,x2,y2) to measurement (cx, cy, area, ratio)."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        area = w * h
        ratio = w / max(h, 1e-6)
        return np.array([[cx], [cy], [area], [ratio]])

    def _z_to_bbox(self, z: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert measurement (cx, cy, area, ratio) to bbox (x1,y1,x2,y2)."""
        cx, cy, area, ratio = z.flatten()
        area = max(area, 1.0)
        ratio = max(ratio, 0.1)
        w = np.sqrt(area * ratio)
        h = area / max(w, 1e-6)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Advance state and return predicted bbox.

        Returns:
            Predicted bounding box (x1, y1, x2, y2)
        """
        # Prevent negative area
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

        # Store prediction in history
        bbox = self._z_to_bbox(self.x[:4])
        self.history.append(bbox)

        return bbox

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """
        Update state with observed bbox.

        Args:
            bbox: Observed bounding box (x1, y1, x2, y2)
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Measurement
        z = self._bbox_to_z(bbox)

        # Kalman update
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current state as bbox."""
        return self._z_to_bbox(self.x[:4])


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1, box2: Boxes in (x1, y1, x2, y2) format

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_iou_matrix(
    boxes1: List[Tuple[int, int, int, int]],
    boxes2: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """Calculate IoU matrix between two sets of boxes."""
    n1, n2 = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((n1, n2))

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = calculate_iou(box1, box2)

    return iou_matrix


def hungarian_assignment(cost_matrix: np.ndarray, threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Perform Hungarian algorithm assignment with threshold.

    Args:
        cost_matrix: Cost matrix (negative IoU)
        threshold: Maximum cost (minimum IoU) for valid assignment

    Returns:
        Tuple of (matches, unmatched_rows, unmatched_cols)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    if SCIPY_AVAILABLE:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    else:
        # Fallback to greedy assignment
        row_indices, col_indices = [], []
        cost_copy = cost_matrix.copy()
        n_rows, n_cols = cost_matrix.shape

        for _ in range(min(n_rows, n_cols)):
            min_val = cost_copy.min()
            if min_val >= threshold:
                break
            idx = np.argmin(cost_copy)
            r, c = idx // n_cols, idx % n_cols
            row_indices.append(r)
            col_indices.append(c)
            cost_copy[r, :] = np.inf
            cost_copy[:, c] = np.inf

        row_indices = np.array(row_indices)
        col_indices = np.array(col_indices)

    matches = []
    unmatched_rows = list(range(cost_matrix.shape[0]))
    unmatched_cols = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] < threshold:
            matches.append((r, c))
            if r in unmatched_rows:
                unmatched_rows.remove(r)
            if c in unmatched_cols:
                unmatched_cols.remove(c)

    return matches, unmatched_rows, unmatched_cols


@dataclass
class TrackedPerson:
    """Represents a tracked person with all attributes."""
    id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    gender: str = "unknown"
    gender_confidence: float = 0.0
    age: float = 0.0
    hits: int = 1
    time_since_update: int = 0

    def to_client_format(self) -> Dict:
        """Convert to client JSON format."""
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        height = y2 - y1
        xc = x1 + width // 2
        yc = y1 + height // 2

        return {
            "id": self.id,
            "age": round(self.age, 1),
            "gender": self.gender.lower(),
            "confidence": round(self.confidence, 2),
            "bbox": {
                "xc": xc,
                "yc": yc,
                "width": width,
                "height": height,
                "top": y1,
                "left": x1
            }
        }


class KalmanPersonTracker:
    """
    Robust multi-person tracker using Kalman filter and Hungarian algorithm.

    Features:
    - Kalman filter for smooth tracking and prediction
    - Hungarian algorithm for optimal assignment
    - Handles occlusions and re-identification
    - Configurable max_age for track retention
    - ReID: Face embedding gallery for re-identifying returning persons
    """

    # ReID configuration
    REID_SIMILARITY_THRESHOLD = 0.5   # Cosine similarity threshold (increased for fewer false matches)
    REID_SECONDARY_THRESHOLD = 0.45   # Secondary threshold when spatial hint matches
    GALLERY_MAX_AGE_SECONDS = 600     # How long to remember lost tracks (10 min)
    GALLERY_MAX_SIZE = 100            # Maximum tracks to store in gallery
    USE_REID = True                   # Feature flag to enable/disable ReID
    EMBEDDING_HISTORY_SIZE = 5        # Store last N embeddings per track for averaging
    SPATIAL_MATCH_DISTANCE = 200      # Pixel distance for spatial hint matching

    def __init__(
        self,
        sensor_id: str = "SENSOR_001",
        max_age: int = 30,          # Frames to keep track without detection
        min_hits: int = 3,           # Minimum hits to confirm track
        iou_threshold: float = 0.25  # Minimum IoU for association
    ):
        """
        Initialize tracker.

        Args:
            sensor_id: Unique sensor/camera identifier
            max_age: Maximum frames to retain unmatched tracks
            min_hits: Minimum detections before track is confirmed
            iou_threshold: Minimum IoU for detection-track association
        """
        self.sensor_id = sensor_id
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.track_attributes: Dict[int, Dict] = {}  # Store gender/age per track

        # ReID: Gallery for storing embeddings of lost tracks
        self.track_gallery: Dict[int, Dict] = {}
        # Gallery entry format: {
        #     'embedding': np.ndarray (512-dim, averaged from history),
        #     'embedding_history': List[np.ndarray] (last N embeddings for averaging),
        #     'last_seen': float (timestamp),
        #     'last_position': Tuple[int, int] (center x, y when last seen),
        #     'attributes': dict (gender, age, etc.),
        #     'confidence': float (average detection confidence)
        # }

        # Track embedding history for averaging (more robust ReID)
        self.track_embedding_history: Dict[int, List[np.ndarray]] = {}

        # Reset ID counter for fresh start
        KalmanBoxTracker.count = 0

        logger.info(
            f"KalmanPersonTracker initialized: sensor={sensor_id}, "
            f"max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}, "
            f"reid={'enabled' if self.USE_REID else 'disabled'}"
        )

    def _find_gallery_match(self, embedding: np.ndarray, detection_center: Optional[Tuple[int, int]] = None) -> Optional[int]:
        """
        Find matching track ID in gallery using cosine similarity with spatial hints.

        Args:
            embedding: 512-dim L2-normalized face embedding
            detection_center: (x, y) center of new detection for spatial matching

        Returns:
            Track ID of best match, or None if no match above threshold
        """
        if embedding is None or not self.USE_REID:
            return None

        best_id = None
        best_sim = 0.0
        best_has_spatial_match = False
        current_time = time.time()
        expired_ids = []

        for track_id, data in self.track_gallery.items():
            # Check for expired entries
            if current_time - data['last_seen'] > self.GALLERY_MAX_AGE_SECONDS:
                expired_ids.append(track_id)
                continue

            gallery_emb = data.get('embedding')
            if gallery_emb is None:
                continue

            # Cosine similarity (embeddings are L2-normalized, so dot product = cosine sim)
            similarity = float(np.dot(embedding, gallery_emb))

            # Check spatial hint (if detection is near where track was last seen)
            has_spatial_match = False
            if detection_center is not None and data.get('last_position') is not None:
                last_pos = data['last_position']
                distance = np.sqrt((detection_center[0] - last_pos[0])**2 +
                                   (detection_center[1] - last_pos[1])**2)
                has_spatial_match = distance < self.SPATIAL_MATCH_DISTANCE

            # Use lower threshold if spatial hint matches
            threshold = self.REID_SECONDARY_THRESHOLD if has_spatial_match else self.REID_SIMILARITY_THRESHOLD

            if similarity > threshold and similarity > best_sim:
                # Prefer matches with spatial hint when similarity is close
                if best_id is not None and abs(similarity - best_sim) < 0.05:
                    if has_spatial_match and not best_has_spatial_match:
                        best_sim = similarity
                        best_id = track_id
                        best_has_spatial_match = has_spatial_match
                else:
                    best_sim = similarity
                    best_id = track_id
                    best_has_spatial_match = has_spatial_match

        # Clean up expired entries
        for track_id in expired_ids:
            del self.track_gallery[track_id]

        if best_id is not None:
            spatial_hint = " (spatial match)" if best_has_spatial_match else ""
            logger.info(f"ReID: Person {best_id} returned! (similarity={best_sim:.3f}{spatial_hint})")

        return best_id

    def _save_to_gallery(self, track_id: int, attrs: Dict, last_bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        Save a track's embedding to the gallery before deletion.

        Args:
            track_id: The track ID being removed
            attrs: Track attributes including embedding
            last_bbox: Last known bounding box (x1, y1, x2, y2)
        """
        if not self.USE_REID:
            return

        # Get averaged embedding from history if available
        embedding = None
        if track_id in self.track_embedding_history and self.track_embedding_history[track_id]:
            # Average all stored embeddings for more robust matching
            embeddings = self.track_embedding_history[track_id]
            embedding = np.mean(embeddings, axis=0)
            # Re-normalize after averaging
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        elif attrs.get('embedding') is not None:
            embedding = attrs['embedding']

        if embedding is None:
            return

        # Calculate center position from bbox
        last_position = None
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            last_position = ((x1 + x2) // 2, (y1 + y2) // 2)

        self.track_gallery[track_id] = {
            'embedding': embedding,
            'last_seen': time.time(),
            'last_position': last_position,
            'attributes': attrs.copy(),
            'confidence': attrs.get('confidence', 0.5)
        }
        logger.info(f"ReID: Saved track {track_id} to gallery (gallery size: {len(self.track_gallery)})")

        # Clean up embedding history
        if track_id in self.track_embedding_history:
            del self.track_embedding_history[track_id]

        # Limit gallery size by removing lowest confidence entry (not just oldest)
        if len(self.track_gallery) > self.GALLERY_MAX_SIZE:
            # Remove entry with lowest confidence that is also old
            candidates = [(k, v) for k, v in self.track_gallery.items()
                          if time.time() - v['last_seen'] > 60]  # At least 1 min old
            if candidates:
                lowest_conf_id = min(candidates, key=lambda x: x[1].get('confidence', 0))[0]
            else:
                # If all entries are recent, remove oldest
                lowest_conf_id = min(self.track_gallery, key=lambda k: self.track_gallery[k]['last_seen'])
            del self.track_gallery[lowest_conf_id]
            logger.debug(f"Gallery full, removed track {lowest_conf_id}")

    def update(self, detections: List[Dict]) -> List[TrackedPerson]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts with keys:
                - bbox: (x1, y1, x2, y2) or 'box': (x, y, w, h)
                - confidence: detection confidence
                - gender: predicted gender (optional)
                - gender_confidence: gender prediction confidence (optional)
                - age: predicted age (optional)

        Returns:
            List of TrackedPerson objects with stable IDs
        """
        # Convert detections to standard format
        det_bboxes = []
        det_data = []

        for det in detections:
            # Handle different bbox formats
            if 'bbox' in det:
                bbox = det['bbox']
            elif 'box' in det:
                x, y, w, h = det['box']
                bbox = (x, y, x + w, y + h)
            else:
                continue

            det_bboxes.append(bbox)
            det_data.append({
                'bbox': bbox,
                'confidence': det.get('confidence', det.get('face_confidence', 0.5)),
                'gender': det.get('gender', 'unknown'),
                'gender_confidence': det.get('gender_confidence', 0.0),
                'age': det.get('age', 0.0),
                'embedding': det.get('embedding')  # For ReID
            })

        # Predict new locations for existing trackers
        predicted_bboxes = []
        for tracker in self.trackers:
            pred_bbox = tracker.predict()
            predicted_bboxes.append(pred_bbox)

        # Associate detections with existing trackers
        matches, unmatched_trackers, unmatched_detections = self._associate(
            predicted_bboxes, det_bboxes
        )

        # Update matched trackers
        for tracker_idx, det_idx in matches:
            self.trackers[tracker_idx].update(det_bboxes[det_idx])

            # Update attributes (use exponential moving average for stability)
            track_id = self.trackers[tracker_idx].id
            det = det_data[det_idx]

            if track_id not in self.track_attributes:
                male_votes = 1 if det['gender'] in ('male', 'Male') else 0
                female_votes = 1 if det['gender'] in ('female', 'Female') else 0
                total_votes = male_votes + female_votes
                if total_votes > 0:
                    vote_ratio = max(male_votes, female_votes) / total_votes
                else:
                    vote_ratio = 0.0
                self.track_attributes[track_id] = {
                    'gender': det['gender'],
                    'gender_confidence': vote_ratio,
                    'age': det['age'],
                    'confidence': det['confidence'],
                    'gender_votes_male': male_votes,
                    'gender_votes_female': female_votes,
                    'embedding': det.get('embedding'),  # Store embedding for ReID
                }
            else:
                attrs = self.track_attributes[track_id]
                alpha = 0.3  # Smoothing factor for age/confidence

                # Majority voting for gender (original logic - more stable)
                if det['gender'] not in ('unknown', 'Unknown'):
                    if det['gender'] in ('male', 'Male'):
                        attrs['gender_votes_male'] = attrs.get('gender_votes_male', 0) + 1
                    else:
                        attrs['gender_votes_female'] = attrs.get('gender_votes_female', 0) + 1

                    m = attrs.get('gender_votes_male', 0)
                    f = attrs.get('gender_votes_female', 0)
                    total = m + f
                    if total > 0:
                        attrs['gender'] = 'male' if m >= f else 'female'
                        attrs['gender_confidence'] = max(m, f) / total

                if det['age'] > 0:
                    if attrs['age'] > 0:
                        attrs['age'] = alpha * det['age'] + (1 - alpha) * attrs['age']
                    else:
                        attrs['age'] = det['age']

                attrs['confidence'] = alpha * det['confidence'] + (1 - alpha) * attrs['confidence']

                # Update embedding with latest (for ReID if track is lost later)
                if det.get('embedding') is not None:
                    attrs['embedding'] = det['embedding']
                    # Store in embedding history for averaging
                    if track_id not in self.track_embedding_history:
                        self.track_embedding_history[track_id] = []
                    self.track_embedding_history[track_id].append(det['embedding'])
                    # Keep only last N embeddings
                    if len(self.track_embedding_history[track_id]) > self.EMBEDDING_HISTORY_SIZE:
                        self.track_embedding_history[track_id].pop(0)

        # Create new trackers for unmatched detections (with ReID check)
        for det_idx in unmatched_detections:
            det = det_data[det_idx]
            embedding = det.get('embedding')
            bbox = det_bboxes[det_idx]

            # Calculate detection center for spatial hint
            det_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Try to match with gallery for ReID (with spatial hint)
            gallery_match_id = self._find_gallery_match(embedding, det_center)

            if gallery_match_id is not None:
                # Resurrect track with original ID
                new_tracker = KalmanBoxTracker(det_bboxes[det_idx])
                new_tracker.id = gallery_match_id  # REUSE OLD ID
                KalmanBoxTracker.count -= 1  # Don't increment counter (was already incremented in __init__)

                # Restore attributes from gallery and update with new detection
                gallery_data = self.track_gallery.pop(gallery_match_id)
                restored_attrs = gallery_data['attributes'].copy()
                # Update with new embedding
                if embedding is not None:
                    restored_attrs['embedding'] = embedding
                    # Start fresh embedding history with new detection
                    self.track_embedding_history[gallery_match_id] = [embedding]
                self.track_attributes[gallery_match_id] = restored_attrs
            else:
                # Create truly new track
                new_tracker = KalmanBoxTracker(det_bboxes[det_idx])

                male_votes = 1 if det['gender'] in ('male', 'Male') else 0
                female_votes = 1 if det['gender'] in ('female', 'Female') else 0
                total_votes = male_votes + female_votes
                self.track_attributes[new_tracker.id] = {
                    'gender': det['gender'],
                    'gender_confidence': max(male_votes, female_votes) / total_votes if total_votes > 0 else 0.0,
                    'age': det['age'],
                    'confidence': det['confidence'],
                    'gender_votes_male': male_votes,
                    'gender_votes_female': female_votes,
                    'embedding': embedding,  # Store embedding for ReID
                }
                # Start embedding history for new track
                if embedding is not None:
                    self.track_embedding_history[new_tracker.id] = [embedding]

            self.trackers.append(new_tracker)

        # Save to gallery before removing dead trackers (for ReID)
        for tracker in self.trackers:
            if tracker.time_since_update > self.max_age:
                attrs = self.track_attributes.get(tracker.id, {})
                last_bbox = tracker.get_state()
                self._save_to_gallery(tracker.id, attrs, last_bbox)

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Build output - only confirmed tracks
        results = []
        for tracker in self.trackers:
            # Only output if track has enough hits or was just updated
            if tracker.hits >= self.min_hits or tracker.time_since_update == 0:
                attrs = self.track_attributes.get(tracker.id, {})

                person = TrackedPerson(
                    id=tracker.id,
                    bbox=tracker.get_state(),
                    confidence=attrs.get('confidence', 0.5),
                    gender=attrs.get('gender', 'unknown'),
                    gender_confidence=attrs.get('gender_confidence', 0.0),
                    age=attrs.get('age', 0.0),
                    hits=tracker.hits,
                    time_since_update=tracker.time_since_update
                )
                results.append(person)

        return results

    def _associate(
        self,
        tracker_bboxes: List[Tuple[int, int, int, int]],
        detection_bboxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with trackers using Hungarian algorithm.

        Returns:
            Tuple of (matches, unmatched_trackers, unmatched_detections)
        """
        if len(tracker_bboxes) == 0:
            return [], [], list(range(len(detection_bboxes)))

        if len(detection_bboxes) == 0:
            return [], list(range(len(tracker_bboxes))), []

        # Calculate IoU matrix
        iou_matrix = calculate_iou_matrix(tracker_bboxes, detection_bboxes)

        # Convert to cost matrix (negative IoU)
        cost_matrix = -iou_matrix

        # Run Hungarian algorithm
        matches, unmatched_trackers, unmatched_detections = hungarian_assignment(
            cost_matrix, -self.iou_threshold
        )

        return matches, unmatched_trackers, unmatched_detections

    def get_active_count(self) -> int:
        """Get number of active tracks."""
        return len([t for t in self.trackers if t.time_since_update == 0])

    def get_total_count(self) -> int:
        """Get total unique persons tracked."""
        return KalmanBoxTracker.count

    def get_gallery_stats(self) -> Dict:
        """Get ReID gallery statistics for monitoring."""
        current_time = time.time()
        active_entries = [
            (track_id, current_time - data['last_seen'])
            for track_id, data in self.track_gallery.items()
        ]
        return {
            "gallery_size": len(self.track_gallery),
            "entries": [{"id": tid, "age_seconds": round(age, 1)} for tid, age in active_entries],
            "reid_enabled": self.USE_REID,
            "threshold": self.REID_SIMILARITY_THRESHOLD
        }

    def reset(self) -> None:
        """Reset tracker state."""
        self.trackers.clear()
        self.track_attributes.clear()
        self.track_gallery.clear()  # Clear ReID gallery
        self.track_embedding_history.clear()  # Clear embedding history
        KalmanBoxTracker.count = 0
        logger.info("Tracker reset")


def format_output_json(
    tracked_persons: List[TrackedPerson],
    sensor_id: str,
    timestamp: Optional[str] = None
) -> Dict:
    """
    Format tracked persons to client JSON specification.

    Args:
        tracked_persons: List of TrackedPerson objects
        sensor_id: Sensor/camera identifier
        timestamp: Optional timestamp string (auto-generated if None)

    Returns:
        Client-formatted JSON dict
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    return {
        "sensor_id": sensor_id,
        "timestamp": timestamp,
        "detections": [p.to_client_format() for p in tracked_persons]
    }
