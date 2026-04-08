"""
Person Tracker using Kalman Filter and Hungarian Algorithm.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..utils.types import TrackedPerson

logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    """
    Kalman Filter for tracking a single bounding box.

    State vector: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    """

    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int]):
        self.dim_x = 7
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1))

        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1

        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        self.R = np.eye(self.dim_z)
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0

        self.Q = np.eye(self.dim_x)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01
        self.Q[3, 3] *= 0.01

        self.x[:4] = self._bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

        self.history: deque = deque(maxlen=30)

    def _bbox_to_z(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        area = w * h
        ratio = w / max(h, 1e-6)
        return np.array([[cx], [cy], [area], [ratio]])

    def _z_to_bbox(self, z: np.ndarray) -> Tuple[int, int, int, int]:
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
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

        bbox = self._z_to_bbox(self.x[:4])
        self.history.append(bbox)

        return bbox

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_state(self) -> Tuple[int, int, int, int]:
        return self._z_to_bbox(self.x[:4])


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate IoU between two bounding boxes in (x1, y1, x2, y2) format."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

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


class KalmanPersonTracker:
    """
    Multi-person tracker using Kalman filter and Hungarian algorithm.

    Features:
    - Kalman filter for smooth tracking and prediction
    - Hungarian algorithm for optimal assignment
    - Handles occlusions
    - Gender majority voting and age EMA
    """

    def __init__(
        self,
        sensor_id: str = "SENSOR_001",
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.25
    ):
        self.sensor_id = sensor_id
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.track_attributes: Dict[int, Dict] = {}

        KalmanBoxTracker.count = 0

        logger.info(
            f"KalmanPersonTracker initialized: sensor={sensor_id}, "
            f"max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}"
        )

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
        det_bboxes = []
        det_data = []

        for det in detections:
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
                }
            else:
                attrs = self.track_attributes[track_id]
                alpha = 0.3

                # Majority voting for gender
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

                # Age EMA
                if det['age'] > 0:
                    if attrs['age'] > 0:
                        attrs['age'] = alpha * det['age'] + (1 - alpha) * attrs['age']
                    else:
                        attrs['age'] = det['age']

                attrs['confidence'] = alpha * det['confidence'] + (1 - alpha) * attrs['confidence']

        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            det = det_data[det_idx]
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
            }
            self.trackers.append(new_tracker)

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Build output - only confirmed tracks
        results = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or tracker.time_since_update == 0:
                attrs = self.track_attributes.get(tracker.id, {})
                raw_age = attrs.get('age', 0.0)
                person = TrackedPerson(
                    id=tracker.id,
                    bbox=tracker.get_state(),
                    confidence=attrs.get('confidence', 0.5),
                    gender=attrs.get('gender', 'unknown'),
                    gender_confidence=attrs.get('gender_confidence', 0.0),
                    age=raw_age,
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
        """Associate detections with trackers using Hungarian algorithm."""
        if len(tracker_bboxes) == 0:
            return [], [], list(range(len(detection_bboxes)))
        if len(detection_bboxes) == 0:
            return [], list(range(len(tracker_bboxes))), []

        iou_matrix = calculate_iou_matrix(tracker_bboxes, detection_bboxes)
        cost_matrix = -iou_matrix
        return hungarian_assignment(cost_matrix, -self.iou_threshold)

    def get_active_count(self) -> int:
        return len([t for t in self.trackers if t.time_since_update == 0])

    def get_total_count(self) -> int:
        return KalmanBoxTracker.count

    def reset(self) -> None:
        self.trackers.clear()
        self.track_attributes.clear()
        KalmanBoxTracker.count = 0
        logger.info("Tracker reset")


def format_output_json(
    tracked_persons: List[TrackedPerson],
    sensor_id: str,
    timestamp: Optional[str] = None
) -> Dict:
    """Format tracked persons to client JSON specification."""
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    return {
        "sensor_id": sensor_id,
        "timestamp": timestamp,
        "detections": [p.to_client_format() for p in tracked_persons]
    }
