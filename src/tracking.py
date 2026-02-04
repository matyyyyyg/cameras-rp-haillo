"""
Simple Multi-Object Tracking Module

This module provides basic person tracking using IoU (Intersection over Union)
based association between frames.

The tracker assigns stable person IDs to detected faces across frames,
enabling consistent logging and analytics.


Created: 2024
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time
import logging

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """
    Represents a tracked person/face across frames.
    
    Attributes:
        person_id: Unique identifier for this tracked person
        bbox: Current bounding box (x1, y1, x2, y2)
        confidence: Detection confidence
        last_seen: Timestamp when last detected
        age_frames: Number of frames this object has been tracked
        hits: Number of successful detections
        misses: Consecutive frames without detection
    """
    person_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    last_seen: float = field(default_factory=time.time)
    age_frames: int = 0
    hits: int = 1
    misses: int = 0
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Update the tracked object with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.age_frames += 1
        self.hits += 1
        self.misses = 0
    
    def mark_missed(self) -> None:
        """Mark this object as missed in current frame."""
        self.age_frames += 1
        self.misses += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "age_frames": self.age_frames,
            "hits": self.hits,
            "misses": self.misses
        }


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate intersection area
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calculate_center_distance(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Calculate Euclidean distance between box centers.
    
    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)
        
    Returns:
        Distance between centers in pixels
    """
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


class SimpleTracker:
    """
    Simple IoU-based multi-object tracker.
    
    Associates detections between frames using IoU overlap.
    Assigns unique person IDs to tracks that persist across frames.
    
    Attributes:
        iou_threshold: Minimum IoU to associate detection with track
        max_age: Maximum frames to keep track without detection
        min_hits: Minimum detections before track is confirmed
    """
    
    def __init__(
        self,
        camera_id: str = "cam_01",
        iou_threshold: float = 0.2,  # Lower threshold for more matches
        max_age: int = 90,  # ~3 seconds at 30fps before losing track
        min_hits: int = 3,
        max_distance: float = 200.0  # Larger distance tolerance for fallback
    ):
        """
        Initialize the tracker.
        
        Args:
            camera_id: Camera identifier for generating person IDs
            iou_threshold: Minimum IoU for association
            max_age: Max frames to keep unmatched track
            min_hits: Min detections to confirm track
            max_distance: Max distance (pixels) for association fallback
        """
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        
        # Active tracks
        self.tracks: Dict[str, TrackedObject] = {}
        
        # ID counter for generating unique person IDs
        self._next_id = 1
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_removed = 0
        
        logger.info(
            f"SimpleTracker initialized: camera={camera_id}, "
            f"iou_threshold={iou_threshold}, max_age={max_age}"
        )
    
    def _generate_person_id(self) -> str:
        """Generate a unique person ID."""
        person_id = f"{self.camera_id}_person_{self._next_id:04d}"
        self._next_id += 1
        return person_id
    
    def _compute_cost_matrix(
        self,
        detections: List[Dict]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute cost matrix between current tracks and new detections.
        
        Uses negative IoU as cost (lower is better).
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Tuple of (cost_matrix, track_ids_list)
        """
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_detections = len(detections)
        
        if n_tracks == 0 or n_detections == 0:
            return np.array([]), track_ids
        
        # Cost matrix: rows = tracks, cols = detections
        cost_matrix = np.zeros((n_tracks, n_detections))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                iou = calculate_iou(track.bbox, det["bbox"])
                # Use negative IoU as cost (we want to maximize IoU)
                cost_matrix[i, j] = -iou
        
        return cost_matrix, track_ids
    
    def _greedy_assignment(
        self,
        cost_matrix: np.ndarray,
        track_ids: List[str],
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Perform greedy assignment based on cost matrix.
        Uses IoU first, then falls back to center distance for unmatched.
        
        Args:
            cost_matrix: Cost matrix (negative IoU)
            track_ids: List of track IDs
            detections: List of detections
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if cost_matrix.size == 0:
            unmatched_tracks = list(range(len(track_ids)))
            unmatched_detections = list(range(len(detections)))
            return [], unmatched_tracks, unmatched_detections
        
        n_tracks, n_detections = cost_matrix.shape
        
        matched = []
        matched_track_indices = set()
        matched_det_indices = set()
        
        # Phase 1: IoU-based matching
        cost_copy = cost_matrix.copy()
        while True:
            min_val = cost_copy.min()
            if min_val >= -self.iou_threshold:
                break
            
            idx = np.argmin(cost_copy)
            track_idx = idx // n_detections
            det_idx = idx % n_detections
            
            matched.append((track_idx, det_idx))
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)
            
            cost_copy[track_idx, :] = np.inf
            cost_copy[:, det_idx] = np.inf
        
        # Phase 2: Distance-based fallback for unmatched
        remaining_tracks = [i for i in range(n_tracks) if i not in matched_track_indices]
        remaining_dets = [j for j in range(n_detections) if j not in matched_det_indices]
        
        if remaining_tracks and remaining_dets:
            # Build distance matrix for remaining pairs
            for track_idx in remaining_tracks:
                track = self.tracks[track_ids[track_idx]]
                best_det_idx = None
                best_dist = self.max_distance
                
                for det_idx in remaining_dets:
                    if det_idx in matched_det_indices:
                        continue
                    det = detections[det_idx]
                    dist = calculate_center_distance(track.bbox, det["bbox"])
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = det_idx
                
                if best_det_idx is not None:
                    matched.append((track_idx, best_det_idx))
                    matched_track_indices.add(track_idx)
                    matched_det_indices.add(best_det_idx)
        
        # Find unmatched
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_indices]
        unmatched_detections = [j for j in range(n_detections) if j not in matched_det_indices]
        
        return matched, unmatched_tracks, unmatched_detections
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections and return tracked objects.
        
        This is the main entry point for the tracker. Call this for each frame
        with the list of face detections.
        
        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            
        Returns:
            List of tracked detection dicts with added 'person_id' field
        """
        # Compute cost matrix
        cost_matrix, track_ids = self._compute_cost_matrix(detections)
        
        # Perform assignment
        matched, unmatched_tracks, unmatched_dets = self._greedy_assignment(
            cost_matrix, track_ids, detections
        )
        
        # Update matched tracks
        results = []
        for track_idx, det_idx in matched:
            track_id = track_ids[track_idx]
            det = detections[det_idx]
            
            self.tracks[track_id].update(det["bbox"], det["confidence"])
            
            # Add to results with person_id
            result = det.copy()
            result["person_id"] = track_id
            results.append(result)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            person_id = self._generate_person_id()
            
            new_track = TrackedObject(
                person_id=person_id,
                bbox=det["bbox"],
                confidence=det["confidence"]
            )
            self.tracks[person_id] = new_track
            self.total_tracks_created += 1
            
            # Add to results
            result = det.copy()
            result["person_id"] = person_id
            results.append(result)
            
            logger.debug(f"Created new track: {person_id}")
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        logger.debug(f"Tracker update: {len(results)} tracked objects, {len(self.tracks)} active tracks")
        
        return results
    
    def _remove_stale_tracks(self) -> None:
        """Remove tracks that haven't been seen for too long."""
        stale_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.misses > self.max_age
        ]
        
        for track_id in stale_ids:
            del self.tracks[track_id]
            self.total_tracks_removed += 1
            logger.debug(f"Removed stale track: {track_id}")
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """
        Get list of currently active tracks.
        
        Returns:
            List of TrackedObject instances
        """
        return list(self.tracks.values())
    
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """
        Get tracks that have been confirmed (enough hits).
        
        Returns:
            List of confirmed TrackedObject instances
        """
        return [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
    
    def reset(self) -> None:
        """Reset the tracker, clearing all tracks."""
        self.tracks.clear()
        self._next_id = 1
        logger.info("Tracker reset")
    
    def get_statistics(self) -> Dict:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary with tracker statistics
        """
        return {
            "camera_id": self.camera_id,
            "active_tracks": len(self.tracks),
            "confirmed_tracks": len(self.get_confirmed_tracks()),
            "total_created": self.total_tracks_created,
            "total_removed": self.total_tracks_removed,
            "next_id": self._next_id
        }
