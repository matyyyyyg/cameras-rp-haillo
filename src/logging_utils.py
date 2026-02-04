"""
Logging Utilities Module

This module provides CSV and JSONL logging for detection events.
"""

from typing import Dict, Optional, List, Any, Union
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import csv
import json
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    """Data class for a single detection event."""
    timestamp: str
    person_id: str
    camera_id: str
    age_midpoint: int
    age_bucket: str
    age_confidence: float
    gender: str
    gender_confidence: float
    bbox: Optional[str] = None
    detection_confidence: Optional[float] = None
    
    @classmethod
    def create(cls, person_id: str, camera_id: str, age_midpoint: int,
               age_bucket: str, age_confidence: float, gender: str,
               gender_confidence: float, bbox: Optional[tuple] = None,
               detection_confidence: Optional[float] = None,
               timestamp: Optional[datetime] = None) -> "DetectionEvent":
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        ts_str = timestamp.isoformat()
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else None
        return cls(
            timestamp=ts_str, person_id=person_id, camera_id=camera_id,
            age_midpoint=age_midpoint, age_bucket=age_bucket,
            age_confidence=round(age_confidence, 4), gender=gender,
            gender_confidence=round(gender_confidence, 4), bbox=bbox_str,
            detection_confidence=round(detection_confidence, 4) if detection_confidence else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp, "person_id": self.person_id,
            "camera_id": self.camera_id, "age_midpoint": self.age_midpoint,
            "age_bucket": self.age_bucket, "age_confidence": self.age_confidence,
            "gender": self.gender, "gender_confidence": self.gender_confidence
        }


class CSVLogger:
    """Thread-safe CSV logger for detection events."""
    FIELDNAMES = ["timestamp", "person_id", "camera_id", "age_midpoint",
                  "age_bucket", "age_confidence", "gender", "gender_confidence"]
    
    def __init__(self, log_path: Union[str, Path], create_dirs: bool = True):
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        if create_dirs:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self._write_headers()
    
    def _write_headers(self) -> None:
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()
    
    def log(self, event: DetectionEvent) -> None:
        with self._lock:
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(event.to_csv_row())
    
    def log_batch(self, events: List[DetectionEvent]) -> None:
        if not events:
            return
        with self._lock:
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                for event in events:
                    writer.writerow(event.to_csv_row())


class JSONLLogger:
    """Thread-safe JSON Lines logger."""
    def __init__(self, log_path: Union[str, Path], create_dirs: bool = True):
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        if create_dirs:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()
    
    def log(self, event: DetectionEvent) -> None:
        with self._lock:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                json.dump(event.to_dict(), f)
                f.write('\n')
    
    def log_batch(self, events: List[DetectionEvent]) -> None:
        if not events:
            return
        with self._lock:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                for event in events:
                    json.dump(event.to_dict(), f)
                    f.write('\n')


class DetectionLogger:
    """Combined logger for CSV and optionally JSONL."""
    def __init__(self, csv_path: Union[str, Path], jsonl_path: Optional[Union[str, Path]] = None,
                 enable_jsonl: bool = False):
        self.csv_logger = CSVLogger(csv_path)
        self.jsonl_logger = JSONLLogger(jsonl_path or Path(csv_path).with_suffix('.jsonl')) if enable_jsonl else None
        self.event_count = 0
    
    def log(self, event: DetectionEvent) -> None:
        self.csv_logger.log(event)
        if self.jsonl_logger:
            self.jsonl_logger.log(event)
        self.event_count += 1
    
    def log_batch(self, events: List[DetectionEvent]) -> None:
        self.csv_logger.log_batch(events)
        if self.jsonl_logger:
            self.jsonl_logger.log_batch(events)
        self.event_count += len(events)
    
    def log_detection(self, person_id: str, camera_id: str, age_midpoint: int,
                      age_bucket: str, age_confidence: float, gender: str,
                      gender_confidence: float, bbox: Optional[tuple] = None,
                      detection_confidence: Optional[float] = None) -> None:
        event = DetectionEvent.create(
            person_id=person_id, camera_id=camera_id, age_midpoint=age_midpoint,
            age_bucket=age_bucket, age_confidence=age_confidence, gender=gender,
            gender_confidence=gender_confidence, bbox=bbox, detection_confidence=detection_confidence
        )
        self.log(event)


def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    """Configure Python logging for the application."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers, force=True
    )
