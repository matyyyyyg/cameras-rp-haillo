"""
Client JSON Output Formatter

Formats detection data into the exact JSON structure required by the client's
web service specification.

Output Format:
{
    "sensor_id": "XXXXXXXXXXXXX",
    "timestamp": "2025-12-10 14:35:12.123456",
    "detections": [
        {
            "id": 1,
            "age": 28.4,
            "gender": "male",
            "confidence": 0.95,
            "bbox": {
                "xc": 320, "yc": 240,
                "width": 100, "height": 120,
                "top": 180, "left": 270
            }
        }
    ]
}


"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
import json


# Default sensor ID - can be configured per deployment
DEFAULT_SENSOR_ID = "SENSOR_001"


@dataclass
class ClientDetection:
    """Single detection in client format."""
    id: int
    age: float
    gender: str
    confidence: float
    bbox: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "age": round(self.age, 1),
            "gender": self.gender.lower(),
            "confidence": round(self.confidence, 2),
            "bbox": self.bbox
        }


def format_bbox_for_client(bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
    """
    Convert internal bbox (x1, y1, x2, y2) to client format.
    
    Client expects:
    - xc, yc: center coordinates
    - width, height: box dimensions
    - top, left: top-left corner coordinates
    
    Args:
        bbox: Tuple of (x1, y1, x2, y2)
        
    Returns:
        Dictionary with client bbox format
    """
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    xc = x1 + width // 2
    yc = y1 + height // 2
    
    return {
        "xc": xc,
        "yc": yc,
        "width": width,
        "height": height,
        "top": y1,
        "left": x1
    }


def format_timestamp_for_client(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp in client-required format.
    
    Format: YYYY-MM-DD HH:MM:SS.microseconds
    Example: 2025-12-10 14:35:12.123456
    
    Args:
        dt: Datetime object (defaults to now UTC)
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # Format: YYYY-MM-DD HH:MM:SS.microseconds
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def extract_numeric_id(person_id: str) -> int:
    """
    Extract numeric ID from internal person_id string.
    
    Internal format: "cam_01_person_0001" -> returns 1
    
    Args:
        person_id: Internal person ID string
        
    Returns:
        Integer ID
    """
    try:
        # Extract last part after underscore and convert to int
        parts = person_id.split("_")
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def format_detection_for_client(
    detection: Dict[str, Any],
    classification: Optional[Dict[str, Any]] = None
) -> ClientDetection:
    """
    Format a single detection for client output.
    
    Args:
        detection: Detection dict with bbox, confidence, person_id
        classification: Optional classification result with age, gender
        
    Returns:
        ClientDetection object
    """
    # Extract bbox
    bbox = detection.get("bbox", (0, 0, 0, 0))
    client_bbox = format_bbox_for_client(bbox)
    
    # Extract ID
    person_id = detection.get("person_id", "unknown_0")
    numeric_id = extract_numeric_id(person_id)
    
    # Get age - prefer raw_age (float) over age_midpoint (int)
    age = 25.0  # default
    if classification:
        if "raw_age" in classification and classification["raw_age"] is not None:
            age = float(classification["raw_age"])
        elif "age_midpoint" in classification:
            age = float(classification["age_midpoint"])
    elif "age_midpoint" in detection:
        age = float(detection["age_midpoint"])
    elif "age" in detection:
        age = float(detection["age"])
    
    # Get gender - ensure lowercase
    gender = "unknown"
    if classification:
        gender = classification.get("gender", "unknown")
    elif "gender" in detection:
        gender = detection["gender"]
    
    # Get confidence
    confidence = detection.get("confidence", 0.0)
    
    return ClientDetection(
        id=numeric_id,
        age=age,
        gender=gender.lower(),
        confidence=confidence,
        bbox=client_bbox
    )


def format_for_client(
    detections: List[Dict[str, Any]],
    sensor_id: str = DEFAULT_SENSOR_ID,
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Format all detections into client JSON structure.
    
    This is the main function to use for generating client-compatible output.
    
    Args:
        detections: List of detection dictionaries
        sensor_id: Sensor/camera identifier
        timestamp: Optional timestamp (defaults to now)
        
    Returns:
        Client-formatted JSON structure
    """
    client_detections = []
    
    for det in detections:
        client_det = format_detection_for_client(det)
        client_detections.append(client_det.to_dict())
    
    return {
        "sensor_id": sensor_id,
        "timestamp": format_timestamp_for_client(timestamp),
        "detections": client_detections
    }


def format_for_client_json(
    detections: List[Dict[str, Any]],
    sensor_id: str = DEFAULT_SENSOR_ID,
    timestamp: Optional[datetime] = None,
    indent: Optional[int] = None
) -> str:
    """
    Format detections as JSON string.
    
    Args:
        detections: List of detection dictionaries
        sensor_id: Sensor/camera identifier
        timestamp: Optional timestamp
        indent: JSON indentation (None for compact)
        
    Returns:
        JSON string
    """
    data = format_for_client(detections, sensor_id, timestamp)
    return json.dumps(data, indent=indent)


# Convenience class for stateful formatting
class ClientOutputFormatter:
    """
    Stateful formatter for client output.
    
    Maintains sensor_id configuration and provides consistent formatting.
    """
    
    def __init__(self, sensor_id: str = DEFAULT_SENSOR_ID):
        """
        Initialize formatter with sensor ID.
        
        Args:
            sensor_id: Unique identifier for this camera/sensor
        """
        self.sensor_id = sensor_id
    
    def format(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Format detections to client structure."""
        return format_for_client(detections, self.sensor_id, timestamp)
    
    def format_json(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
        indent: Optional[int] = None
    ) -> str:
        """Format detections to JSON string."""
        return format_for_client_json(
            detections, self.sensor_id, timestamp, indent
        )


if __name__ == "__main__":
    # Test the formatter
    test_detections = [
        {
            "person_id": "cam_01_person_0001",
            "bbox": (270, 180, 370, 300),
            "confidence": 0.95,
            "age_midpoint": 28,
            "gender": "Male"
        },
        {
            "person_id": "cam_01_person_0002", 
            "bbox": (455, 145, 545, 255),
            "confidence": 0.88,
            "age_midpoint": 34,
            "gender": "Female"
        }
    ]
    
    output = format_for_client(test_detections)
    print("Client JSON Output:")
    print(json.dumps(output, indent=2))
