from dataclasses import dataclass
from typing import Dict, Tuple


def get_age_bucket(age: float) -> str:
    """Convert continuous age to age bucket string."""
    age = int(age)
    if age <= 2:
        return "(0-2)"
    elif age <= 6:
        return "(3-6)"
    elif age <= 12:
        return "(7-12)"
    elif age <= 19:
        return "(13-19)"
    elif age <= 29:
        return "(20-29)"
    elif age <= 39:
        return "(30-39)"
    elif age <= 49:
        return "(40-49)"
    elif age <= 59:
        return "(50-59)"
    else:
        return "(60+)"


@dataclass
class ClassificationResult:
    """Age and gender classification result."""
    age: float                # Raw continuous age (e.g., 32.5)
    age_bucket: str           # e.g., "(30-39)"
    gender: str               # "male" or "female"
    gender_confidence: float  # 0.0-1.0


@dataclass
class TrackedPerson:
    """Represents a tracked person with all attributes."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
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
