import logging
import os
import threading
from datetime import datetime, timezone
from typing import List, Optional

import requests

from src.utils.types import TrackedPerson

logger = logging.getLogger(__name__)

SHOPPER_ENDPOINT = "/api/visioncraft/detection/v1/shopper-detection"
REQUEST_TIMEOUT = 5


def create_client_from_env() -> Optional["VisionCraftClient"]:
    
    api_url = os.environ.get("VISIONCRAFT_API_URL", "").strip()
    token = os.environ.get("VISIONCRAFT_API_TOKEN", "").strip()
    sensor_id = os.environ.get("VISIONCRAFT_SENSOR_ID", "cam-01").strip()

    if not api_url or not token:
        logger.info(
            "VISIONCRAFT_API_URL or VISIONCRAFT_API_TOKEN not set — "
            "API reporting disabled"
        )
        return None

    logger.info("VisionCraft API client configured for %s", api_url)
    return VisionCraftClient(api_url=api_url, token=token, sensor_id=sensor_id)


class VisionCraftClient:
    """Sends tracked person detections to the VisionCraft API. """

    def __init__(self, api_url: str, token: str, sensor_id: str = "cam-01"):
        self._api_url = api_url.rstrip("/")
        self._token = token
        self._sensor_id = sensor_id
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        })

    def update_token(self, token: str) -> None:
        """Update the access token (e.g. after refresh)."""
        self._token = token
        self._session.headers["Authorization"] = f"Bearer {token}"

    def send_detections(
        self,
        tracked_persons: List[TrackedPerson],
        timestamp: str | None = None,
    ) -> bool:
        """POST a frame of detections to /shopper-detection.
        """
        if not tracked_persons:
            return True

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "timestamp": timestamp,
            "sensorId": self._sensor_id,
            "detections": [p.to_client_format() for p in tracked_persons],
        }

        try:
            resp = self._session.post(
                f"{self._api_url}{SHOPPER_ENDPOINT}",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return True
            logger.warning(
                "API responded %d: %s", resp.status_code, resp.text[:200]
            )
            return False
        except requests.RequestException as exc:
            logger.warning("API request failed: %s", exc)
            return False

    def send_detections_async(
        self,
        tracked_persons: List[TrackedPerson],
        timestamp: str | None = None,
    ) -> None:
        """Fire-and-forget version that runs in a background thread."""
        thread = threading.Thread(
            target=self.send_detections,
            args=(tracked_persons, timestamp),
            daemon=True,
        )
        thread.start()
