import logging
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from src.utils.types import TrackedPerson

logger = logging.getLogger(__name__)

# API paths (all relative to base URL)
_AUTH_LOGIN = "/visioncraft/detection/v1/auth/login"
_AUTH_REFRESH = "/visioncraft/detection/v1/auth/refresh"
_AUTH_REVOKE = "/visioncraft/detection/v1/auth/revoke"
_DETECTION = "/visioncraft/detection/v1/shopper-detection"

REQUEST_TIMEOUT = 5

# Refresh the access token this many seconds before it actually expires,
_REFRESH_MARGIN = timedelta(seconds=30)


def create_client_from_env() -> Optional["VisionCraftClient"]:

    api_url = os.environ.get("VISIONCRAFT_API_URL", "").strip()
    username = os.environ.get("VISIONCRAFT_USERNAME", "").strip()
    password = os.environ.get("VISIONCRAFT_PASSWORD", "").strip()
    sensor_id = os.environ.get("VISIONCRAFT_SENSOR_ID", "cam-01").strip()

    if not api_url or not username or not password:
        logger.info(
            "VISIONCRAFT_API_URL, USERNAME, or PASSWORD not set — "
        )
        return None

    client = VisionCraftClient(
        api_url=api_url,
        username=username,
        password=password,
        sensor_id=sensor_id,
    )

    if not client.login():
        logger.error("VisionCraft login failed")
        return None

    logger.info("VisionCraft API client authenticated for %s", api_url)
    return client


class VisionCraftClient:
    """Sends tracked-person detections to the VisionCraft Detection API.
    """

    def __init__(
        self,
        api_url: str,
        username: str,
        password: str,
        sensor_id: str = "cam-01",
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._username = username
        self._password = password
        self._sensor_id = sensor_id

        # JWT state — guarded by _token_lock for thread safety
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._access_expires: Optional[datetime] = None
        self._refresh_expires: Optional[datetime] = None
        self._token_lock = threading.Lock()

        # Separate lock so only one thread refreshes at a time (TOCTOU fix)
        self._refresh_lock = threading.Lock()

        # Shared session (connection pooling)
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    # Authentication

    def login(self) -> bool:
        
        try:
            resp = self._session.post(
                f"{self._api_url}{_AUTH_LOGIN}",
                json={
                    "username": self._username,
                    "password": self._password,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.error(
                    "Login failed (HTTP %d): %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False

            return self._store_tokens(resp.json())

        except requests.RequestException as exc:
            logger.error("Login request failed: %s", exc)
            return False

    def _refresh(self) -> bool:
        with self._token_lock:
            refresh_tok = self._refresh_token

        if not refresh_tok:
            logger.warning("No refresh token available — cannot refresh")
            return False

        try:
            resp = self._session.post(
                f"{self._api_url}{_AUTH_REFRESH}",
                json={"refresh_token": refresh_tok},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Token refresh failed (HTTP %d): %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False

            return self._store_tokens(resp.json())

        except requests.RequestException as exc:
            logger.warning("Token refresh request failed: %s", exc)
            return False

    def revoke(self) -> None:
        """Invalidate the current tokens (call on shutdown)."""
        with self._token_lock:
            access_tok = self._access_token
            refresh_tok = self._refresh_token

        if not access_tok or not refresh_tok:
            return

        try:
            # Per-request header — no shared state mutation
            self._session.post(
                f"{self._api_url}{_AUTH_REVOKE}",
                json={"refresh_token": refresh_tok},
                headers={"Authorization": f"Bearer {access_tok}"},
                timeout=REQUEST_TIMEOUT,
            )
            logger.info("Tokens revoked successfully")
        except requests.RequestException as exc:
            logger.warning("Token revoke failed (non-critical): %s", exc)
        finally:
            with self._token_lock:
                self._access_token = None
                self._refresh_token = None

    def _store_tokens(self, data: dict) -> bool:
        """Parse a token response and store all four fields thread-safely."""
        try:
            access_token = data["access_token"]
            refresh_token = data["refresh_token"]
            access_exp = _parse_datetime(data["access_token_expiration"])
            refresh_exp = _parse_datetime(data["refresh_token_expiration"])
        except (KeyError, ValueError) as exc:
            logger.error("Invalid token response: %s", exc)
            return False

        with self._token_lock:
            self._access_token = access_token
            self._refresh_token = refresh_token
            self._access_expires = access_exp
            self._refresh_expires = refresh_exp

        logger.debug(
            "Tokens stored — access expires %s, refresh expires %s",
            access_exp.isoformat(),
            refresh_exp.isoformat(),
        )
        return True

    def _ensure_valid_token(self) -> bool:
        """Refresh the access token if it is about to expire.

        Uses _refresh_lock so only one thread performs the refresh
        while others wait and then re-check (avoids TOCTOU race).
        Returns True if a valid token is available afterwards.
        """
        with self._token_lock:
            access_exp = self._access_expires
            refresh_exp = self._refresh_expires

        if access_exp is None:
            return False

        now = datetime.now(timezone.utc)

        # Access token still fresh — nothing to do
        if now < access_exp - _REFRESH_MARGIN:
            return True

        # Only one thread refreshes; others wait then re-check
        with self._refresh_lock:
            # Re-read under refresh lock — another thread may have just refreshed
            with self._token_lock:
                access_exp = self._access_expires

            if access_exp and now < access_exp - _REFRESH_MARGIN:
                return True  # Another thread already refreshed

            # Refresh token also expired — need full re-login
            if refresh_exp and now >= refresh_exp:
                logger.info("Refresh token expired — attempting re-login")
                return self.login()

            # Normal case: refresh the access token
            logger.info("Access token expiring soon — refreshing")
            return self._refresh()

    # Detection sending

    def send_detections(
        self,
        tracked_persons: List[TrackedPerson],
        timestamp: Optional[str] = None,
    ) -> bool:
        """POST a frame of detections to the detection endpoint
        """
        if not tracked_persons:
            return True

        if not self._ensure_valid_token():
            logger.warning("No valid token — skipping detection send")
            return False

        if timestamp is None:
            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

        # Read token under lock, pass as per-request header (thread-safe)
        with self._token_lock:
            token = self._access_token

        payload = {
            "sensorId": self._sensor_id,
            "timestamp": timestamp,
            "detections": [p.to_client_format() for p in tracked_persons],
        }

        try:
            resp = self._session.post(
                f"{self._api_url}{_DETECTION}",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code in (200, 202):
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
        timestamp: Optional[str] = None,
    ) -> None:
        """Fire-and-forget version that runs in a background thread."""
        thread = threading.Thread(
            target=self.send_detections,
            args=(tracked_persons, timestamp),
            daemon=True,
        )
        thread.start()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_datetime(value: str) -> datetime:
    """Parse an ISO 8601 datetime string to a timezone-aware datetime.
    """
    # Truncate fractional seconds to 6 digits max (.NET sends 7)
    value = re.sub(r"(\.\d{6})\d+", r"\1", value)

    # Python 3.9/3.10 fromisoformat doesn't accept 'Z' suffix
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    dt = datetime.fromisoformat(value)

    # Ensure timezone-aware (assume UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
