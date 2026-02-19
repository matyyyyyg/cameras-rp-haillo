import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class JSONLogger:
    """Logger for client JSON format output (JSON Lines)."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear/create file
        with open(self.output_path, 'w') as f:
            pass

        self.frame_count = 0
        logger.info(f"JSON logger initialized: {output_path}")

    def log(self, client_output: Dict) -> None:
        """Append a JSON object to the log file."""
        with open(self.output_path, 'a') as f:
            json.dump(client_output, f)
            f.write('\n')
        self.frame_count += 1

    def get_stats(self) -> Dict:
        return {"frames_logged": self.frame_count, "output_path": str(self.output_path)}
