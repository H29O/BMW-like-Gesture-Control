"""
macOS System Volume Controller using AppleScript (osascript).
"""

import subprocess
import logging

logger = logging.getLogger(__name__)


class VolumeController:
    """Controls macOS system volume via osascript."""

    def __init__(self):
        # Query initial actual macOS volume
        self._current_volume = self.get_volume()

    def get_volume(self) -> int:
        """Obtain current system output volume (0-100)."""
        try:
            cmd = ["osascript", "-e", "output volume of (get volume settings)"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1.0
            )
            raw = result.stdout.strip()
            if raw.isdigit():
                self._current_volume = max(0, min(100, int(raw)))
            return self._current_volume
        except Exception as e:
            logger.debug("Failed to get system volume: %s", e)
            return self._current_volume if self._current_volume is not None else 50

    def set_volume(self, volume: int) -> int:
        """Set system output volume clamped between 0 and 100."""
        clamped = max(0, min(100, int(round(volume))))
        try:
            cmd = ["osascript", "-e", f"set volume output volume {clamped}"]
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1.0
            )
            self._current_volume = clamped
        except Exception as e:
            logger.debug("Failed to set system volume: %s", e)
        return self._current_volume

    def increase_volume(self, step: int = 1) -> int:
        """Increase volume by step and return new volume."""
        return self.set_volume(self._current_volume + step)

    def decrease_volume(self, step: int = 1) -> int:
        """Decrease volume by step and return new volume."""
        return self.set_volume(self._current_volume - step)

    @property
    def current_volume(self) -> int:
        """Get the cached current volume value."""
        return self._current_volume
