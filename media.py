"""
macOS Media Controller using AppleScript (osascript).
Controls playback, next track, and previous track for active macOS media players (Music / Spotify).
"""

import subprocess
import logging

logger = logging.getLogger(__name__)


def is_app_running(app_name: str) -> bool:
    """Check if an application process is currently running on macOS."""
    try:
        res = subprocess.run(
            ["pgrep", "-ix", app_name],
            capture_output=True,
            timeout=0.5
        )
        return res.returncode == 0
    except Exception as e:
        logger.debug("Error checking if %s is running: %s", app_name, e)
        return False


class MediaController:
    """Controls macOS media playback through native AppleScript commands."""

    @staticmethod
    def _run_script(script: str) -> bool:
        """Execute an AppleScript snippet via osascript with error handling."""
        try:
            cmd = ["osascript", "-e", script]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1.5
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("AppleScript failed (rc=%s): %s", e.returncode, e.stderr.strip())
            return False
        except Exception as e:
            logger.warning("Error running AppleScript: %s", e)
            return False

    def _get_target_player(self) -> str:
        """Determine which player to command based on running processes."""
        if is_app_running("Spotify"):
            return "Spotify"
        return "Music"

    def play_pause(self) -> bool:
        """Toggle play/pause on the active media player."""
        player = self._get_target_player()
        script = f'tell application "{player}" to playpause'
        success = self._run_script(script)
        if not success and player == "Spotify":
            # Fallback to Music
            success = self._run_script('tell application "Music" to playpause')
        return success

    def next_track(self) -> bool:
        """Skip to next track on the active media player."""
        player = self._get_target_player()
        script = f'tell application "{player}" to next track'
        success = self._run_script(script)
        if not success and player == "Spotify":
            success = self._run_script('tell application "Music" to next track')
        return success

    def previous_track(self) -> bool:
        """Go to previous track on the active media player."""
        player = self._get_target_player()
        script = f'tell application "{player}" to previous track'
        success = self._run_script(script)
        if not success and player == "Spotify":
            success = self._run_script('tell application "Music" to previous track')
        return success
