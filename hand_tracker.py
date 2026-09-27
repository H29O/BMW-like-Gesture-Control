"""
MediaPipe Hand Tracking Module.
Uses MediaPipe HandLandmarker Tasks API with local model processing.
Encapsulates real-time hand detection, landmark extraction, coordinate normalization, and visual drawing.
"""

from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple
import urllib.request
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

# 21 standard hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20), # Pinky & palm base
]


class LandmarkPoint(NamedTuple):
    x: float
    y: float
    z: float


class HandTrackingResult:
    """Encapsulates hand detection result for a single frame."""

    def __init__(
        self,
        detected: bool = False,
        landmarks: Optional[List[LandmarkPoint]] = None,
        handedness: Optional[str] = None,
    ):
        self.detected = detected
        self.landmarks = landmarks or []
        self.handedness = handedness

    def get_landmark(self, index: int) -> Optional[LandmarkPoint]:
        if self.detected and 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None


def ensure_model_file(target_path: Path) -> Path:
    """Ensure the local hand landmarker model file exists."""
    if not target_path.exists():
        print(f"Downloading MediaPipe hand model to {target_path}...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, target_path)
        print("Model download complete.", flush=True)
    return target_path


class HandTracker:
    """MediaPipe Hands wrapper for single-hand real-time tracking."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_path: Optional[str] = None,
    ):
        if model_path is None:
            base_dir = Path(__file__).parent.resolve()
            model_file = ensure_model_file(base_dir / MODEL_FILENAME)
            model_path = str(model_file)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process_frame(self, frame_bgr) -> HandTrackingResult:
        """
        Process a BGR frame (from OpenCV) and extract 21 hand landmarks.
        Assumes frame is already oriented as desired (e.g. horizontally mirrored).
        """
        if frame_bgr is None:
            return HandTrackingResult(detected=False)

        # Convert OpenCV BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return HandTrackingResult(detected=False)

        # Primary hand
        primary_landmarks = result.hand_landmarks[0]
        handedness_label = None
        if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
            handedness_label = result.handedness[0][0].category_name

        normalized_landmarks = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z)
            for lm in primary_landmarks
        ]

        return HandTrackingResult(
            detected=True,
            landmarks=normalized_landmarks,
            handedness=handedness_label,
        )

    def draw_landmarks(self, frame_bgr, tracking_result: HandTrackingResult):
        """Draw hand landmarks and skeleton connections onto the provided frame."""
        if not tracking_result.detected or not tracking_result.landmarks:
            return

        h, w, _ = frame_bgr.shape
        landmarks = tracking_result.landmarks

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(frame_bgr, pt1, pt2, (0, 220, 255), 2, cv2.LINE_AA)

        # Draw landmark points
        for i, lm in enumerate(landmarks):
            center = (int(lm.x * w), int(lm.y * h))
            # Highlight index tip (8) and thumb tip (4) specially
            if i == 8:
                cv2.circle(frame_bgr, center, 7, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, center, 9, (255, 255, 255), 1, cv2.LINE_AA)
            elif i == 4:
                cv2.circle(frame_bgr, center, 7, (255, 100, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, center, 4, (0, 160, 255), -1, cv2.LINE_AA)

    def release(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
