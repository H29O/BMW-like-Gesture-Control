"""
Gesture Recognition and State Machine Module.
Implements mutually exclusive gesture classification:
- Open Palm (IDLE activation only)
- Play / Pause (index + middle, 300ms stability, 1000ms cooldown)
- Thumb Next (fist + thumb right, 300ms stability, 1000ms cooldown)
- Thumb Previous (fist + thumb left, 300ms stability, 1000ms cooldown)
- Index Rotation (index pointing, 5-8 deg dead zone, 15 deg per step, discrete lock)
- Neutral
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from hand_tracker import HandTrackingResult, LandmarkPoint


@dataclass
class GestureConfig:
    """Tunable gesture parameters."""
    OPEN_PALM_STABILITY_TIME: float = 0.30     # Palm hold time to activate from IDLE
    DISCRETE_STABILITY_TIME: float = 0.30      # 300 ms stability required for discrete gestures
    DISCRETE_COOLDOWN: float = 1.00            # 1000 ms cooldown after discrete action
    POST_DISCRETE_LOCKOUT: float = 0.20        # 200 ms lockout after discrete gesture ends
    HAND_LOST_TIMEOUT: float = 0.50            # Grace period before returning to IDLE
    ROTATION_DEAD_ZONE: float = 6.0            # Dead zone: 5-8 degrees per frame
    DEGREES_PER_VOLUME_STEP: float = 15.0      # 15 degrees intentional rotation = 1 volume step
    ROTATION_SMOOTHING_ALPHA: float = 0.35     # EMA smoothing for angular movement
    DETECTION_CONFIDENCE: float = 0.60         # MediaPipe detection confidence
    TRACKING_CONFIDENCE: float = 0.60          # MediaPipe tracking confidence


class ControlState:
    IDLE = "IDLE"
    CONTROL_ACTIVE = "CONTROL_ACTIVE"


class GestureType:
    NEUTRAL = "NEUTRAL"
    PLAY_PAUSE = "PLAY_PAUSE"
    THUMB_NEXT = "THUMB_NEXT"
    THUMB_PREVIOUS = "THUMB_PREVIOUS"
    ROTATION = "ROTATION"
    OPEN_PALM = "OPEN_PALM"


@dataclass
class GestureOutput:
    """Per-frame gesture engine output."""
    state: str                                  # IDLE or CONTROL_ACTIVE
    state_changed: bool                         # True if state changed on this frame
    active_gesture: str                         # NEUTRAL, ROTATING, NEXT, PREVIOUS, PLAY_PAUSE
    action: Optional[str]                       # "ACTIVATE", "DEACTIVATE", "PLAY_PAUSE", "NEXT_TRACK", "PREVIOUS_TRACK"
    volume_delta: int                           # Net volume adjustment (+1, -1, 0)
    rotation_direction: str                     # "CLOCKWISE", "ANTICLOCKWISE", "NONE"
    finger_angle: Optional[float]               # Current index angle in degrees
    accumulated_rotation: float                 # Rotation accumulated towards next step
    stability_progress: float                   # Progress (0.0 to 1.0) towards gesture stability


def distance_2d(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
    """Euclidean distance between two normalized 2D points."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


# ============================================================================
# EXPLICIT GESTURE CLASSIFIER HELPERS
# ============================================================================

def _finger_extension_ratios(landmarks: List[LandmarkPoint]):
    """
    Compute extension metrics for all 5 fingers based on normalized landmark geometry.
    Returns (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, palm_size, thumb_vec)
    """
    p0 = landmarks[0]   # Wrist
    p9 = landmarks[9]   # Middle MCP (palm center)
    palm_size = distance_2d(p0, p9)
    if palm_size < 1e-4:
        return False, False, False, False, False, 0.0, (0.0, 0.0)

    # 4 Main fingers: (tip, pip, mcp)
    finger_defs = [
        (8, 6, 5),    # Index
        (12, 10, 9),  # Middle
        (16, 14, 13), # Ring
        (20, 18, 17), # Pinky
    ]
    ext_flags = []
    for tip_idx, pip_idx, mcp_idx in finger_defs:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]

        d_tip_mcp = distance_2d(tip, mcp)
        d_pip_mcp = distance_2d(pip, mcp)
        d_tip_wrist = distance_2d(tip, p0)
        d_pip_wrist = distance_2d(pip, p0)

        # Extended when tip extends well beyond pip relative to both knuckle and wrist
        is_extended = (d_tip_mcp > d_pip_mcp * 1.35) and (d_tip_wrist > d_pip_wrist * 1.05)
        ext_flags.append(is_extended)

    index_ext, middle_ext, ring_ext, pinky_ext = ext_flags

    # Thumb metrics
    p4 = landmarks[4]   # Thumb tip
    p2 = landmarks[2]   # Thumb MCP
    p17 = landmarks[17] # Pinky MCP

    d_thumb_tip_pinky = distance_2d(p4, p17)
    d_thumb_mcp_pinky = distance_2d(p2, p17)
    thumb_ext = d_thumb_tip_pinky > d_thumb_mcp_pinky * 1.15

    dx = p4.x - p2.x
    dy = p4.y - p2.y

    return thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, palm_size, (dx, dy)


def is_open_palm(landmarks: List[LandmarkPoint]) -> bool:
    """Open Palm: all 5 fingers extended."""
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, _, _ = _finger_extension_ratios(landmarks)
    return thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext


def is_play_pause_gesture(landmarks: List[LandmarkPoint]) -> bool:
    """
    Play/Pause Gesture:
    - Index finger extended
    - Middle finger extended
    - Ring finger folded
    - Pinky finger folded
    - Thumb folded or neutral (not open palm)
    """
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, _, _ = _finger_extension_ratios(landmarks)
    return index_ext and middle_ext and (not ring_ext) and (not pinky_ext)


def is_thumb_next_gesture(landmarks: List[LandmarkPoint]) -> bool:
    """
    Thumb Next Gesture:
    - Index, middle, ring, and pinky are all folded into a fist
    - Thumb is clearly extended horizontally
    - Thumb points toward user's physical right (dx > 0 in mirrored webcam)
    """
    _, index_ext, middle_ext, ring_ext, pinky_ext, palm_size, (dx, dy) = _finger_extension_ratios(landmarks)
    all_four_folded = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext)
    is_horizontal = abs(dx) > abs(dy) * 1.15 and abs(dx) > (0.35 * palm_size)
    return all_four_folded and is_horizontal and (dx > 0)


def is_thumb_previous_gesture(landmarks: List[LandmarkPoint]) -> bool:
    """
    Thumb Previous Gesture:
    - Index, middle, ring, and pinky are all folded into a fist
    - Thumb is clearly extended horizontally
    - Thumb points toward user's physical left (dx < 0 in mirrored webcam)
    """
    _, index_ext, middle_ext, ring_ext, pinky_ext, palm_size, (dx, dy) = _finger_extension_ratios(landmarks)
    all_four_folded = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext)
    is_horizontal = abs(dx) > abs(dy) * 1.15 and abs(dx) > (0.35 * palm_size)
    return all_four_folded and is_horizontal and (dx < 0)


def is_index_rotation_pose(landmarks: List[LandmarkPoint]) -> bool:
    """
    Index Rotation Pose:
    - Index finger clearly extended
    - Middle, ring, and pinky fingers folded into hand
    - Thumb not horizontally extended
    """
    _, index_ext, middle_ext, ring_ext, pinky_ext, palm_size, (dx, dy) = _finger_extension_ratios(landmarks)
    thumb_horizontal = abs(dx) > abs(dy) * 1.15 and abs(dx) > (0.35 * palm_size)
    return index_ext and (not middle_ext) and (not ring_ext) and (not pinky_ext) and (not thumb_horizontal)


def classify_raw_gesture(landmarks: List[LandmarkPoint]) -> str:
    """
    Mutually exclusive classification in strict priority order:
    1. PLAY_PAUSE
    2. THUMB_NEXT
    3. THUMB_PREVIOUS
    4. ROTATION
    5. OPEN_PALM / NEUTRAL
    """
    if is_play_pause_gesture(landmarks):
        return GestureType.PLAY_PAUSE
    if is_thumb_next_gesture(landmarks):
        return GestureType.THUMB_NEXT
    if is_thumb_previous_gesture(landmarks):
        return GestureType.THUMB_PREVIOUS
    if is_index_rotation_pose(landmarks):
        return GestureType.ROTATION
    if is_open_palm(landmarks):
        return GestureType.OPEN_PALM
    return GestureType.NEUTRAL


# ============================================================================
# STATE MACHINE & GESTURE ENGINE
# ============================================================================

class GestureRecognizer:
    """
    State machine maintaining mutual exclusion, discrete stability timers,
    action cooldowns, and intentional index rotation.
    """

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        self.state = ControlState.IDLE

        # Timers
        self._palm_candidate_start: Optional[float] = None
        self._last_hand_seen_time: Optional[float] = None

        # Discrete gesture stability & debounce tracking
        self._discrete_candidate: Optional[str] = None
        self._discrete_candidate_start: Optional[float] = None
        self._discrete_action_fired_for: Optional[str] = None
        self._last_discrete_action_time: float = 0.0

        # Discrete lockout: pauses rotation during and for 200ms after discrete gestures
        self._discrete_in_progress: bool = False
        self._discrete_ended_time: Optional[float] = None

        # Rotation tracking
        self._prev_angle: Optional[float] = None
        self._smoothed_delta: float = 0.0
        self._rotation_accumulator: float = 0.0

    def reset_rotation_baseline(self):
        """Reset rotation accumulator and baseline angle."""
        self._prev_angle = None
        self._smoothed_delta = 0.0
        self._rotation_accumulator = 0.0

    def _process_rotation(
        self, landmarks: List[LandmarkPoint]
    ) -> Tuple[int, str, Optional[float]]:
        """
        Process index rotation with dead zone, EMA smoothing, and 15 deg accumulation.
        Returns (volume_delta, direction_str, raw_angle).
        """
        p0 = landmarks[0]  # Wrist
        p8 = landmarks[8]  # Index tip

        dx = p8.x - p0.x
        dy = p8.y - p0.y

        # Screen coordinates: dy down, dx right -> clockwise angle increases
        raw_angle = math.degrees(math.atan2(dy, dx)) % 360.0

        if self._prev_angle is None:
            self._prev_angle = raw_angle
            return 0, "NONE", raw_angle

        # Shortest angle delta handling 359° <-> 0° wraparound
        delta = (raw_angle - self._prev_angle + 180.0) % 360.0 - 180.0
        self._prev_angle = raw_angle

        # Dead zone filtering (5-8 degrees, default 6.0)
        if abs(delta) < self.config.ROTATION_DEAD_ZONE:
            delta = 0.0
            # Decay smoothed delta towards zero when stationary
            self._smoothed_delta *= 0.6
            if abs(self._smoothed_delta) < 0.1:
                self._smoothed_delta = 0.0
        else:
            alpha = self.config.ROTATION_SMOOTHING_ALPHA
            self._smoothed_delta = alpha * delta + (1.0 - alpha) * self._smoothed_delta

        # Determine rotation direction
        direction = "NONE"
        if self._smoothed_delta > 0.8:
            direction = "CLOCKWISE"
        elif self._smoothed_delta < -0.8:
            direction = "ANTICLOCKWISE"

        # Accumulate intentional movement
        self._rotation_accumulator += self._smoothed_delta
        step_threshold = self.config.DEGREES_PER_VOLUME_STEP
        volume_delta = 0

        if self._rotation_accumulator >= step_threshold:
            steps = int(self._rotation_accumulator // step_threshold)
            self._rotation_accumulator -= steps * step_threshold
            volume_delta = steps
        elif self._rotation_accumulator <= -step_threshold:
            steps = int(abs(self._rotation_accumulator) // step_threshold)
            self._rotation_accumulator += steps * step_threshold
            volume_delta = -steps

        return volume_delta, direction, raw_angle

    def process(self, tracking_result: HandTrackingResult) -> GestureOutput:
        """Process tracking result through the mutually exclusive gesture engine."""
        now = time.time()
        state_changed = False
        action: Optional[str] = None
        volume_delta = 0
        rotation_direction = "NONE"
        finger_angle: Optional[float] = None
        stability_progress = 0.0
        active_gesture = GestureType.NEUTRAL

        # --------------------------------------------------------------------
        # 1. TRACKING LOSS HANDLING
        # --------------------------------------------------------------------
        if not tracking_result.detected or not tracking_result.landmarks:
            if self.state == ControlState.CONTROL_ACTIVE:
                if self._last_hand_seen_time is not None:
                    if (now - self._last_hand_seen_time) >= self.config.HAND_LOST_TIMEOUT:
                        self.state = ControlState.IDLE
                        state_changed = True
                        action = "DEACTIVATE"
                        self.reset_rotation_baseline()
                        self._discrete_candidate = None
                        self._discrete_in_progress = False
            self._palm_candidate_start = None
            return GestureOutput(
                state=self.state,
                state_changed=state_changed,
                active_gesture="NEUTRAL",
                action=action,
                volume_delta=0,
                rotation_direction="NONE",
                finger_angle=None,
                accumulated_rotation=0.0,
                stability_progress=0.0,
            )

        self._last_hand_seen_time = now
        landmarks = tracking_result.landmarks
        raw_pose = classify_raw_gesture(landmarks)

        # --------------------------------------------------------------------
        # 2. IDLE STATE: WATCH FOR OPEN PALM ACTIVATION
        # --------------------------------------------------------------------
        if self.state == ControlState.IDLE:
            if raw_pose == GestureType.OPEN_PALM:
                if self._palm_candidate_start is None:
                    self._palm_candidate_start = now
                stability_duration = now - self._palm_candidate_start
                stability_progress = min(1.0, stability_duration / self.config.OPEN_PALM_STABILITY_TIME)
                if stability_duration >= self.config.OPEN_PALM_STABILITY_TIME:
                    self.state = ControlState.CONTROL_ACTIVE
                    state_changed = True
                    action = "ACTIVATE"
                    self._palm_candidate_start = None
                    self.reset_rotation_baseline()
            else:
                self._palm_candidate_start = None
                stability_progress = 0.0

            return GestureOutput(
                state=self.state,
                state_changed=state_changed,
                active_gesture="NEUTRAL" if not state_changed else "NEUTRAL",
                action=action,
                volume_delta=0,
                rotation_direction="NONE",
                finger_angle=None,
                accumulated_rotation=0.0,
                stability_progress=stability_progress,
            )

        # --------------------------------------------------------------------
        # 3. CONTROL_ACTIVE STATE: MUTUALLY EXCLUSIVE GESTURE HANDLING
        # --------------------------------------------------------------------
        is_discrete_pose = raw_pose in (
            GestureType.PLAY_PAUSE,
            GestureType.THUMB_NEXT,
            GestureType.THUMB_PREVIOUS,
        )

        if is_discrete_pose:
            # Lock rotation immediately upon discrete gesture pose
            self._rotation_accumulator = 0.0
            self._prev_angle = None
            self._smoothed_delta = 0.0
            self._discrete_in_progress = True
            self._discrete_ended_time = None

            # Track stability of the candidate discrete pose
            if self._discrete_candidate != raw_pose:
                self._discrete_candidate = raw_pose
                self._discrete_candidate_start = now

            stability_elapsed = now - self._discrete_candidate_start
            stability_progress = min(1.0, stability_elapsed / self.config.DISCRETE_STABILITY_TIME)

            # Check if candidate has reached required stability (300 ms)
            if stability_elapsed >= self.config.DISCRETE_STABILITY_TIME:
                # Map active gesture name for HUD
                if raw_pose == GestureType.PLAY_PAUSE:
                    active_gesture = "PLAY_PAUSE"
                elif raw_pose == GestureType.THUMB_NEXT:
                    active_gesture = "NEXT"
                elif raw_pose == GestureType.THUMB_PREVIOUS:
                    active_gesture = "PREVIOUS"

                # Check action trigger (one-shot per gesture hold + cooldown)
                cooldown_elapsed = now - self._last_discrete_action_time
                can_trigger = (
                    (cooldown_elapsed >= self.config.DISCRETE_COOLDOWN)
                    and (self._discrete_action_fired_for != raw_pose)
                )

                if can_trigger:
                    self._discrete_action_fired_for = raw_pose
                    self._last_discrete_action_time = now
                    if raw_pose == GestureType.PLAY_PAUSE:
                        action = "PLAY_PAUSE"
                    elif raw_pose == GestureType.THUMB_NEXT:
                        action = "NEXT_TRACK"
                    elif raw_pose == GestureType.THUMB_PREVIOUS:
                        action = "PREVIOUS_TRACK"
            else:
                active_gesture = "NEUTRAL"

        else:
            # Discrete gesture pose has disappeared/released
            if self._discrete_in_progress:
                self._discrete_in_progress = False
                self._discrete_ended_time = now   # Start 200 ms lockout
                self._discrete_action_fired_for = None  # Ready for next engagement

            self._discrete_candidate = None
            self._discrete_candidate_start = None

            # Check 200 ms post-discrete lockout
            is_locked_out = (
                self._discrete_ended_time is not None
                and (now - self._discrete_ended_time) < self.config.POST_DISCRETE_LOCKOUT
            )

            if is_locked_out:
                active_gesture = "NEUTRAL"
                self.reset_rotation_baseline()
            elif raw_pose == GestureType.ROTATION:
                active_gesture = "ROTATING"
                volume_delta, rotation_direction, finger_angle = self._process_rotation(landmarks)
            else:
                # Open palm or neutral while CONTROL_ACTIVE
                active_gesture = "NEUTRAL"
                self.reset_rotation_baseline()

        return GestureOutput(
            state=self.state,
            state_changed=state_changed,
            active_gesture=active_gesture,
            action=action,
            volume_delta=volume_delta,
            rotation_direction=rotation_direction,
            finger_angle=finger_angle,
            accumulated_rotation=self._rotation_accumulator,
            stability_progress=stability_progress,
        )
