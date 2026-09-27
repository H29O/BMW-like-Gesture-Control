"""
Unit tests for gesture recognition, geometry, state machine, and volume calculation.
"""

import math
import time
import unittest
from hand_tracker import LandmarkPoint, HandTrackingResult
from gesture import (
    GestureConfig,
    GestureRecognizer,
    ControlState,
    GestureType,
    distance_2d,
    is_open_palm,
    is_play_pause_gesture,
    is_thumb_next_gesture,
    is_thumb_previous_gesture,
    is_index_rotation_pose,
    classify_raw_gesture,
)
from volume import VolumeController


def make_landmark(x: float, y: float, z: float = 0.0) -> LandmarkPoint:
    return LandmarkPoint(x=x, y=y, z=z)


def make_hand_landmarks(
    thumb_ext=True,
    index_ext=True,
    middle_ext=True,
    ring_ext=True,
    pinky_ext=True,
    thumb_dx=0.0,
    thumb_dy=0.0,
    index_tip_pos=(0.5, 0.2),
):
    """Generate a synthetic 21-point hand landmark set with desired finger states."""
    p = [make_landmark(0.5, 0.8)] * 21

    # Wrist: 0
    p[0] = make_landmark(0.5, 0.8)

    # Middle MCP: 9 (defines palm reference)
    p[9] = make_landmark(0.5, 0.5)

    # Index MCP (5), PIP (6), DIP (7), TIP (8)
    p[5] = make_landmark(0.42, 0.52)
    if index_ext:
        p[8] = make_landmark(index_tip_pos[0], index_tip_pos[1])
        p[6] = make_landmark(
            p[5].x + 0.45 * (p[8].x - p[5].x),
            p[5].y + 0.45 * (p[8].y - p[5].y),
        )
        p[7] = make_landmark(
            p[5].x + 0.75 * (p[8].x - p[5].x),
            p[5].y + 0.75 * (p[8].y - p[5].y),
        )
    else:
        p[6] = make_landmark(0.40, 0.40)
        p[7] = make_landmark(0.39, 0.30)
        p[8] = make_landmark(0.43, 0.53)

    # Middle: 9, 10, 11, 12
    if middle_ext:
        p[10] = make_landmark(0.50, 0.38)
        p[11] = make_landmark(0.50, 0.28)
        p[12] = make_landmark(0.50, 0.18)
    else:
        p[10] = make_landmark(0.50, 0.45)
        p[11] = make_landmark(0.50, 0.48)
        p[12] = make_landmark(0.50, 0.52)

    # Ring: 13, 14, 15, 16
    p[13] = make_landmark(0.58, 0.52)
    if ring_ext:
        p[14] = make_landmark(0.60, 0.40)
        p[15] = make_landmark(0.61, 0.30)
        p[16] = make_landmark(0.62, 0.20)
    else:
        p[14] = make_landmark(0.58, 0.45)
        p[15] = make_landmark(0.58, 0.48)
        p[16] = make_landmark(0.58, 0.53)

    # Pinky: 17, 18, 19, 20
    p[17] = make_landmark(0.66, 0.55)
    if pinky_ext:
        p[18] = make_landmark(0.68, 0.45)
        p[19] = make_landmark(0.69, 0.38)
        p[20] = make_landmark(0.70, 0.28)
    else:
        p[18] = make_landmark(0.66, 0.48)
        p[19] = make_landmark(0.66, 0.52)
        p[20] = make_landmark(0.66, 0.56)

    # Thumb: 1, 2, 3, 4
    p[1] = make_landmark(0.44, 0.72)
    p[2] = make_landmark(0.38, 0.65)
    p[3] = make_landmark(0.32, 0.60)
    if thumb_ext:
        if thumb_dx != 0.0 or thumb_dy != 0.0:
            p[4] = make_landmark(0.38 + thumb_dx, 0.65 + thumb_dy)
        else:
            p[4] = make_landmark(0.24, 0.54)
    else:
        p[4] = make_landmark(0.44, 0.60)

    return p


class TestGestureRecognition(unittest.TestCase):

    def setUp(self):
        self.config = GestureConfig(
            OPEN_PALM_STABILITY_TIME=0.10,
            DISCRETE_STABILITY_TIME=0.10,
            DISCRETE_COOLDOWN=0.20,
            POST_DISCRETE_LOCKOUT=0.08,
            HAND_LOST_TIMEOUT=0.15,
            ROTATION_DEAD_ZONE=5.0,
            DEGREES_PER_VOLUME_STEP=15.0,
        )
        self.recognizer = GestureRecognizer(self.config)

    def test_open_palm_detection(self):
        landmarks = make_hand_landmarks(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        self.assertTrue(is_open_palm(landmarks))
        self.assertEqual(classify_raw_gesture(landmarks), GestureType.OPEN_PALM)

    def test_play_pause_detection(self):
        # Index & Middle extended, Ring & Pinky folded, Thumb folded
        landmarks = make_hand_landmarks(
            thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False
        )
        self.assertTrue(is_play_pause_gesture(landmarks))
        self.assertFalse(is_index_rotation_pose(landmarks))  # Mutually exclusive!
        self.assertEqual(classify_raw_gesture(landmarks), GestureType.PLAY_PAUSE)

    def test_thumb_next_detection(self):
        # Thumb extended right (+dx), 4 fingers folded
        landmarks = make_hand_landmarks(
            thumb_ext=True,
            index_ext=False,
            middle_ext=False,
            ring_ext=False,
            pinky_ext=False,
            thumb_dx=0.20,
            thumb_dy=0.01,
        )
        self.assertTrue(is_thumb_next_gesture(landmarks))
        self.assertFalse(is_thumb_previous_gesture(landmarks))
        self.assertEqual(classify_raw_gesture(landmarks), GestureType.THUMB_NEXT)

    def test_thumb_previous_detection(self):
        # Thumb extended left (-dx), 4 fingers folded
        landmarks = make_hand_landmarks(
            thumb_ext=True,
            index_ext=False,
            middle_ext=False,
            ring_ext=False,
            pinky_ext=False,
            thumb_dx=-0.20,
            thumb_dy=0.01,
        )
        self.assertTrue(is_thumb_previous_gesture(landmarks))
        self.assertFalse(is_thumb_next_gesture(landmarks))
        self.assertEqual(classify_raw_gesture(landmarks), GestureType.THUMB_PREVIOUS)

    def test_rotation_pose_detection(self):
        # Index extended, middle folded, ring folded, pinky folded
        landmarks = make_hand_landmarks(
            thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False
        )
        self.assertTrue(is_index_rotation_pose(landmarks))
        self.assertFalse(is_play_pause_gesture(landmarks))  # Mutually exclusive!
        self.assertEqual(classify_raw_gesture(landmarks), GestureType.ROTATION)

    def test_state_machine_activation_and_timeout(self):
        # Starts in IDLE
        self.assertEqual(self.recognizer.state, ControlState.IDLE)

        palm_lms = make_hand_landmarks(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        # First frame: candidate only
        result1 = self.recognizer.process(HandTrackingResult(detected=True, landmarks=palm_lms))
        self.assertEqual(result1.state, ControlState.IDLE)
        self.assertFalse(result1.state_changed)

        # After stability timeout: activates
        time.sleep(0.12)
        result2 = self.recognizer.process(HandTrackingResult(detected=True, landmarks=palm_lms))
        self.assertEqual(result2.state, ControlState.CONTROL_ACTIVE)
        self.assertTrue(result2.state_changed)
        self.assertEqual(result2.action, "ACTIVATE")

        # Hand lost grace period
        empty_res = HandTrackingResult(detected=False)
        result3 = self.recognizer.process(empty_res)
        self.assertEqual(result3.state, ControlState.CONTROL_ACTIVE)

        time.sleep(0.18)
        result4 = self.recognizer.process(empty_res)
        self.assertEqual(result4.state, ControlState.IDLE)
        self.assertTrue(result4.state_changed)
        self.assertEqual(result4.action, "DEACTIVATE")

    def test_discrete_gesture_stability_and_debounce(self):
        # Activate control
        palm_lms = make_hand_landmarks(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        self.recognizer.process(HandTrackingResult(detected=True, landmarks=palm_lms))
        time.sleep(0.12)
        self.recognizer.process(HandTrackingResult(detected=True, landmarks=palm_lms))
        self.assertEqual(self.recognizer.state, ControlState.CONTROL_ACTIVE)

        # Present Play/Pause pose
        two_fingers = make_hand_landmarks(
            thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False
        )
        # Frame 1: candidate stabilizing, no action yet
        r1 = self.recognizer.process(HandTrackingResult(detected=True, landmarks=two_fingers))
        self.assertIsNone(r1.action)

        # Wait 300ms stability -> fires action
        time.sleep(0.12)
        r2 = self.recognizer.process(HandTrackingResult(detected=True, landmarks=two_fingers))
        self.assertEqual(r2.action, "PLAY_PAUSE")
        self.assertEqual(r2.active_gesture, "PLAY_PAUSE")

        # Held pose on next frame -> must NOT fire again!
        r3 = self.recognizer.process(HandTrackingResult(detected=True, landmarks=two_fingers))
        self.assertIsNone(r3.action)
        self.assertEqual(r3.active_gesture, "PLAY_PAUSE")

    def test_discrete_lockout_prevents_rotation(self):
        # Activate control
        palm_lms = make_hand_landmarks(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        self.recognizer.process(HandTrackingResult(detected=True, landmarks=palm_lms))
        time.sleep(0.12)
        self.recognizer.process(HandTrackingResult(detected=True, landmarks=palm_lms))

        # While Play/Pause pose is active, rotation accumulator must remain 0
        two_fingers = make_hand_landmarks(
            thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False
        )
        time.sleep(0.12)
        r = self.recognizer.process(HandTrackingResult(detected=True, landmarks=two_fingers))
        self.assertEqual(r.volume_delta, 0)
        self.assertEqual(self.recognizer._rotation_accumulator, 0.0)

    def test_wraparound_rotation_math(self):
        cases = [
            (359.0, 1.0, 2.0),
            (1.0, 359.0, -2.0),
            (10.0, 30.0, 20.0),
            (30.0, 10.0, -20.0),
            (180.0, 180.0, 0.0),
        ]
        for prev, curr, expected in cases:
            raw_delta = curr - prev
            wrapped = (raw_delta + 180.0) % 360.0 - 180.0
            self.assertAlmostEqual(wrapped, expected, places=1)


class TestVolumeClamping(unittest.TestCase):
    def test_volume_clamp(self):
        vc = VolumeController()
        self.assertEqual(max(0, min(100, -10)), 0)
        self.assertEqual(max(0, min(100, 110)), 100)
        self.assertEqual(max(0, min(100, 50)), 50)


if __name__ == "__main__":
    unittest.main()
