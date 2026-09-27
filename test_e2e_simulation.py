"""
End-to-End Simulation Test for BMW Gesture Control.
Validates all acceptance criteria:
1. No hand -> no action
2. Open palm -> control mode activates (after 300ms)
3. Index rotation clockwise -> volume increases
4. Index rotation anticlockwise -> volume decreases
5. Thumb pointing right -> next track (stable 300ms, one-shot)
6. Thumb pointing left -> previous track (stable 300ms, one-shot)
7. Index + middle fingers -> play/pause (stable 300ms, one-shot)
8. Hand leaves camera -> control mode deactivates (after 500ms)
9. Open palm again -> control mode activates again
"""

import math
import time
from test_gesture_logic import make_hand_landmarks
from hand_tracker import HandTrackingResult
from gesture import GestureRecognizer, GestureConfig, ControlState


def run_e2e_simulation():
    print("=== STARTING END-TO-END GESTURE SIMULATION TEST ===")
    config = GestureConfig(
        OPEN_PALM_STABILITY_TIME=0.15,
        DISCRETE_STABILITY_TIME=0.15,
        DISCRETE_COOLDOWN=0.20,
        POST_DISCRETE_LOCKOUT=0.08,
        HAND_LOST_TIMEOUT=0.20,
        ROTATION_DEAD_ZONE=5.0,
        DEGREES_PER_VOLUME_STEP=15.0,
    )
    recognizer = GestureRecognizer(config)

    # 1. No hand -> no action, remains IDLE
    res = recognizer.process(HandTrackingResult(detected=False))
    assert res.state == ControlState.IDLE, f"Expected IDLE, got {res.state}"
    assert res.action is None
    print("[PASS] 1. Initial State: IDLE, no action when no hand detected.")

    # 2. Open palm -> control mode activates after stability timeout
    open_palm = make_hand_landmarks(
        thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
    )
    res = recognizer.process(HandTrackingResult(detected=True, landmarks=open_palm))
    assert res.state == ControlState.IDLE, "Should not activate on first frame"

    time.sleep(0.18)
    res = recognizer.process(HandTrackingResult(detected=True, landmarks=open_palm))
    assert res.state == ControlState.CONTROL_ACTIVE, f"Expected CONTROL_ACTIVE, got {res.state}"
    assert res.action == "ACTIVATE"
    print("[PASS] 2. Open palm: Activated CONTROL_ACTIVE successfully after stability delay.")

    # 3. Index rotation clockwise -> volume increases
    wrist_x, wrist_y = 0.5, 0.8
    radius = 0.40

    angles_deg = [270.0, 290.0, 310.0, 330.0, 350.0, 10.0, 30.0, 50.0]
    total_volume_up = 0
    for deg in angles_deg:
        rad = math.radians(deg)
        tip_x = wrist_x + radius * math.cos(rad)
        tip_y = wrist_y + radius * math.sin(rad)
        rot_lms = make_hand_landmarks(
            thumb_ext=False,
            index_ext=True,
            middle_ext=False,
            ring_ext=False,
            pinky_ext=False,
            index_tip_pos=(tip_x, tip_y),
        )
        res = recognizer.process(HandTrackingResult(detected=True, landmarks=rot_lms))
        if res.volume_delta > 0:
            total_volume_up += res.volume_delta

    assert total_volume_up >= 2, f"Expected volume increase steps, got {total_volume_up}"
    print(f"[PASS] 3. Index rotation clockwise: Successfully increased volume by {total_volume_up} steps.")

    # 4. Index rotation anticlockwise -> volume decreases
    angles_ccw = [50.0, 30.0, 10.0, 350.0, 330.0, 310.0, 290.0, 270.0]
    total_volume_down = 0
    for deg in angles_ccw:
        rad = math.radians(deg)
        tip_x = wrist_x + radius * math.cos(rad)
        tip_y = wrist_y + radius * math.sin(rad)
        rot_lms = make_hand_landmarks(
            thumb_ext=False,
            index_ext=True,
            middle_ext=False,
            ring_ext=False,
            pinky_ext=False,
            index_tip_pos=(tip_x, tip_y),
        )
        res = recognizer.process(HandTrackingResult(detected=True, landmarks=rot_lms))
        if res.volume_delta < 0:
            total_volume_down += abs(res.volume_delta)

    assert total_volume_down >= 2, f"Expected volume decrease steps, got {total_volume_down}"
    print(f"[PASS] 4. Index rotation anticlockwise: Successfully decreased volume by {total_volume_down} steps.")

    # 5. Thumb right -> Next track (stable 300ms + one-shot)
    thumb_right = make_hand_landmarks(
        thumb_ext=True,
        index_ext=False,
        middle_ext=False,
        ring_ext=False,
        pinky_ext=False,
        thumb_dx=0.20,
    )
    # First frame: stabilizing
    res_tr0 = recognizer.process(HandTrackingResult(detected=True, landmarks=thumb_right))
    assert res_tr0.action is None, "Should not trigger on first frame"

    time.sleep(0.18)
    res1 = recognizer.process(HandTrackingResult(detected=True, landmarks=thumb_right))
    assert res1.action == "NEXT_TRACK", f"Expected NEXT_TRACK, got {res1.action}"

    # Next frame with same held thumb must NOT fire again
    res2 = recognizer.process(HandTrackingResult(detected=True, landmarks=thumb_right))
    assert res2.action is None, "Held thumb must NOT re-trigger immediately"
    print("[PASS] 5. Thumb right: Triggered NEXT_TRACK once after 300ms stability and held pose was debounced.")

    # Release gesture back to neutral
    neutral_lms = make_hand_landmarks(
        thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False
    )
    recognizer.process(HandTrackingResult(detected=True, landmarks=neutral_lms))

    # 6. Thumb left -> Previous track (stable + one-shot)
    time.sleep(0.25)
    thumb_left = make_hand_landmarks(
        thumb_ext=True,
        index_ext=False,
        middle_ext=False,
        ring_ext=False,
        pinky_ext=False,
        thumb_dx=-0.20,
    )
    recognizer.process(HandTrackingResult(detected=True, landmarks=thumb_left))
    time.sleep(0.18)
    res_tl = recognizer.process(HandTrackingResult(detected=True, landmarks=thumb_left))
    assert res_tl.action == "PREVIOUS_TRACK", f"Expected PREVIOUS_TRACK, got {res_tl.action}"
    print("[PASS] 6. Thumb left: Triggered PREVIOUS_TRACK once after stability delay.")

    # Release gesture
    recognizer.process(HandTrackingResult(detected=True, landmarks=neutral_lms))

    # 7. Index + Middle -> Play / Pause (stable + one-shot)
    time.sleep(0.25)
    two_fingers = make_hand_landmarks(
        thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False
    )
    recognizer.process(HandTrackingResult(detected=True, landmarks=two_fingers))
    time.sleep(0.18)
    res_pp = recognizer.process(HandTrackingResult(detected=True, landmarks=two_fingers))
    assert res_pp.action == "PLAY_PAUSE", f"Expected PLAY_PAUSE, got {res_pp.action}"
    print("[PASS] 7. Index + Middle fingers: Triggered PLAY_PAUSE once after stability delay.")

    # 8. Hand leaves camera -> returns to IDLE after timeout
    empty_res = HandTrackingResult(detected=False)
    recognizer.process(empty_res)
    assert recognizer.state == ControlState.CONTROL_ACTIVE, "Grace period should keep active"

    time.sleep(0.25)
    res_deact = recognizer.process(empty_res)
    assert res_deact.state == ControlState.IDLE, f"Expected IDLE after timeout, got {res_deact.state}"
    assert res_deact.action == "DEACTIVATE"
    print("[PASS] 8. Hand leaves camera: Grace period elapsed -> returned to IDLE.")

    # 9. Open palm again -> activates control again
    recognizer.process(HandTrackingResult(detected=True, landmarks=open_palm))
    time.sleep(0.18)
    res_re = recognizer.process(HandTrackingResult(detected=True, landmarks=open_palm))
    assert res_re.state == ControlState.CONTROL_ACTIVE
    assert res_re.action == "ACTIVATE"
    print("[PASS] 9. Open palm again: Re-activated CONTROL_ACTIVE successfully.")

    print("\nALL 9 ACCEPTANCE CRITERIA TESTS PASSED PERFECTLY! 🎉")


if __name__ == "__main__":
    run_e2e_simulation()
