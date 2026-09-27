"""
BMW-style macOS Gesture Control.
Main application entry point.
"""

import argparse
import signal
import sys
import time
import cv2

from hand_tracker import HandTracker
from gesture import GestureRecognizer, GestureConfig, ControlState
from volume import VolumeController
from media import MediaController


def draw_debug_overlay(frame, tracking_result, gesture_output, current_volume, fps):
    """Render a clean HUD overlay on the frame for debugging and calibration."""
    h, w, _ = frame.shape

    # HUD background panel (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (370, 200), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (10, 10), (370, 200), (60, 60, 75), 1)

    # 1. CONTROL: ACTIVE / INACTIVE
    is_active = (gesture_output.state == ControlState.CONTROL_ACTIVE)
    ctrl_str = "ACTIVE" if is_active else "INACTIVE"
    ctrl_color = (0, 255, 120) if is_active else (140, 140, 150)
    cv2.putText(
        frame,
        f"CONTROL: {ctrl_str}",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ctrl_color,
        2,
        cv2.LINE_AA,
    )

    # 2. GESTURE: NEUTRAL / ROTATING / NEXT / PREVIOUS / PLAY_PAUSE
    cv2.putText(
        frame,
        f"GESTURE: {gesture_output.active_gesture}",
        (25, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # 3. ROTATION: CLOCKWISE / ANTICLOCKWISE / NONE
    rot_str = gesture_output.rotation_direction
    cv2.putText(
        frame,
        f"ROTATION: {rot_str} ({gesture_output.accumulated_rotation:+.1f} deg)",
        (25, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 210),
        1,
        cv2.LINE_AA,
    )

    # 4. VOLUME: XX%
    cv2.putText(
        frame,
        f"VOLUME: {current_volume}%",
        (25, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 200, 50),
        2,
        cv2.LINE_AA,
    )
    # Mini volume bar
    cv2.rectangle(frame, (25, 138), (350, 146), (50, 50, 60), -1)
    bar_width = int(325 * (current_volume / 100.0))
    cv2.rectangle(frame, (25, 138), (25 + bar_width, 146), (255, 200, 50), -1)

    # Stability timer / confidence progress bar
    if gesture_output.stability_progress > 0.0 and gesture_output.stability_progress < 1.0:
        cv2.putText(
            frame,
            f"Stabilizing: {int(gesture_output.stability_progress * 100)}%",
            (25, 166),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        prog_w = int(120 * gesture_output.stability_progress)
        cv2.rectangle(frame, (160, 156), (160 + prog_w, 166), (0, 220, 255), -1)

    # FPS & Handedness info
    hand_info = tracking_result.handedness or "None"
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  |  Hand: {hand_info}",
        (25, 188),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (130, 130, 140),
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="BMW-style macOS Gesture Control via Webcam"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable visual debug window with hand landmarks and HUD",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera device index (default: 1 for built-in MacBook camera)",
    )
    args = parser.parse_args()

    # Terminal output banner
    print("Initializing camera...", flush=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            f"Error: Unable to open camera (device index {args.camera}). "
            "Please check that your webcam is connected and macOS Camera permissions are granted.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    # Attempt to set reasonable capture resolution for real-time 30 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm up camera auto-exposure (FaceTime HD camera auto-exposure needs a few frames)
    ret, test_frame = None, None
    for _ in range(5):
        ret, test_frame = cap.read()
        time.sleep(0.05)

    if not ret or test_frame is None:
        print(
            "Error: Camera opened but failed to capture a frame. Exiting.",
            file=sys.stderr,
            flush=True,
        )
        cap.release()
        sys.exit(1)

    print("Camera ready.", flush=True)

    # Initialize subsystems
    config = GestureConfig()
    hand_tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=config.DETECTION_CONFIDENCE,
        min_tracking_confidence=config.TRACKING_CONFIDENCE,
    )
    gesture_recognizer = GestureRecognizer(config=config)
    volume_ctrl = VolumeController()
    media_ctrl = MediaController()

    current_volume = volume_ctrl.current_volume

    # Print startup status
    print("BMW Gesture Control", flush=True)
    print("-------------------", flush=True)
    print("Camera: Ready", flush=True)
    print("Hand Tracking: Ready", flush=True)
    print("Waiting for open palm...", flush=True)

    # Graceful shutdown flag
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # FPS calculation helpers
    fps_time = time.time()
    frame_count = 0
    fps = 30.0

    try:
        while running:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Brief sleep to avoid busy looping on camera glitch
                time.sleep(0.01)
                continue

            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

            # Mirror the frame horizontally so interaction behaves like a mirror
            mirrored_frame = cv2.flip(frame, 1)

            # Process hand tracking
            tracking_result = hand_tracker.process_frame(mirrored_frame)

            # Process gesture state machine
            gesture_output = gesture_recognizer.process(tracking_result)

            # State transition events
            if gesture_output.state_changed:
                if gesture_output.state == ControlState.CONTROL_ACTIVE:
                    # Sync with latest macOS volume
                    current_volume = volume_ctrl.get_volume()
                    print("CONTROL ACTIVE", flush=True)
                    print(f"Volume: {current_volume}%", flush=True)
                elif gesture_output.state == ControlState.IDLE:
                    print("CONTROL INACTIVE", flush=True)
                    print("Waiting for open palm...", flush=True)

            # Discrete action triggers
            if gesture_output.action == "PLAY_PAUSE":
                media_ctrl.play_pause()
                print("PLAY / PAUSE", flush=True)
            elif gesture_output.action == "NEXT_TRACK":
                media_ctrl.next_track()
                print("NEXT TRACK", flush=True)
            elif gesture_output.action == "PREVIOUS_TRACK":
                media_ctrl.previous_track()
                print("PREVIOUS TRACK", flush=True)

            # Continuous volume rotation
            if gesture_output.volume_delta != 0:
                if gesture_output.volume_delta > 0:
                    current_volume = volume_ctrl.increase_volume(
                        abs(gesture_output.volume_delta)
                    )
                else:
                    current_volume = volume_ctrl.decrease_volume(
                        abs(gesture_output.volume_delta)
                    )
                print(f"Volume: {current_volume}%", flush=True)

            # Debug mode rendering
            if args.debug:
                hand_tracker.draw_landmarks(mirrored_frame, tracking_result)
                draw_debug_overlay(
                    mirrored_frame,
                    tracking_result,
                    gesture_output,
                    current_volume,
                    fps,
                )
                cv2.imshow("BMW Gesture Control - Debug", mirrored_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    running = False

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        cap.release()
        hand_tracker.release()
        if args.debug:
            cv2.destroyAllWindows()
        print("\nShutdown complete.", flush=True)


if __name__ == "__main__":
    main()
