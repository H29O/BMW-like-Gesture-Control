# BMW Gesture Control for macOS 🖐️🚗

A lightweight, local macOS gesture-control application written in Python that brings BMW-style in-air hand gestures to your MacBook. Built with OpenCV and MediaPipe, it lets you seamlessly adjust system volume, control media playback (play/pause), and skip tracks using your built-in webcam.

Zero cloud services, zero external APIs, zero paid dependencies—100% private and processed locally on device.

---

## 🌟 Features

- **Open Palm Activation**: Natural wake-up gesture. Stabilized for 300 ms to avoid accidental activation.
- **Index Finger Rotation (Volume Control)**: Spin your index finger clockwise to turn volume up (+1% per 15° arc) or anticlockwise to turn volume down (-1%), with smooth dead-zone filtering and continuous 360° wraparound.
- **Index + Middle Gesture (Play / Pause)**: Two extended fingers toggles playback on active macOS media players (Apple Music / Spotify).
- **Thumb Right (Next Track)**: Hitchhike thumb flick to the user's physical right skips to the next song.
- **Thumb Left (Previous Track)**: Hitchhike thumb flick to the user's physical left returns to the previous song.
- **Automatic Sleep / Idle Timeout**: When you lower your hand, control mode automatically deactivates after 500 ms and preserves the last volume level.
- **Real macOS Integration**: Direct system volume control and AppleScript media commands via `osascript`.
- **Visual Debug Mode**: HUD overlay displaying live landmarks, tracking state, angular rotation, and current volume.

---

## 📋 Technology Stack

- **Python 3**
- **OpenCV (`opencv-python`)**: High-performance webcam capture and frame rendering
- **MediaPipe**: Real-time 21-point hand landmark tracking
- **macOS AppleScript (`osascript`)**: Direct native integration for system volume and media players
- **Math & Subprocess**: Python standard library

---

## 📁 Project Architecture

```
gesture-control/
├── main.py            # Application entry point, event loop & terminal interface
├── hand_tracker.py    # MediaPipe hand tracking wrapper & landmark extraction
├── gesture.py         # Gesture state machine, mathematical algorithms & cooldowns
├── volume.py          # macOS system volume controller (0-100) via osascript
├── media.py           # macOS media playback controller (Music/Spotify)
├── requirements.txt   # Third-party Python dependencies
├── README.md          # Setup instructions, architecture & troubleshooting
└── .gitignore         # Ignores virtual environments, caches, and system files
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- macOS (tested on macOS Sonoma, Sequoia, and Apple Silicon / Intel Macs)
- Built-in MacBook FaceTime HD Camera or external USB webcam

### 2. Setup

```bash
# 1. Create a Python virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python3 main.py
```

### 3. Visual Calibration & Debug Mode

To calibrate your gestures or visualize landmark tracking in real-time, run with the `--debug` flag:

```bash
python3 main.py --debug
```

Press `q` or `ESC` in the debug window or `Ctrl + C` in the terminal to exit cleanly.

---

## 🖐️ Gesture Reference Guide

| Gesture | Pose Details | Action | Behavior & Timing |
| :--- | :--- | :--- | :--- |
| **Open Palm** | All 5 fingers extended facing camera | **Activate Control Mode** | Wakes system from `IDLE` to `CONTROL_ACTIVE` after 300 ms stability |
| **Index Rotation (Clockwise)** | Index extended, other 3 fingers curled | **Increase Volume** | +1% volume step per 15° arc (6° dead-zone, 200 ms discrete lock) |
| **Index Rotation (Anticlockwise)** | Index extended, other 3 fingers curled | **Decrease Volume** | -1% volume step per 15° arc (6° dead-zone, 200 ms discrete lock) |
| **Index + Middle Fingers** | Index & middle extended, ring & pinky folded | **Play / Pause** | Stable for 300 ms, triggers 1 action, 1000 ms cooldown, must release |
| **Thumb Pointing Right** | Fist with thumb extended to user's right | **Next Track** | Stable for 300 ms, triggers 1 action, 1000 ms cooldown, must release |
| **Thumb Pointing Left** | Fist with thumb extended to user's left | **Previous Track** | Stable for 300 ms, triggers 1 action, 1000 ms cooldown, must release |
| **Lower Hand / Leave Frame** | Hand absent from camera for > 500 ms | **Deactivate Control Mode** | Returns to `IDLE` state |

---

## ⚙️ Configuration & Tuning

All tunable thresholds can be customized directly in `gesture.py` under `GestureConfig`:

```python
@dataclass
class GestureConfig:
    OPEN_PALM_STABILITY_TIME: float = 0.30     # Palm hold time to activate from IDLE
    DISCRETE_STABILITY_TIME: float = 0.30      # 300 ms stability required for discrete gestures
    DISCRETE_COOLDOWN: float = 1.00            # 1000 ms cooldown after discrete action
    POST_DISCRETE_LOCKOUT: float = 0.20        # 200 ms lockout after discrete gesture ends
    HAND_LOST_TIMEOUT: float = 0.50            # Grace period before returning to IDLE
    ROTATION_DEAD_ZONE: float = 6.0            # Dead zone: 5-8 degrees per frame
    DEGREES_PER_VOLUME_STEP: float = 15.0      # 15 degrees intentional rotation = 1 volume step
    ROTATION_SMOOTHING_ALPHA: float = 0.35     # EMA smoothing for angular movement
```

---

## 🔍 Troubleshooting & macOS Permissions

### 1. macOS Camera Permission
When running the application for the first time from Terminal or iTerm:
- macOS may display a prompt asking: *"Terminal would like to access the camera"*.
- Click **Allow**.
- If denied or not prompted:
  1. Open **System Settings** -> **Privacy & Security** -> **Camera**.
  2. Ensure your terminal app (Terminal, iTerm2, or VS Code) is toggled **ON**.

### 2. AppleScript Automation Permission
When triggering media playback for the first time:
- macOS may prompt for permission for Terminal to control *Music* or *Spotify*.
- Click **OK** / **Allow**.
- If blocked, check **System Settings** -> **Privacy & Security** -> **Automation**.

### 3. Camera Selection
By default, the application uses **Camera index 1** (`cv2.VideoCapture(1)`), which corresponds to the built-in MacBook Air camera. On modern macOS systems, Camera index 0 is typically claimed by iPhone Continuity Camera. If you wish to switch or override the device index, pass the `--camera` argument:

```bash
python3 main.py --camera 1
```

---

## 🛑 Clean Shutdown

Press `Ctrl + C` in the terminal or `q` in the debug window. The program cleanly releases the camera hardware and OpenCV window resources.
