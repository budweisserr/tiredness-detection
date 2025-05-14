# Driver Drowsiness Detection System Documentation

## Overview

This project is a GUI-based application for detecting driver drowsiness using facial landmarks. It utilizes the PyQt6 framework for the user interface, OpenCV and Dlib for video and image processing, and VLC for audio alerts.

The application tracks eye aspect ratio (EAR), yawning, and head position to estimate drowsiness levels. Alerts are issued when signs of fatigue are detected.

---

## Project Structure

```
root/
├── ui.py             # GUI application using PyQt5/6
├── video_processor.py # Background thread for video processing and detection
├── config.py          # Configuration constants and resource paths
```

---

## `config.py`

Defines a dictionary `APP_CONFIG` containing:

* `predictor_path`: Path to the Dlib shape predictor file (68 landmarks).
* `left_eye_path`, `right_eye_path`: Output paths for saving eye images.
* `alert_sound`, `focus_sound`, `break_sound`: Paths to audio files for different alert types.

---

## `ui.py` — `DrowsinessDetectionUI` Class

### Purpose

The main PyQt6 GUI window that:

* Displays the camera feed.
* Shows status information (EAR, eye/yawn status, alertness bar).
* Contains Start/Stop buttons.
* Updates based on signals from the `VideoProcessor` thread.

### Key Components

* `initUI()`: Sets up the GUI layout and styles.
* `start_video()`: Creates and starts `VideoProcessor`, connects its signals.
* `stop_video()`: Stops the thread and resets UI components.
* `update_frame(frame)`: Converts OpenCV frame to QImage and displays it.
* `update_status(status)`: Updates alertness bar, label styles, and textual status.

### Signals Handled

* `update_frame`: New frame from `VideoProcessor` to be shown.
* `update_status`: Dict containing EAR, yawning flag, alert level, and message.

---

## `video_processor.py` — `VideoProcessor` Class

### Purpose

A `QThread` that runs continuously in the background:

* Captures video from webcam.
* Detects faces and facial landmarks.
* Calculates EAR and yawn ratios.
* Determines drowsiness level based on thresholds.
* Emits GUI update signals.

### Initialization

* Takes `predictor_path` and `sound_paths` (from config) as input.
* Initializes `dlib` detectors and `vlc` audio players.
* Prepares detection thresholds and internal state variables.

### Detection Logic in `run()`

1. Capture frame from webcam.
2. Convert to grayscale.
3. Detect face and predict landmarks.
4. Extract eye and mouth landmarks.
5. Compute:

   * **EAR**: Eye Aspect Ratio to detect eye closure.
   * **Yawn ratio**: Mouth openness.
   * **Head pose**: Estimate face direction (leaning forward).
6. Determine `alert_level`:

   * 0: Normal
   * 1: Eyes Closed
   * 2: Bad Posture
   * 3: Eyes closed after yawning
7. Emit `update_frame` and `update_status` signals.
8. Play audio alerts based on severity.
9. Save eye images for logging.

### Other Functions

* `ear(eye)`: Calculates the eye aspect ratio.
* `yawn(mouth)`: Calculates a yawn ratio.
* `euclideanDist(a, b)`: Helper for distance computation.
* `getFaceDirection(shape, size)`: Uses `cv2.solvePnP` to get head pose.
* `writeEyes(left_eye, right_eye, img)`: Saves eye images to disk.
* `stop()`: Gracefully stops the video thread.

---

## Detection Criteria Summary

| Condition                   | Threshold / Action                | Alert Level | Sound     |
| --------------------------- | --------------------------------- | ----------- | --------- |
| Eyes Closed                 | EAR < `close_thresh` for N frames | 1           | focus.mp3 |
| Posture Forward (Head Down) | Y coordinate from solvePnP        | 2           | focus.mp3 |
| Yawning + Eyes Closed       | Yawn ratio > 0.6 + Eyes closed    | 3           | focus.mp3 |
| Frequent Fatigue            | 3 alerts triggered                | -           | break.mp3 |

---

## Technologies Used

* **PyQt6**: GUI framework.
* **OpenCV**: Image capture and processing.
* **Dlib**: Face detection and landmark extraction.
* **Imutils**: Landmark convenience utilities.
* **VLC Python bindings**: Audio alerts.

---

## How to Run

1. Install dependencies:

   ```bash
   pip install pyqt6 opencv-python dlib imutils python-vlc
   ```
2. Place required files in correct paths (matching `config.py`). Will improve later :)
3. Run the app:

   ```bash
   python main.py
   ```

---

## Potential Improvements

* Add logging to a file or database.
* Implement eyelid and blink frequency detection.
* Support recording of video footage when drowsiness is detected.
* Improve UI responsiveness and error handling.
* Use a pre-trained model for yawn detection instead of geometric heuristics.

---

## Authors

* HDS - Initial work and documentation

---
