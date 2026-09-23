# MediaPipe Computer Vision Practice

A practical computer-vision learning project using **MediaPipe Tasks, Python, OpenCV, and a mobile phone IP camera**.

The project focuses on understanding and practicing MediaPipe vision tasks from basic landmark detection through human-pose/face/hand analysis, holistic tracking, object detection, gesture logic, and real-time camera processing.

> **Note:** This README is a generic project guide. It intentionally focuses on concepts, setup, models, commands, architecture, and troubleshooting rather than explaining every Python line.

---

## 1. Project Objective

The objective of this project is to practice modern MediaPipe computer-vision tasks in a local Python environment.

The PC does not have a webcam, so a **mobile phone camera is used as the live video source** through an IP video stream.

The overall learning path is:

```text
Camera Input
     ↓
OpenCV
     ↓
MediaPipe Tasks
     ↓
Landmarks / Detections
     ↓
Feature Extraction
     ↓
Gesture / Application Logic
     ↓
Real-Time Computer Vision
```

---

## 2. What is MediaPipe?

[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) is a framework and collection of machine-learning/perception solutions for building computer-vision applications.

The MediaPipe Tasks API provides ready-to-use tasks for areas such as:

- Hand landmark detection
- Pose landmark detection
- Face landmark detection
- Holistic landmark detection
- Object detection
- Gesture-oriented processing
- Real-time vision applications

---

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| MediaPipe Tasks | Vision inference and landmark/detection tasks |
| OpenCV | Camera capture, frame processing and visualization |
| NumPy | Image and numerical processing |
| TensorFlow Lite | Model format used by compatible models |
| Python Virtual Environment | Isolated project environment |
| Mobile IP Camera | Live camera source |

---

## 4. Overall Architecture

```text
                 Mobile Phone Camera
                         │
                         ▼
                  IP Video Stream
                         │
                         ▼
                       OpenCV
                         │
                    BGR → RGB
                         │
                         ▼
                  MediaPipe Image
                         │
                         ▼
                 MediaPipe Task
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        Hand            Pose           Face
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Holistic / Object
                     Detection
                         │
                         ▼
              Feature / Gesture Logic
                         │
                         ▼
                Visualization / App
```

---

# 5. PC Environment Setup

## 5.1 Create the project

```powershell
mkdir E:\Media_Pipe
cd E:\Media_Pipe
```

## 5.2 Create a virtual environment

```powershell
python -m venv .venv
```

## 5.3 Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Expected prompt:

```text
(.venv) PS E:\Media_Pipe>
```

## 5.4 Install dependencies

```powershell
pip install mediapipe
pip install opencv-python
pip install numpy
```

Or:

```powershell
pip install mediapipe opencv-python numpy
```

## 5.5 Verify installation

```powershell
python --version
```

```powershell
python -c "import mediapipe as mp; print(mp.__version__)"
```

```powershell
python -c "import cv2; print(cv2.__version__)"
```

---

# 6. Project Structure

Recommended structure:

```text
E:\Media_Pipe
│
├── .venv/
│
├── models/
│   ├── hand_landmarker.task
│   ├── pose_landmarker_lite.task
│   ├── face_landmarker.task
│   ├── holistic_landmarker.task
│   └── object_detector.tflite
│
├── phone_camera.py
├── hand_landmarks.py
├── pose_landmarks.py
├── face_landmarker.py
├── holistic_landmarker.py
├── object_detector.py
│
├── README.md
└── .gitignore
```

---

# 7. Mobile Phone Camera

Since the PC does not have a webcam, the mobile phone is used as the camera.

```text
Mobile Phone
     ↓
IP Camera Application
     ↓
Wi-Fi / Local Network
     ↓
HTTP Video Stream
     ↓
OpenCV
     ↓
MediaPipe
```

Example camera URL:

```python
PHONE_CAMERA_URL = "http://192.168.1.110:8080/video"
```

### Important

The IP address is dependent on the local network and may change.

The phone and PC should be connected to a network that allows the PC to access the phone's video stream.

---

# 8. MediaPipe Tasks Practiced

| Task | Main Output | Typical Applications |
|---|---|---|
| Hand Landmarker | 21 hand landmarks | Hand tracking, finger analysis |
| Pose Landmarker | 33 body landmarks | Pose/posture/movement analysis |
| Face Landmarker | Facial landmarks and optional facial features | Face analysis |
| Holistic Landmarker | Face + pose + hands | Full-body human tracking |
| Object Detector | Bounding boxes + categories + scores | Object detection |

---

# 9. Hand Landmarker

The Hand Landmarker detects a hand and provides **21 hand landmarks**.

General pipeline:

```text
Input Image
     ↓
Hand Detection
     ↓
21 Hand Landmarks
     ↓
Coordinate Extraction
     ↓
Geometric Analysis
     ↓
Finger / Gesture Logic
```

The landmarks can be used for:

- Hand tracking
- Finger-state analysis
- Gesture recognition
- Distance measurement
- Angle calculation
- Human-computer interaction

---

# 10. Hand Landmark Structure

The 21 landmarks represent important points of the hand.

Conceptually:

```text
                 8
                 │
                 7
                 │
                 6
                 │
                 5
                 │
                 0 ───── 9 ───── 13 ───── 17
                 │
                 │
             Thumb
```

The complete hand model contains:

- Wrist
- Thumb
- Index finger
- Middle finger
- Ring finger
- Pinky

Each landmark provides normalized coordinate information.

---

# 11. Hand Geometry

After obtaining landmarks, geometric calculations can be performed.

## Distance

Euclidean distance can be used to measure the separation between two landmarks.

```text
distance(P1, P2)
```

Applications:

- Finger proximity
- Thumb/index relationship
- Gesture features
- Hand measurements

## Angle

Angles can be calculated from three landmarks.

```text
A → B → C

Angle ABC
```

Applications:

- Finger extension
- Finger bending
- Joint analysis
- Gesture rules

---

# 12. Finger Detection

MediaPipe provides the landmarks, but **application-specific finger-state logic** can be built using those landmarks.

General approach:

```text
Hand Landmarks
      ↓
Joint Angles
      +
Landmark Distances
      ↓
Finger State
      ↓
Open / Closed
```

Functions practiced in this project include:

```text
distance()
calculate_angle()
detect_fingers()
```

These functions are geometric/application-level logic and are not the same thing as the underlying MediaPipe landmark model.

---

# 13. Gesture Recognition

Gesture recognition can be built on top of the detected finger states.

```text
21 Hand Landmarks
        ↓
Finger Detection
        ↓
Finger States
        ↓
Rule-Based Logic
        ↓
Gesture
```

Practice examples include:

```text
Open Palm
Fist
One Finger
Two Fingers
Thumb Up
```

These rules are application-specific heuristics.

They should be tested across:

- Different users
- Different hand orientations
- Different distances from the camera
- Different lighting conditions
- Left/right hands

---

# 14. Pose Landmarker

The Pose Landmarker estimates human body landmarks.

The standard pose representation contains **33 body landmarks**.

General pipeline:

```text
Person
  ↓
Pose Detection
  ↓
33 Body Landmarks
  ↓
Coordinate Analysis
  ↓
Posture / Movement Analysis
```

Potential applications:

- Human pose estimation
- Posture analysis
- Exercise monitoring
- Movement analysis
- Human activity applications

Pose results can include image-relative landmarks and world-coordinate landmarks depending on the configured task/model.

---

# 15. Face Landmarker

Face Landmarker provides facial landmark information.

General pipeline:

```text
Face
 ↓
Face Detection / Landmarking
 ↓
Facial Landmarks
 ↓
Facial Feature Analysis
```

Potential applications:

- Facial landmark tracking
- Face geometry
- Facial movement analysis
- Head-related analysis
- Facial-expression-related features when supported/configured

---

# 16. Holistic Landmarker

Holistic Landmarker combines multiple human landmark components.

```text
                 Holistic
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        Face       Pose      Hands
```

It is useful when an application needs coordinated information from:

- Face
- Body pose
- Left hand
- Right hand

This makes Holistic Landmarker useful for full-body human interaction and motion analysis.

### Important API consideration

The Python Tasks API returns landmark collections. Individual landmark objects and landmark lists must not be treated as the same data structure.

For example, code should distinguish between:

```text
list of landmarks
```

and:

```text
single NormalizedLandmark
```

This distinction is important when iterating through Holistic results.

---

# 17. Object Detector

Object detection is different from landmark detection.

### Landmark detection

```text
Image
 ↓
Landmarks / Key Points
```

### Object detection

```text
Image
 ↓
Object
 ↓
Bounding Box
+
Category
+
Confidence Score
```

Typical output:

```text
Object
├── Category
├── Confidence
└── Bounding Box
```

Potential applications:

- Industrial inspection
- Object counting
- Scene understanding
- Product detection
- Real-time monitoring

---

# 18. `.task` vs `.tflite`

These two model formats should not automatically be treated as interchangeable.

### `.task`

A MediaPipe Task model asset/package designed to work with a particular MediaPipe Task.

Example:

```text
hand_landmarker.task
pose_landmarker_lite.task
face_landmarker.task
holistic_landmarker.task
```

### `.tflite`

TensorFlow Lite model format.

Example:

```text
object_detector.tflite
```

A `.tflite` model must be compatible with the selected MediaPipe task, including the expected model metadata and outputs.

---

# 19. Model Directory

Create the model directory:

```powershell
mkdir models
```

Check it:

```powershell
Get-ChildItem .\models
```

---

# 20. Model Download Commands

> **Important:** MediaPipe model assets and storage paths can change between releases. Verify the current official MediaPipe model documentation before relying on a model URL.

## Hand Landmarker

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile ".\models\hand_landmarker.task"
```

Verify:

```powershell
Test-Path .\models\hand_landmarker.task
```

Expected:

```text
True
```

---

## Pose Landmarker Lite

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" `
  -OutFile ".\models\pose_landmarker_lite.task"
```

Verify:

```powershell
Test-Path .\models\pose_landmarker_lite.task
```

---

## Face Landmarker

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile ".\models\face_landmarker.task"
```

Verify:

```powershell
Test-Path .\models\face_landmarker.task
```

---

## Holistic Landmarker

The model URL used during this project:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" `
  -OutFile ".\models\holistic_landmarker.task"
```

Verify:

```powershell
Test-Path .\models\holistic_landmarker.task
```

---

## Object Detector

For Object Detector, use a model asset specifically documented as compatible with the **MediaPipe Object Detector** task.

Do not assume that any `.tflite` model is automatically compatible.

Compatibility should be checked for:

- Model metadata
- Input format
- Output tensors
- Label information
- MediaPipe Task requirements

---

# 21. Verify All Models

```powershell
Get-ChildItem .\models
```

Or check individual files:

```powershell
Test-Path .\models\hand_landmarker.task
Test-Path .\models\pose_landmarker_lite.task
Test-Path .\models\face_landmarker.task
Test-Path .\models\holistic_landmarker.task
```

---

# 22. MediaPipe Running Modes

MediaPipe Tasks commonly supports different processing modes.

| Mode | Purpose |
|---|---|
| IMAGE | Process individual images |
| VIDEO | Process sequential video frames |
| LIVE_STREAM | Process live input asynchronously |

### IMAGE

```text
Image → Task → Result
```

Useful for:

- Single-image testing
- Debugging
- Offline processing

### VIDEO

```text
Frame 1 → Result
Frame 2 → Result
Frame 3 → Result
...
```

Useful for:

- Recorded videos
- Timestamp-based processing
- Video analysis

### LIVE_STREAM

```text
Camera
  ↓
Live Frames
  ↓
Async MediaPipe Processing
  ↓
Results
```

Useful for:

- Real-time applications
- Camera applications
- Low-latency pipelines

---

# 23. Standard OpenCV + MediaPipe Pipeline

```text
OpenCV Camera Frame
        ↓
      BGR
        ↓
   BGR → RGB
        ↓
MediaPipe Image
        ↓
MediaPipe Task
        ↓
Detection / Landmarks
        ↓
Application Logic
        ↓
Visualization
```

OpenCV commonly reads images in **BGR**, while MediaPipe processing expects the appropriate image representation such as **RGB** for standard camera-image workflows.

---

# 24. FPS vs Latency

FPS and latency are different measurements.

### FPS

Frames processed/displayed per second.

```text
Higher FPS
    ↓
Higher throughput
```

### Latency

Delay between the real-world scene and the displayed/processed result.

```text
Real Scene
    ↓
Camera
    ↓
Network
    ↓
Processing
    ↓
Display

Total Delay = Latency
```

It is possible to have:

```text
High FPS + High Latency
```

or:

```text
Low FPS + Low Latency
```

Therefore, FPS alone does not tell you whether a real-time system feels responsive.

---

# 25. IP Camera Latency

With a mobile phone camera, the complete pipeline can be:

```text
Phone Camera
     ↓
Frame Capture
     ↓
Encoding
     ↓
Wi-Fi / Network
     ↓
HTTP Stream
     ↓
OpenCV Buffer
     ↓
MediaPipe
     ↓
Rendering
     ↓
Display
```

Therefore, visible latency may come from several stages.

### Debugging approach

Test progressively:

```text
1. Phone Camera
       ↓
2. Raw IP Stream
       ↓
3. OpenCV
       ↓
4. MediaPipe
       ↓
5. Visualization
```

This helps identify whether the delay is primarily caused by:

- Network
- Camera buffering
- OpenCV buffering
- Inference
- Rendering
- Processing backlog

---

# 26. Common Problems Encountered

## Model file not found

Example:

```text
FileNotFoundError:
Unable to open file at models/hand_landmarker.task
```

Check:

```powershell
Test-Path .\models\hand_landmarker.task
```

If it returns:

```text
False
```

the model is not present at the expected path.

---

## Holistic option error

Example:

```text
TypeError:
HolisticLandmarkerOptions.__init__()
got an unexpected keyword argument
```

This generally indicates that the code is using an option name that is not supported by the installed MediaPipe version.

Check:

```powershell
python -c "import mediapipe as mp; print(mp.__version__)"
```

Then verify the API for that installed version.

---

## Holistic landmark iteration error

Example:

```text
TypeError:
'NormalizedLandmark' object is not iterable
```

This occurs when code expects a collection but receives a single landmark object.

The result hierarchy should be inspected before writing loops.

---

## Camera connection failure

Check:

- Phone and PC network connection
- IP address
- Port
- IP-camera application
- Camera stream URL
- Windows firewall/network restrictions

---

# 27. Recommended Learning Progression

```text
Phase 1
Camera + OpenCV
       ↓
Phase 2
Hand Landmarker
       ↓
Phase 3
Finger Geometry
       ↓
Phase 4
Gesture Recognition
       ↓
Phase 5
Pose Landmarker
       ↓
Phase 6
Face Landmarker
       ↓
Phase 7
Holistic Landmarker
       ↓
Phase 8
Object Detection
       ↓
Phase 9
LIVE_STREAM
       ↓
Phase 10
Latency / Performance Optimization
       ↓
Phase 11
Multi-task Integration
       ↓
Phase 12
Real Computer-Vision Application
```

---

# 28. Useful Commands

## Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## Check Python

```powershell
python --version
```

## Check MediaPipe

```powershell
python -c "import mediapipe as mp; print(mp.__version__)"
```

## Check OpenCV

```powershell
python -c "import cv2; print(cv2.__version__)"
```

## Check installed packages

```powershell
pip list
```

## List models

```powershell
Get-ChildItem .\models
```

## Run camera test

```powershell
python phone_camera.py
```

## Run Hand Landmarker

```powershell
python hand_landmarks.py
```

## Run Pose Landmarker

```powershell
python pose_landmarks.py
```

## Run Face Landmarker

```powershell
python face_landmarker.py
```

## Run Holistic Landmarker

```powershell
python holistic_landmarker.py
```

## Run Object Detector

```powershell
python object_detector.py
```

---

# 29. Recommended Git Configuration

Create `.gitignore`:

```text
.venv/
__pycache__/
*.pyc
```

If model files are large, decide separately whether they should be committed to GitHub or downloaded during setup.

---

# 30. Final Architecture

```text
                  MOBILE CAMERA
                       │
                       ▼
                IP VIDEO STREAM
                       │
                       ▼
                    OpenCV
                       │
                  BGR → RGB
                       │
                       ▼
                MediaPipe Tasks
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
      Hand            Pose           Face
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                  HOLISTIC
                       │
                       ▼
                OBJECT DETECTOR
                       │
                       ▼
              FEATURE EXTRACTION
                       │
                       ▼
             GESTURE / CV LOGIC
                       │
                       ▼
               REAL-TIME OUTPUT
```

---

# 31. Summary

This project provides practical exposure to a complete MediaPipe computer-vision workflow:

```text
Camera Input
     ↓
OpenCV
     ↓
MediaPipe
     ↓
Landmarks / Detection
     ↓
Geometric Features
     ↓
Gesture / Recognition Logic
     ↓
Real-Time Application
```

The key concepts practiced are:

- Mobile IP camera integration
- OpenCV video capture
- MediaPipe Tasks
- Hand landmarks
- Pose landmarks
- Face landmarks
- Holistic landmarks
- Object detection
- Distance calculation
- Angle calculation
- Finger-state detection
- Gesture recognition
- Model asset management
- `.task` and `.tflite`
- IMAGE / VIDEO / LIVE_STREAM modes
- FPS and latency
- Real-time computer-vision pipelines

The next major focus is to complete the remaining model validation, move the camera pipeline toward **LIVE_STREAM**, measure latency/throughput, and then combine multiple MediaPipe capabilities into a practical computer-vision application.

---

## References

- [MediaPipe Documentation](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/tasks)
- [MediaPipe Python](https://ai.google.dev/edge/mediapipe/solutions/setup_python)
