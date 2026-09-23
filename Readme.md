# MediaPipe Computer Vision Practice

A practical **MediaPipe-based computer vision practice project** for
learning and implementing real-time landmark detection, object
detection, and holistic human/face/hand/pose analysis using Python.

The project is designed as a progressive learning environment, starting
with individual MediaPipe tasks and moving toward combining multiple
vision capabilities.

------------------------------------------------------------------------

## Project Overview

This project currently contains practice implementations for:

-   **Face Landmarker**
-   **Hand Landmarker**
-   **Pose Landmarker**
-   **Holistic Landmarker**
-   **Object Detector**
-   **Phone Camera Streaming**

The camera source can be a mobile phone camera streamed over the local
network, which is useful when the development computer does not have a
built-in webcam.

### Current Architecture

``` text
                    Phone Camera
                         |
                         v
                 Video Stream / URL
                         |
                         v
                   OpenCV Capture
                         |
                         v
                  RGB Video Frames
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          Face        Hand        Pose
       Landmarker   Landmarker  Landmarker
             |           |           |
             +-----------+-----------+
                         |
                         v
                  Vision Results
                         |
                         v
                Visualization / Analysis
```

Holistic Landmarker can be used when multiple human landmark components
need to be processed together.

Object detection is handled as a separate MediaPipe task.

------------------------------------------------------------------------

## Project Structure

``` text
Media_Pipe/
│
├── .venv/
│
├── models/
│   ├── face_landmarker.task
│   ├── hand_landmarker.task
│   ├── holistic_landmarker.task
│   ├── object_detector.tflite
│   └── pose_landmarker_lite.task
│
├── face_landmarker.py
├── hand_landmarks.py
├── holistic_landmarker.py
├── object_detector.py
├── phone_camera.py
├── pose_landmarks.py
│
├── .gitignore
└── Readme.md
```

### Directory Description

  -----------------------------------------------------------------------
  Path                                Purpose
  ----------------------------------- -----------------------------------
  `.venv/`                            Python virtual environment

  `models/`                           MediaPipe task/model files

  `face_landmarker.py`                Face landmark detection practice

  `hand_landmarks.py`                 Hand landmark detection practice

  `holistic_landmarker.py`            Combined human landmark detection
                                      practice

  `object_detector.py`                Object detection practice

  `phone_camera.py`                   Phone camera/video stream handling

  `pose_landmarks.py`                 Human pose landmark detection
                                      practice

  `Readme.md`                         Project documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 1. Technologies Used

-   Python
-   MediaPipe
-   MediaPipe Tasks API
-   OpenCV
-   NumPy
-   OpenCV VideoCapture
-   TensorFlow Lite models through MediaPipe Tasks
-   Python virtual environment

The project primarily uses the **MediaPipe Tasks API** for the landmark
and detection tasks.

------------------------------------------------------------------------

# 2. Environment Setup

## Create Virtual Environment

From the project directory:

``` powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

``` text
(.venv) PS E:\Media_Pipe>
```

------------------------------------------------------------------------

## Install Dependencies

Install the main packages:

``` powershell
pip install mediapipe opencv-python numpy
```

You can verify MediaPipe:

``` powershell
python -c "import mediapipe as mp; print(mp.__version__)"
```

Verify OpenCV:

``` powershell
python -c "import cv2; print(cv2.__version__)"
```

------------------------------------------------------------------------

# 3. MediaPipe Models

The `models` directory contains the model/task files required by the
Python scripts.

Current model files:

``` text
models/
├── face_landmarker.task
├── hand_landmarker.task
├── holistic_landmarker.task
├── object_detector.tflite
└── pose_landmarker_lite.task
```

The scripts reference these files locally instead of downloading the
model every time the program starts.

## Important

The model path used in a script must match the actual filename.

For example:

``` python
MODEL_PATH = "models/face_landmarker.task"
```

If the model is missing, the corresponding script will not be able to
initialize the MediaPipe task.

------------------------------------------------------------------------

# 4. Face Landmarker

File:

``` text
face_landmarker.py
```

The Face Landmarker is used to detect facial landmarks from video
frames.

### Processing Pipeline

``` text
Camera Frame
     |
     v
BGR -> RGB
     |
     v
MediaPipe Image
     |
     v
Face Landmarker
     |
     +----> Face Landmarks
     |
     +----> Face Blendshapes
     |
     +----> Facial Transformation Matrix
     |
     v
Visualization
```

The face landmarks can later be used for:

-   Face geometry
-   Eye analysis
-   Blink detection
-   Mouth analysis
-   Smile-related measurements
-   Head pose analysis
-   Facial movement analysis

### Run

``` powershell
python face_landmarker.py
```

Typical keyboard controls:

``` text
Q -> Quit
I -> Toggle landmark indices
```

------------------------------------------------------------------------

# 5. Hand Landmarker

File:

``` text
hand_landmarks.py
```

The Hand Landmarker detects hand keypoints/landmarks from the camera
stream.

### Processing Pipeline

``` text
Camera Frame
     |
     v
RGB Conversion
     |
     v
Hand Landmarker
     |
     v
Hand Landmarks
     |
     v
Visualization
```

Hand landmarks can be used for:

-   Finger tracking
-   Hand position
-   Gesture recognition
-   Distance calculation
-   Joint angle calculation
-   Custom gesture classification

A useful next step after landmark detection is to calculate geometric
features such as distances and joint angles.

Example:

``` text
Thumb Tip -------- Index Tip
       \          /
        \        /
         Distance
```

------------------------------------------------------------------------

# 6. Pose Landmarker

File:

``` text
pose_landmarks.py
```

The Pose Landmarker is used to identify human body landmarks.

### Processing Pipeline

``` text
Camera Frame
     |
     v
Pose Landmarker
     |
     v
Body Landmarks
     |
     +----> Shoulder
     +----> Elbow
     +----> Wrist
     +----> Hip
     +----> Knee
     +----> Ankle
     |
     v
Pose Visualization
```

Pose landmarks can be used for:

-   Body tracking
-   Joint angle calculation
-   Posture analysis
-   Exercise analysis
-   Human activity analysis
-   Movement analysis

------------------------------------------------------------------------

# 7. Holistic Landmarker

File:

``` text
holistic_landmarker.py
```

The Holistic Landmarker is intended for combined human landmark
analysis.

Conceptually:

``` text
                 Human
                   |
        +----------+----------+
        |          |          |
        v          v          v
      Face        Pose       Hands
        |          |          |
        +----------+----------+
                   |
                   v
          Combined Analysis
```

This is useful when an application needs information from multiple parts
of the body at the same time.

Possible applications include:

-   Full-body tracking
-   Gesture + pose analysis
-   Human-computer interaction
-   Exercise analysis
-   Interactive applications

------------------------------------------------------------------------

# 8. Object Detector

File:

``` text
object_detector.py
```

The Object Detector performs object detection using the configured
TensorFlow Lite/MediaPipe-compatible model.

Model:

``` text
models/object_detector.tflite
```

Typical detection output can contain:

``` text
Object
Class
Confidence
Bounding Box
```

Conceptually:

``` text
Camera Frame
     |
     v
Object Detector
     |
     v
+-------------------------+
| Class                   |
| Confidence              |
| Bounding Box            |
+-------------------------+
```

Object detection is different from landmark detection.

### Landmark Detection

Returns key points:

``` text
     •
   •   •
     •
```

### Object Detection

Returns regions/bounding boxes:

``` text
+-------------------+
|                   |
|      Object       |
|                   |
+-------------------+
```

------------------------------------------------------------------------

# 9. Phone Camera

File:

``` text
phone_camera.py
```

The project can use a mobile phone as the camera source.

Example stream:

``` text
http://192.168.1.110:8080/video
```

The exact IP address depends on the phone's current local network
address.

A typical OpenCV connection is:

``` python
cap = cv2.VideoCapture(CAMERA_URL)
```

The phone and development computer normally need to be connected to the
same local network for this type of stream.

## Important

The IP address shown above is an example/current development
configuration. If the phone receives a different IP address, update the
camera URL.

------------------------------------------------------------------------

# 10. OpenCV + MediaPipe Pipeline

Most of the real-time scripts follow this general processing flow:

``` text
             Camera
                |
                v
        OpenCV VideoCapture
                |
                v
             Frame
                |
                v
          BGR -> RGB
                |
                v
        MediaPipe mp.Image
                |
                v
          MediaPipe Task
                |
                v
          Detection Result
                |
                v
       Visualization / Logic
                |
                v
          cv2.imshow()
```

This pipeline is fundamental for real-time computer vision applications.

------------------------------------------------------------------------

# 11. Landmark Coordinates

MediaPipe landmarks generally provide normalized coordinates.

Conceptually:

``` text
x -> Horizontal position
y -> Vertical position
z -> Relative depth information
```

For an image with:

``` text
width  = W
height = H
```

pixel coordinates can be obtained approximately as:

``` python
pixel_x = int(landmark.x * W)
pixel_y = int(landmark.y * H)
```

This allows normalized landmark coordinates to be mapped onto the camera
frame.

------------------------------------------------------------------------

# 12. Geometry Practice

After basic landmark detection, geometric calculations can be performed.

## Distance

For two points:

``` text
P1 = (x1, y1)
P2 = (x2, y2)
```

Euclidean distance:

``` text
distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

Python:

``` python
import math

distance = math.sqrt(
    (x2 - x1) ** 2 +
    (y2 - y1) ** 2
)
```

------------------------------------------------------------------------

## Angle

For three points:

``` text
A ---- B ---- C
      angle
```

The required angle is:

``` text
angle ABC
```

This can be calculated using vector mathematics.

These calculations are useful for:

-   Finger gestures
-   Joint angles
-   Eye measurements
-   Mouth measurements
-   Pose analysis

------------------------------------------------------------------------

# 13. Planned Face Analysis

The Face Landmarker can be extended beyond visualization.

Recommended progression:

``` text
Phase 1
Face Landmark Detection
        |
        v
Phase 2
Landmark Coordinate Analysis
        |
        v
Phase 3
Distance + Angle Calculations
        |
        v
Phase 4
Eye Landmark Analysis
        |
        v
Phase 5
Blink Detection
        |
        v
Phase 6
Mouth / Lip Analysis
        |
        v
Phase 7
Smile / Expression Features
        |
        v
Phase 8
Head Pose / Orientation
        |
        v
Phase 9
Face + Hand + Pose Integration
```

------------------------------------------------------------------------

# 14. Planned Computer Vision Learning Path

The project can progressively move from individual models to complete
vision systems.

``` text
                    MediaPipe Practice
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
       Face             Hand              Pose
          |                |                |
          +----------------+----------------+
                           |
                           v
                       Holistic
                           |
                           v
                  Object Detection
                           |
                           v
                  Geometry Features
                           |
                           v
                 Gesture Recognition
                           |
                           v
                 Human Understanding
                           |
                           v
                Real-Time CV Application
```

------------------------------------------------------------------------

# 15. Recommended Practice Order

Follow the modules in this order:

### Step 1 --- Phone Camera

Understand:

-   Video URL
-   OpenCV VideoCapture
-   Frame reading
-   BGR/RGB conversion

### Step 2 --- Hand Landmarker

Learn:

-   Landmark coordinates
-   Landmark visualization
-   Finger points
-   Distance
-   Joint angles
-   Gesture logic

### Step 3 --- Pose Landmarker

Learn:

-   Body landmarks
-   Joint relationships
-   Pose geometry
-   Body movement

### Step 4 --- Face Landmarker

Learn:

-   Face landmarks
-   Facial geometry
-   Eye landmarks
-   Mouth landmarks
-   Blink analysis
-   Facial movement

### Step 5 --- Holistic

Combine:

-   Face
-   Hands
-   Pose

### Step 6 --- Object Detection

Learn:

-   Bounding boxes
-   Classes
-   Confidence scores
-   Detection filtering

### Step 7 --- Integrated Vision Application

Combine multiple vision outputs into a single application.

------------------------------------------------------------------------

# 16. Running the Project

Activate the environment:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Then run the required script.

### Face

``` powershell
python face_landmarker.py
```

### Hand

``` powershell
python hand_landmarks.py
```

### Pose

``` powershell
python pose_landmarks.py
```

### Holistic

``` powershell
python holistic_landmarker.py
```

### Object Detection

``` powershell
python object_detector.py
```

### Phone Camera Test

``` powershell
python phone_camera.py
```

------------------------------------------------------------------------

# 17. Troubleshooting

## MediaPipe Import Error

If MediaPipe cannot be imported:

``` powershell
pip install --upgrade mediapipe
```

Verify:

``` powershell
python -c "import mediapipe; print(mediapipe.__version__)"
```

This project uses the newer MediaPipe Tasks API. Avoid assuming that
older examples using:

``` python
mp.solutions
```

are compatible with every installed MediaPipe version.

------------------------------------------------------------------------

## Model Not Found

If a script reports that a model is missing, verify:

``` powershell
Get-ChildItem .\models
```

Check that the required model filename exactly matches the path in the
Python script.

------------------------------------------------------------------------

## Camera Cannot Be Opened

Check:

1.  Phone camera streaming is running.
2.  PC and phone are on the same network.
3.  The IP address is correct.
4.  The stream URL is correct.
5.  Windows Firewall is not blocking the connection.

Test the URL independently before debugging MediaPipe.

------------------------------------------------------------------------

## Low FPS

Real-time performance depends on:

-   Camera resolution
-   Network streaming latency
-   CPU performance
-   MediaPipe model
-   Number of detected objects/faces/hands
-   Additional processing performed per frame

For initial practice, prioritize correctness before optimization.

------------------------------------------------------------------------

# 18. Model Download Commands

The required MediaPipe model files should be stored inside:

```text
models/
```

## Face Landmarker

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
```

## Hand Landmarker

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

## Pose Landmarker Lite

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" -OutFile "models\pose_landmarker_lite.task"
```

## Verify the downloaded models

```powershell
Get-ChildItem .\models
```

For example:

```text
models/
├── face_landmarker.task
├── hand_landmarker.task
├── holistic_landmarker.task
├── object_detector.tflite
└── pose_landmarker_lite.task
```

> **Note:** `holistic_landmarker.task` and `object_detector.tflite` are separate model files. Their download URLs depend on the exact model variant being used by the corresponding scripts. Use the official MediaPipe model source for the exact model required by those scripts rather than assuming that every model uses the same URL pattern.

---

# Summary

This repository is a practical learning project for building real-time computer vision applications with **MediaPipe + OpenCV + Python**.

The current focus is on understanding individual MediaPipe tasks first and then progressively combining their outputs into more advanced computer vision applications.

The overall pipeline is:

```text
Camera
  ↓
Frame Acquisition
  ↓
Preprocessing
  ↓
AI / Vision Model
  ↓
Landmarks / Detections
  ↓
Feature Extraction
  ↓
Decision Logic
  ↓
Application Output
```
