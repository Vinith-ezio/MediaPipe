import os
import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

CAMERA_URL = "http://192.168.1.110:8080/video"

NUM_FACES = 2

MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MIN_FACE_TRACKING_CONFIDENCE = 0.5

SHOW_LANDMARK_INDEX = False


# ============================================================
# FACE MESH CONNECTIONS
# ============================================================

# MediaPipe Face Mesh topology.
# These are the main connections used for visualization.

FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10)
]


LEFT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145),
    (145, 153), (153, 154), (154, 155), (155, 133),
    (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33)
]


RIGHT_EYE = [
    (362, 382), (382, 381), (381, 380), (380, 374),
    (374, 373), (373, 390), (390, 249), (249, 263),
    (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362)
]


OUTER_LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84),
    (84, 17), (17, 314), (314, 405), (405, 321),
    (321, 375), (375, 291), (291, 409), (409, 270),
    (270, 269), (269, 267), (267, 0), (0, 37),
    (37, 39), (39, 40), (40, 185), (185, 61)
]


# ============================================================
# DRAW CONNECTIONS
# ============================================================

def draw_connections(
    image,
    landmarks,
    connections,
    width,
    height
):
    """
    Draw landmark connections manually.
    """

    for start_idx, end_idx in connections:

        if (
            start_idx >= len(landmarks)
            or end_idx >= len(landmarks)
        ):
            continue

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * width)
        y1 = int(start.y * height)

        x2 = int(end.x * width)
        y2 = int(end.y * height)

        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )


# ============================================================
# DRAW FACE LANDMARKS
# ============================================================

def draw_face_landmarks(image, detection_result):

    if not detection_result.face_landmarks:
        return image

    height, width, _ = image.shape

    for face_landmarks in detection_result.face_landmarks:

        # ----------------------------------------------------
        # Draw face points
        # ----------------------------------------------------

        for index, landmark in enumerate(face_landmarks):

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:

                cv2.circle(
                    image,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

                if SHOW_LANDMARK_INDEX:

                    cv2.putText(
                        image,
                        str(index),
                        (x + 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

        # ----------------------------------------------------
        # Draw face outline
        # ----------------------------------------------------

        draw_connections(
            image,
            face_landmarks,
            FACE_OVAL,
            width,
            height
        )

        # ----------------------------------------------------
        # Draw eyes
        # ----------------------------------------------------

        draw_connections(
            image,
            face_landmarks,
            LEFT_EYE,
            width,
            height
        )

        draw_connections(
            image,
            face_landmarks,
            RIGHT_EYE,
            width,
            height
        )

        # ----------------------------------------------------
        # Draw lips
        # ----------------------------------------------------

        draw_connections(
            image,
            face_landmarks,
            OUTER_LIPS,
            width,
            height
        )

    return image


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_information(image, face_count, fps):

    cv2.rectangle(
        image,
        (10, 10),
        (350, 105),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        "Face Landmarker",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"Faces: {face_count}",
        (20, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    global SHOW_LANDMARK_INDEX

    print("=" * 60)
    print("MediaPipe Face Landmarker")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: Model not found.")
        print(f"Expected: {MODEL_PATH}")
        return

    print(f"\nModel: {MODEL_PATH}")

    # --------------------------------------------------------
    # Create Face Landmarker
    # --------------------------------------------------------

    print("Loading Face Landmarker...")

    try:

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=NUM_FACES,
            min_face_detection_confidence=(
                MIN_FACE_DETECTION_CONFIDENCE
            ),
            min_face_presence_confidence=(
                MIN_FACE_PRESENCE_CONFIDENCE
            ),
            min_tracking_confidence=(
                MIN_FACE_TRACKING_CONFIDENCE
            ),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        detector = vision.FaceLandmarker.create_from_options(
            options
        )

    except Exception as e:

        print("\nERROR: Could not create Face Landmarker.")
        print(e)
        return

    print("Face Landmarker loaded successfully.")

    # --------------------------------------------------------
    # Open phone camera
    # --------------------------------------------------------

    print("\nOpening phone camera...")
    print(CAMERA_URL)

    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():

        print("\nERROR: Could not open camera.")
        print("Check your phone camera stream.")
        detector.close()
        return

    print("Camera connected successfully.")

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    previous_time = time.time()

    # MediaPipe VIDEO mode requires monotonically
    # increasing timestamps.
    timestamp_ms = 0

    print("\nStarting Face Landmarker...")
    print("Press Q to quit.")
    print("Press I to show/hide landmark indices.")

    # --------------------------------------------------------
    # Processing loop
    # --------------------------------------------------------

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print("WARNING: Could not read frame.")
                continue

            # ------------------------------------------------
            # Mirror image
            # ------------------------------------------------

            frame = cv2.flip(frame, 1)

            # ------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # MediaPipe image
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp_ms = int(time.time() * 1000)

            # ------------------------------------------------
            # Detect
            # ------------------------------------------------

            detection_result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            frame = draw_face_landmarks(
                frame,
                detection_result
            )

            # ------------------------------------------------
            # Face count
            # ------------------------------------------------

            face_count = len(
                detection_result.face_landmarks
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.time()

            elapsed = current_time - previous_time

            if elapsed > 0:
                fps = 1.0 / elapsed
            else:
                fps = 0.0

            previous_time = current_time

            # ------------------------------------------------
            # Display information
            # ------------------------------------------------

            frame = draw_information(
                frame,
                face_count,
                fps
            )

            # ------------------------------------------------
            # Display frame
            # ------------------------------------------------

            cv2.imshow(
                "MediaPipe Face Landmarker",
                frame
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("i"):

                SHOW_LANDMARK_INDEX = (
                    not SHOW_LANDMARK_INDEX
                )

                print(
                    "Landmark indices:",
                    SHOW_LANDMARK_INDEX
                )

    except KeyboardInterrupt:

        print("\nStopped by user.")

    except Exception as e:

        print("\nRuntime error:")
        print(e)

    finally:

        print("\nCleaning up...")

        cap.release()

        detector.close()

        cv2.destroyAllWindows()

        print("Camera released.")
        print("Face Landmarker closed.")
        print("Program finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()