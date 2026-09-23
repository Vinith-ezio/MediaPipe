import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

PHONE_CAMERA_URL = "http://192.168.1.110:8080/video"

MODEL_PATH = "models/holistic_landmarker.task"


# ============================================================
# LANDMARK CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [
    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32),
]


HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17),
]


# ============================================================
# FUNCTION 1
# Initialize Holistic Landmarker
# ============================================================

def initialize_holistic():

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,

        running_mode=vision.RunningMode.IMAGE,

        output_face_blendshapes=True,

        output_segmentation_mask=False
    )

    holistic = vision.HolisticLandmarker.create_from_options(
        options
    )

    return holistic


# ============================================================
# FUNCTION 2
# Open Phone Camera
# ============================================================

def open_camera():

    cap = cv2.VideoCapture(
        PHONE_CAMERA_URL
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not connect to phone camera."
        )

        return None

    print(
        "Phone camera connected."
    )

    return cap


# ============================================================
# FUNCTION 3
# Convert OpenCV Frame → MediaPipe Image
# ============================================================

def convert_to_mp_image(frame):

    # OpenCV uses BGR
    # MediaPipe expects RGB

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    return mp_image


# ============================================================
# FUNCTION 4
# Run Holistic Detection
# ============================================================

def detect_holistic(
    holistic,
    mp_image
):

    result = holistic.detect(
        mp_image
    )

    return result


# ============================================================
# FUNCTION 5
# Convert Normalized Coordinates → Pixel Coordinates
# ============================================================

def landmark_to_pixel(
    landmark,
    frame
):

    h, w, _ = frame.shape

    x = int(
        landmark.x * w
    )

    y = int(
        landmark.y * h
    )

    return x, y


# ============================================================
# FUNCTION 6
# Draw Pose Landmarks
# ============================================================

def draw_pose(
    frame,
    pose_landmarks
):

    if not pose_landmarks:
        return


    # --------------------------------------------------------
    # MediaPipe result can contain:
    #
    # [landmark, landmark, ...]
    #
    # or a single landmark object.
    #
    # Normalize it into a list.
    # --------------------------------------------------------

    if hasattr(
        pose_landmarks,
        "x"
    ):

        landmarks = [
            pose_landmarks
        ]

    else:

        landmarks = pose_landmarks


    # --------------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------------

    for landmark in landmarks:

        x, y = landmark_to_pixel(
            landmark,
            frame
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )


    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for start, end in POSE_CONNECTIONS:

        if (
            start >= len(landmarks)
            or
            end >= len(landmarks)
        ):

            continue


        x1, y1 = landmark_to_pixel(
            landmarks[start],
            frame
        )

        x2, y2 = landmark_to_pixel(
            landmarks[end],
            frame
        )


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


# ============================================================
# FUNCTION 7
# Draw Hand Landmarks
# ============================================================

def draw_hand(
    frame,
    hand_landmarks
):

    if not hand_landmarks:
        return


    # Normalize structure

    if hasattr(
        hand_landmarks,
        "x"
    ):

        landmarks = [
            hand_landmarks
        ]

    else:

        landmarks = hand_landmarks


    # --------------------------------------------------------
    # Draw points
    # --------------------------------------------------------

    for landmark in landmarks:

        x, y = landmark_to_pixel(
            landmark,
            frame
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 0, 0),
            -1
        )


    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for start, end in HAND_CONNECTIONS:

        if (
            start >= len(landmarks)
            or
            end >= len(landmarks)
        ):

            continue


        x1, y1 = landmark_to_pixel(
            landmarks[start],
            frame
        )

        x2, y2 = landmark_to_pixel(
            landmarks[end],
            frame
        )


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


# ============================================================
# FUNCTION 8
# Draw Face Landmarks
# ============================================================

def draw_face(
    frame,
    face_landmarks
):

    if not face_landmarks:
        return


    # Normalize structure

    if hasattr(
        face_landmarks,
        "x"
    ):

        landmarks = [
            face_landmarks
        ]

    else:

        landmarks = face_landmarks


    # --------------------------------------------------------
    # Draw face points
    # --------------------------------------------------------

    for landmark in landmarks:

        x, y = landmark_to_pixel(
            landmark,
            frame
        )

        cv2.circle(
            frame,
            (x, y),
            1,
            (0, 0, 255),
            -1
        )


# ============================================================
# FUNCTION 9
# Get Landmark Counts
# ============================================================

def get_landmark_counts(
    result
):

    face_count = 0

    pose_count = 0

    left_hand_count = 0

    right_hand_count = 0


    # --------------------------------------------------------
    # Face
    # --------------------------------------------------------

    if result.face_landmarks:

        face_data = result.face_landmarks

        if hasattr(
            face_data,
            "x"
        ):

            face_count = 1

        else:

            try:

                face_count = len(
                    face_data[0]
                )

            except (
                TypeError,
                IndexError
            ):

                face_count = len(
                    face_data
                )


    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    if result.pose_landmarks:

        pose_data = result.pose_landmarks

        if hasattr(
            pose_data,
            "x"
        ):

            pose_count = 1

        else:

            try:

                pose_count = len(
                    pose_data[0]
                )

            except (
                TypeError,
                IndexError
            ):

                pose_count = len(
                    pose_data
                )


    # --------------------------------------------------------
    # Left hand
    # --------------------------------------------------------

    if result.left_hand_landmarks:

        left_data = result.left_hand_landmarks

        if hasattr(
            left_data,
            "x"
        ):

            left_hand_count = 1

        else:

            try:

                left_hand_count = len(
                    left_data[0]
                )

            except (
                TypeError,
                IndexError
            ):

                left_hand_count = len(
                    left_data
                )


    # --------------------------------------------------------
    # Right hand
    # --------------------------------------------------------

    if result.right_hand_landmarks:

        right_data = result.right_hand_landmarks

        if hasattr(
            right_data,
            "x"
        ):

            right_hand_count = 1

        else:

            try:

                right_hand_count = len(
                    right_data[0]
                )

            except (
                TypeError,
                IndexError
            ):

                right_hand_count = len(
                    right_data
                )


    return (
        face_count,
        pose_count,
        left_hand_count,
        right_hand_count
    )


# ============================================================
# FUNCTION 10
# Calculate FPS
# ============================================================

def calculate_fps(
    previous_time
):

    current_time = time.time()

    elapsed_time = (
        current_time - previous_time
    )


    if elapsed_time <= 0:

        return 0, current_time


    fps = 1 / elapsed_time

    return fps, current_time


# ============================================================
# FUNCTION 11
# Display Information
# ============================================================

def display_information(
    frame,
    result,
    fps
):

    (
        face_count,
        pose_count,
        left_hand_count,
        right_hand_count
    ) = get_landmark_counts(
        result
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Face
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Face: {face_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Pose: {pose_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Left hand
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Left Hand: {left_hand_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Right hand
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Right Hand: {right_hand_count}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Q: Quit | P: Print",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


# ============================================================
# FUNCTION 12
# Print Landmark Information
# ============================================================

def print_holistic_information(
    result
):

    print()
    print("=" * 60)
    print("HOLISTIC LANDMARK INFORMATION")
    print("=" * 60)


    # --------------------------------------------------------
    # Face
    # --------------------------------------------------------

    if result.face_landmarks:

        try:

            print(
                "Face landmarks:",
                len(
                    result.face_landmarks[0]
                )
            )

        except TypeError:

            print(
                "Face landmarks: 1"
            )

    else:

        print(
            "Face landmarks: Not detected"
        )


    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    if result.pose_landmarks:

        try:

            print(
                "Pose landmarks:",
                len(
                    result.pose_landmarks[0]
                )
            )

        except TypeError:

            print(
                "Pose landmarks: 1"
            )

    else:

        print(
            "Pose landmarks: Not detected"
        )


    # --------------------------------------------------------
    # Left hand
    # --------------------------------------------------------

    if result.left_hand_landmarks:

        try:

            print(
                "Left hand landmarks:",
                len(
                    result.left_hand_landmarks[0]
                )
            )

        except TypeError:

            print(
                "Left hand landmarks: 1"
            )

    else:

        print(
            "Left hand landmarks: Not detected"
        )


    # --------------------------------------------------------
    # Right hand
    # --------------------------------------------------------

    if result.right_hand_landmarks:

        try:

            print(
                "Right hand landmarks:",
                len(
                    result.right_hand_landmarks[0]
                )
            )

        except TypeError:

            print(
                "Right hand landmarks: 1"
            )

    else:

        print(
            "Right hand landmarks: Not detected"
        )


    # --------------------------------------------------------
    # Face blendshapes
    # --------------------------------------------------------

    if result.face_blendshapes:

        try:

            print(
                "Face blendshapes:",
                len(
                    result.face_blendshapes[0]
                )
            )

        except TypeError:

            print(
                "Face blendshapes available."
            )

    else:

        print(
            "Face blendshapes: Not detected"
        )


    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    if result.segmentation_masks:

        print(
            "Segmentation mask: Available"
        )

    else:

        print(
            "Segmentation mask: Not enabled"
        )


    print("=" * 60)


# ============================================================
# FUNCTION 13
# Cleanup
# ============================================================

def cleanup(
    cap,
    holistic
):

    cap.release()

    holistic.close()

    cv2.destroyAllWindows()

    print()
    print("Camera released.")
    print("Holistic Landmarker closed.")
    print("Program finished.")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MediaPipe Holistic Landmarker")
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    print(
        "Loading Holistic Landmarker..."
    )

    holistic = initialize_holistic()

    print(
        "Holistic Landmarker loaded successfully."
    )


    # --------------------------------------------------------
    # Open phone camera
    # --------------------------------------------------------

    cap = open_camera()


    if cap is None:

        holistic.close()

        return


    print()
    print(
        "Holistic detection started."
    )

    print(
        "Press Q to quit."
    )

    print(
        "Press P to print landmark information."
    )


    # --------------------------------------------------------
    # FPS initialization
    # --------------------------------------------------------

    previous_time = time.time()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        success, frame = cap.read()


        if not success:

            print(
                "ERROR: Failed to receive frame."
            )

            break


        # ----------------------------------------------------
        # Convert frame
        # ----------------------------------------------------

        mp_image = convert_to_mp_image(
            frame
        )


        # ----------------------------------------------------
        # Run Holistic
        # ----------------------------------------------------

        result = detect_holistic(
            holistic,
            mp_image
        )


        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        if result.face_landmarks:

            draw_face(
                frame,
                result.face_landmarks
            )


        # ----------------------------------------------------
        # POSE
        # ----------------------------------------------------

        if result.pose_landmarks:

            draw_pose(
                frame,
                result.pose_landmarks
            )


        # ----------------------------------------------------
        # LEFT HAND
        # ----------------------------------------------------

        if result.left_hand_landmarks:

            draw_hand(
                frame,
                result.left_hand_landmarks
            )


        # ----------------------------------------------------
        # RIGHT HAND
        # ----------------------------------------------------

        if result.right_hand_landmarks:

            draw_hand(
                frame,
                result.right_hand_landmarks
            )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        fps, previous_time = calculate_fps(
            previous_time
        )


        # ----------------------------------------------------
        # Information overlay
        # ----------------------------------------------------

        display_information(
            frame,
            result,
            fps
        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "MediaPipe Holistic Landmarker",
            frame
        )


        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if key == ord("q"):

            break


        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        if key == ord("p"):

            print_holistic_information(
                result
            )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cleanup(
        cap,
        holistic
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()