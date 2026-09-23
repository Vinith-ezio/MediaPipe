import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Configuration
# ============================================================

PHONE_CAMERA_URL = "http://192.168.1.110:8080/video"

MODEL_PATH = "models/pose_landmarker_lite.task"

MAX_POSES = 1


# ============================================================
# Pose Landmark Names
# MediaPipe Pose contains 33 landmarks
# ============================================================

LANDMARK_NAMES = {
    0: "Nose",

    1: "Left Eye Inner",
    2: "Left Eye",
    3: "Left Eye Outer",

    4: "Right Eye Inner",
    5: "Right Eye",
    6: "Right Eye Outer",

    7: "Left Ear",
    8: "Right Ear",

    9: "Mouth Left",
    10: "Mouth Right",

    11: "Left Shoulder",
    12: "Right Shoulder",

    13: "Left Elbow",
    14: "Right Elbow",

    15: "Left Wrist",
    16: "Right Wrist",

    17: "Left Pinky",
    18: "Right Pinky",

    19: "Left Index",
    20: "Right Index",

    21: "Left Thumb",
    22: "Right Thumb",

    23: "Left Hip",
    24: "Right Hip",

    25: "Left Knee",
    26: "Right Knee",

    27: "Left Ankle",
    28: "Right Ankle",

    29: "Left Heel",
    30: "Right Heel",

    31: "Left Foot Index",
    32: "Right Foot Index",
}


# ============================================================
# Pose Skeleton Connections
#
# Each tuple represents:
#
# (start_landmark, end_landmark)
# ============================================================

POSE_CONNECTIONS = [

    # Face
    (0, 1),
    (1, 2),
    (2, 3),

    (0, 4),
    (4, 5),
    (5, 6),

    (3, 7),
    (6, 8),

    (9, 10),

    # Upper body
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Left hand
    (15, 17),
    (15, 19),
    (15, 21),

    # Right hand
    (16, 18),
    (16, 20),
    (16, 22),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28),

    # Left foot
    (27, 29),
    (27, 31),

    # Right foot
    (28, 30),
    (28, 32),
]


# ============================================================
# Function 1
# Initialize MediaPipe Pose Landmarker
# ============================================================

def initialize_pose_landmarker():

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,

        running_mode=vision.RunningMode.IMAGE,

        num_poses=MAX_POSES
    )

    landmarker = vision.PoseLandmarker.create_from_options(
        options
    )

    return landmarker


# ============================================================
# Function 2
# Open Camera
# ============================================================

def open_camera():

    cap = cv2.VideoCapture(
        PHONE_CAMERA_URL
    )

    if not cap.isOpened():

        print("ERROR: Could not open phone camera.")

        return None

    print("Phone camera connected.")

    return cap


# ============================================================
# Function 3
# Process OpenCV Frame
#
# Converts BGR → RGB and creates MediaPipe Image
# ============================================================

def process_frame(frame):

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
# Function 4
# Detect Pose
# ============================================================

def detect_pose(landmarker, mp_image):

    result = landmarker.detect(
        mp_image
    )

    return result


# ============================================================
# Function 5
# Get Pixel Coordinates
# ============================================================

def get_landmark_coordinates(
    landmark,
    frame_width,
    frame_height
):

    x = int(
        landmark.x * frame_width
    )

    y = int(
        landmark.y * frame_height
    )

    return x, y


# ============================================================
# Function 6
# Get Landmark By Name
# ============================================================

def get_landmark_by_name(
    pose_landmarks,
    name
):

    for index, landmark_name in LANDMARK_NAMES.items():

        if landmark_name == name:

            if index < len(pose_landmarks):

                return pose_landmarks[index]

    return None


# ============================================================
# Function 7
# Print All 33 Landmarks
# ============================================================

def print_pose_landmarks(
    pose_landmarks
):

    print()
    print("=" * 60)
    print("POSE LANDMARKS")
    print("=" * 60)

    for index, landmark in enumerate(
        pose_landmarks
    ):

        name = LANDMARK_NAMES.get(
            index,
            "Unknown"
        )

        print(
            f"{index:02d} | "
            f"{name:<20} | "
            f"x={landmark.x:.4f} | "
            f"y={landmark.y:.4f} | "
            f"z={landmark.z:.4f}"
        )

    print("=" * 60)
    print()


# ============================================================
# Function 8
# Draw Pose Landmarks
# ============================================================

def draw_pose_landmarks(
    frame,
    pose_landmarks
):

    frame_height, frame_width, _ = frame.shape

    for index, landmark in enumerate(
        pose_landmarks
    ):

        x, y = get_landmark_coordinates(
            landmark,
            frame_width,
            frame_height
        )

        # Draw landmark
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

        # Draw landmark number
        cv2.putText(
            frame,
            str(index),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

    return frame


# ============================================================
# Function 9
# Draw Pose Skeleton Connections
# ============================================================

def draw_pose_connections(
    frame,
    pose_landmarks
):

    frame_height, frame_width, _ = frame.shape

    for start_index, end_index in POSE_CONNECTIONS:

        if (
            start_index >= len(pose_landmarks)
            or
            end_index >= len(pose_landmarks)
        ):
            continue

        start_point = get_landmark_coordinates(
            pose_landmarks[start_index],
            frame_width,
            frame_height
        )

        end_point = get_landmark_coordinates(
            pose_landmarks[end_index],
            frame_width,
            frame_height
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (255, 0, 0),
            2
        )

    return frame


# ============================================================
# Function 10
# Calculate Euclidean Distance
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# Function 11
# Calculate Angle
#
# Angle ABC
#
#       A
#        \
#         \
#          B
#           \
#            \
#             C
#
# Angle is measured at B
# ============================================================

def calculate_angle(a, b, c):

    ba = (
        a[0] - b[0],
        a[1] - b[1]
    )

    bc = (
        c[0] - b[0],
        c[1] - b[1]
    )

    dot_product = (
        ba[0] * bc[0]
        +
        ba[1] * bc[1]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:

        return 0

    cosine_angle = dot_product / (
        magnitude_ba * magnitude_bc
    )

    cosine_angle = max(
        -1,
        min(1, cosine_angle)
    )

    angle = math.degrees(
        math.acos(cosine_angle)
    )

    return angle


# ============================================================
# Function 12
# Calculate FPS
# ============================================================

def calculate_fps(previous_time):

    current_time = time.time()

    elapsed_time = (
        current_time - previous_time
    )

    if elapsed_time <= 0:

        return 0, current_time

    fps = 1 / elapsed_time

    return fps, current_time


# ============================================================
# Function 13
# Display Information
# ============================================================

def display_information(
    frame,
    fps,
    pose_count
):

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Poses: {pose_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q: Quit | P: Print Landmarks",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# Function 14
# Cleanup
# ============================================================

def cleanup(
    cap,
    landmarker
):

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()

    print()
    print("Pose detection stopped.")
    print("Resources released.")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MediaPipe Pose Landmarker Practice")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Initialize Pose Landmarker
    # --------------------------------------------------------

    print("Loading Pose Landmarker...")

    landmarker = initialize_pose_landmarker()

    print("Pose Landmarker loaded.")


    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = open_camera()

    if cap is None:

        landmarker.close()

        return


    print()
    print("Controls:")
    print("Q → Quit")
    print("P → Print all 33 landmarks")
    print()


    # --------------------------------------------------------
    # FPS initialization
    # --------------------------------------------------------

    previous_time = time.time()


    # ========================================================
    # Main Camera Loop
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Read frame
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break


        # ----------------------------------------------------
        # Get frame dimensions
        # ----------------------------------------------------

        frame_height, frame_width, _ = (
            frame.shape
        )


        # ----------------------------------------------------
        # Convert frame for MediaPipe
        # ----------------------------------------------------

        mp_image = process_frame(
            frame
        )


        # ----------------------------------------------------
        # Run pose detection
        # ----------------------------------------------------

        result = detect_pose(
            landmarker,
            mp_image
        )


        # ----------------------------------------------------
        # Process detected poses
        # ----------------------------------------------------

        pose_count = len(
            result.pose_landmarks
        )


        if result.pose_landmarks:

            for pose_landmarks in (
                result.pose_landmarks
            ):

                # --------------------------------------------
                # Draw skeleton
                # --------------------------------------------

                draw_pose_connections(
                    frame,
                    pose_landmarks
                )


                # --------------------------------------------
                # Draw 33 landmarks
                # --------------------------------------------

                draw_pose_landmarks(
                    frame,
                    pose_landmarks
                )


        # ----------------------------------------------------
        # Calculate FPS
        # ----------------------------------------------------

        fps, previous_time = calculate_fps(
            previous_time
        )


        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        display_information(
            frame,
            fps,
            pose_count
        )


        # ----------------------------------------------------
        # Show frame
        # ----------------------------------------------------

        cv2.imshow(
            "MediaPipe Pose Landmarks",
            frame
        )


        # ----------------------------------------------------
        # Keyboard input
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if key == ord("q"):

            break


        # ----------------------------------------------------
        # Print landmarks
        # ----------------------------------------------------

        if (
            key == ord("p")
            and result.pose_landmarks
        ):

            print_pose_landmarks(
                result.pose_landmarks[0]
            )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cleanup(
        cap,
        landmarker
    )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":

    main()