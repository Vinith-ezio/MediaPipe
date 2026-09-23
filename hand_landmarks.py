
import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

PHONE_CAMERA_URL = "http://192.168.1.110:8080/video"
MODEL_PATH = "models/hand_landmarker.task"

MAX_HANDS = 2


# ============================================================
# HAND LANDMARK CONNECTIONS
# ============================================================

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
# LANDMARK NAMES
# ============================================================

LANDMARK_NAMES = {
    0: "Wrist",

    1: "Thumb CMC",
    2: "Thumb MCP",
    3: "Thumb IP",
    4: "Thumb TIP",

    5: "Index MCP",
    6: "Index PIP",
    7: "Index DIP",
    8: "Index TIP",

    9: "Middle MCP",
    10: "Middle PIP",
    11: "Middle DIP",
    12: "Middle TIP",

    13: "Ring MCP",
    14: "Ring PIP",
    15: "Ring DIP",
    16: "Ring TIP",

    17: "Pinky MCP",
    18: "Pinky PIP",
    19: "Pinky DIP",
    20: "Pinky TIP",
}


# ============================================================
# FINGER LANDMARK INDICES
# ============================================================

FINGER_TIPS = {
    "Thumb": 4,
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20,
}


# ============================================================
# HELPER FUNCTIONS  ------- > Finger/Gesture Detection Logic 
# ============================================================

def distance(p1, p2):
    """
    Calculate Euclidean distance between two 2D points.
    """

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def calculate_angle(a, b, c):
    """
    Calculate angle ABC.

    a = first point
    b = middle point
    c = last point
    """

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


def detect_fingers(points):
    """
    Basic geometric finger-state detection.

    Returns:
        dictionary containing
        open/closed state of fingers.
    """

    fingers = {}

    # --------------------------------------------------------
    # Index finger
    # --------------------------------------------------------

    index_angle = calculate_angle(
        points[5],
        points[6],
        points[8]
    )

    fingers["Index"] = index_angle > 150


    # --------------------------------------------------------
    # Middle finger
    # --------------------------------------------------------

    middle_angle = calculate_angle(
        points[9],
        points[10],
        points[12]
    )

    fingers["Middle"] = middle_angle > 150


    # --------------------------------------------------------
    # Ring finger
    # --------------------------------------------------------

    ring_angle = calculate_angle(
        points[13],
        points[14],
        points[16]
    )

    fingers["Ring"] = ring_angle > 150


    # --------------------------------------------------------
    # Pinky
    # --------------------------------------------------------

    pinky_angle = calculate_angle(
        points[17],
        points[18],
        points[20]
    )

    fingers["Pinky"] = pinky_angle > 150


    # --------------------------------------------------------
    # Thumb
    #
    # For the first experiment we use distance between
    # thumb tip and index MCP.
    # This is a simple heuristic, not a universal
    # handedness-independent solution.
    # --------------------------------------------------------

    thumb_distance = distance(
        points[4],
        points[5]
    )

    index_distance = distance(
        points[3],
        points[5]
    )

    fingers["Thumb"] = (
        thumb_distance > index_distance
    )

    return fingers


def detect_gesture(fingers):
    """
    Detect a basic hand gesture from finger states.

    Returns:
        String containing the detected gesture.
    """

    # Open palm
    if all(fingers.values()):
        return "OPEN PALM"

    # Fist
    if not any(fingers.values()):
        return "FIST"

    # Pointing
    if (
        fingers["Index"]
        and not fingers["Middle"]
        and not fingers["Ring"]
        and not fingers["Pinky"]
    ):
        return "POINTING"

    # Peace
    if (
        fingers["Index"]
        and fingers["Middle"]
        and not fingers["Ring"]
        and not fingers["Pinky"]
    ):
        return "PEACE"

    # Thumb only
    if (
        fingers["Thumb"]
        and not fingers["Index"]
        and not fingers["Middle"]
        and not fingers["Ring"]
        and not fingers["Pinky"]
    ):
        return "THUMBS UP"

    return "UNKNOWN"

def perform_gesture_action(gesture):
    """
    Perform an action based on the detected gesture.
    """

    if gesture == "OPEN PALM":
        print("ACTION: START")

    elif gesture == "FIST":
        print("ACTION: STOP")

    elif gesture == "POINTING":
        print("ACTION: NEXT")

    elif gesture == "PEACE":
        print("ACTION: MODE 2")

    elif gesture == "THUMBS UP":
        print("ACTION: CONFIRMED")

    else:
        pass


def print_landmarks(hand_landmarks):
    """
    Print all 21 landmark coordinates.
    """

    print("\n" + "=" * 60)
    print("21 HAND LANDMARKS")
    print("=" * 60)

    for index, landmark in enumerate(hand_landmarks):

        name = LANDMARK_NAMES.get(
            index,
            "Unknown"
        )

        print(
            f"{index:2d} | "
            f"{name:12s} | "
            f"x={landmark.x:.4f} | "
            f"y={landmark.y:.4f} | "
            f"z={landmark.z:.4f}"
        )

    print("=" * 60)


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=MAX_HANDS
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(
    PHONE_CAMERA_URL
)

if not cap.isOpened():

    print(
        "ERROR: Could not connect to phone camera."
    )

    landmarker.close()
    exit()


print("=" * 60)
print("MediaPipe Hand Practice Started")
print("=" * 60)

print("Camera      : Connected")
print("Model       : Loaded")
print("Max Hands   :", MAX_HANDS)

print()
print("CONTROLS")
print("------------------------------------------------------------")
print("Q = Quit")
print("P = Print 21 landmark coordinates")
print("F = Print finger states")
print("A = Print handedness")
print("Z = Print Z/depth values")
print("C = Print coordinate conversion")
print("T = Run landmark distance test")
print("")


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Read frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Failed to receive frame."
        )

        break


    # --------------------------------------------------------
    # Frame dimensions
    # --------------------------------------------------------

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # NumPy → MediaPipe Image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # MediaPipe inference
    # --------------------------------------------------------

    result = landmarker.detect(
        mp_image
    )


    # ========================================================
    # PROCESS DETECTED HANDS
    # ========================================================

    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        points = []


        # ----------------------------------------------------
        # Convert normalized coordinates → pixels
        # ----------------------------------------------------

        for landmark in hand_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )


        # ----------------------------------------------------
        # Draw skeleton
        # ----------------------------------------------------

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        for index, (x, y) in enumerate(points):

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


            # Landmark number

            cv2.putText(
                frame,
                str(index),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )


        # ----------------------------------------------------
        # Finger detection
        # ----------------------------------------------------

        finger_states = detect_fingers(
            points
        )


        # ----------------------------------------------------
        # Count fingers
        # ----------------------------------------------------

        finger_count = sum(
            finger_states.values()
        )

        # ----------------------------------------------------
        # Gesture Recognition
        # ----------------------------------------------------

        gesture = detect_gesture(
            finger_states
        )

        perform_gesture_action(gesture)

        # ----------------------------------------------------
        # Display finger count
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Hand {hand_index + 1}: {finger_count} fingers",
            (20, 110 + hand_index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:

        fps = 1 / elapsed

    else:

        fps = 0

    previous_time = current_time


    # ========================================================
    # GENERAL INFORMATION
    # ========================================================

    hand_count = len(
        result.hand_landmarks
    )


    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "MediaPipe Hand Practice",
        frame
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Q = Quit
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # P = Print landmarks
    # --------------------------------------------------------

    elif key == ord("p"):

        for hand_landmarks in result.hand_landmarks:

            print_landmarks(
                hand_landmarks
            )


    # --------------------------------------------------------
    # F = Finger states
    # --------------------------------------------------------

    elif key == ord("f"):

        for hand_landmarks in result.hand_landmarks:

            points = []

            for landmark in hand_landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                points.append(
                    (x, y)
                )


            fingers = detect_fingers(
                points
            )


            print("\nFINGER STATES")
            print("-" * 40)

            for finger, state in fingers.items():

                status = (
                    "OPEN"
                    if state
                    else "CLOSED"
                )

                print(
                    f"{finger:8s}: {status}"
                )

            print(
                "Count:",
                sum(fingers.values())
            )


    # --------------------------------------------------------
    # A = Handedness
    # --------------------------------------------------------

    elif key == ord("a"):

        print("\nHANDEDNESS")
        print("-" * 40)

        for hand_index, handedness in enumerate(
            result.handedness
        ):

            if handedness:

                category = handedness[0]

                print(
                    f"Hand {hand_index + 1}: "
                    f"{category.category_name} "
                    f"({category.score:.3f})"
                )


    # --------------------------------------------------------
    # Z = Depth
    # --------------------------------------------------------

    elif key == ord("z"):

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            print(
                f"\nHAND {hand_index + 1} Z VALUES"
            )

            print("-" * 40)

            for index, landmark in enumerate(
                hand_landmarks
            ):

                print(
                    f"{index:2d}: "
                    f"z={landmark.z:.5f}"
                )


    # --------------------------------------------------------
    # C = Coordinate conversion
    # --------------------------------------------------------

    elif key == ord("c"):

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            print(
                f"\nHAND {hand_index + 1} "
                "COORDINATE CONVERSION"
            )

            print("-" * 60)

            for index, landmark in enumerate(
                hand_landmarks
            ):

                pixel_x = int(
                    landmark.x * width
                )

                pixel_y = int(
                    landmark.y * height
                )

                print(
                    f"{index:2d}: "
                    f"normalized=({landmark.x:.3f}, "
                    f"{landmark.y:.3f}) "
                    f"→ pixel=({pixel_x}, "
                    f"{pixel_y})"
                )


    # --------------------------------------------------------
    # T = Distance test
    # --------------------------------------------------------

    elif key == ord("t"):

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            points = []

            for landmark in hand_landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                points.append(
                    (x, y)
                )


            wrist = points[0]
            index_tip = points[8]
            middle_tip = points[12]

            wrist_index_distance = distance(
                wrist,
                index_tip
            )

            wrist_middle_distance = distance(
                wrist,
                middle_tip
            )


            print(
                f"\nHAND {hand_index + 1}"
            )

            print(
                f"Wrist → Index TIP: "
                f"{wrist_index_distance:.2f} px"
            )

            print(
                f"Wrist → Middle TIP: "
                f"{wrist_middle_distance:.2f} px"
            )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print()
print("MediaPipe Hand Practice stopped.")

