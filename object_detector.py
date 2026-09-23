import cv2
import mediapipe as mp
import time

from collections import Counter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

PHONE_CAMERA_URL = "http://192.168.1.110:8080/video"

MODEL_PATH = "models/object_detector.tflite"

MAX_RESULTS = 5

SCORE_THRESHOLD = 0.5


# Set to None to display all detected classes.
#
# Example:
# TARGET_CLASSES = {"person", "bottle"}
#
# Then only these classes will be displayed.
TARGET_CLASSES = None


# ============================================================
# FUNCTION 1
# Initialize Object Detector
# ============================================================

def initialize_object_detector():

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        max_results=MAX_RESULTS,
        score_threshold=SCORE_THRESHOLD
    )

    detector = vision.ObjectDetector.create_from_options(
        options
    )

    return detector


# ============================================================
# FUNCTION 2
# Open Phone Camera
# ============================================================

def open_camera():

    cap = cv2.VideoCapture(
        PHONE_CAMERA_URL
    )

    if not cap.isOpened():

        print("ERROR: Could not connect to phone camera.")

        return None

    print("Phone camera connected.")

    return cap


# ============================================================
# FUNCTION 3
# Convert OpenCV Frame → MediaPipe Image
# ============================================================

def process_frame(frame):

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
# Run Object Detection
# ============================================================

def detect_objects(
    detector,
    mp_image
):

    result = detector.detect(
        mp_image
    )

    return result


# ============================================================
# FUNCTION 5
# Get Detection Information
# ============================================================

def get_detection_information(
    detection
):

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bounding_box = detection.bounding_box

    x = bounding_box.origin_x
    y = bounding_box.origin_y

    width = bounding_box.width
    height = bounding_box.height


    # --------------------------------------------------------
    # Category information
    # --------------------------------------------------------

    if detection.categories:

        category = detection.categories[0]

        category_name = category.category_name

        confidence = category.score

    else:

        category_name = "Unknown"

        confidence = 0.0


    return (
        category_name,
        confidence,
        x,
        y,
        width,
        height
    )


# ============================================================
# FUNCTION 6
# Check Class Filter
# ============================================================

def is_allowed_class(
    category_name
):

    # If no filter is configured,
    # allow every class.

    if TARGET_CLASSES is None:

        return True


    return (
        category_name.lower()
        in
        {
            name.lower()
            for name in TARGET_CLASSES
        }
    )


# ============================================================
# FUNCTION 7
# Draw Bounding Box
# ============================================================

def draw_detection(
    frame,
    category_name,
    confidence,
    x,
    y,
    width,
    height
):

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        f"{category_name}: "
        f"{confidence:.2f}"
    )


    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    text_width, text_height = text_size


    cv2.rectangle(
        frame,
        (x, max(0, y - text_height - 10)),
        (
            x + text_width + 10,
            y
        ),
        (0, 255, 0),
        -1
    )


    # --------------------------------------------------------
    # Label text
    # --------------------------------------------------------

    cv2.putText(
        frame,
        label,
        (
            x + 5,
            max(text_height, y - 5)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )


# ============================================================
# FUNCTION 8
# Count Objects
# ============================================================

def count_objects(
    detected_classes
):

    return Counter(
        detected_classes
    )


# ============================================================
# FUNCTION 9
# Draw Object Count
# ============================================================

def draw_object_counts(
    frame,
    object_counts
):

    y_position = 140

    cv2.putText(
        frame,
        "Detected:",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y_position += 30


    for class_name, count in object_counts.items():

        text = (
            f"{class_name}: {count}"
        )

        cv2.putText(
            frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y_position += 25


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
# Display Application Information
# ============================================================

def display_information(
    frame,
    fps,
    detection_count
):

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Detections: {detection_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q: Quit | P: Print Detections",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


# ============================================================
# FUNCTION 12
# Print Detection Details
# ============================================================

def print_detections(
    result
):

    print()
    print("=" * 70)
    print("OBJECT DETECTIONS")
    print("=" * 70)


    if not result.detections:

        print("No objects detected.")

        print("=" * 70)

        return


    for index, detection in enumerate(
        result.detections
    ):

        (
            category_name,
            confidence,
            x,
            y,
            width,
            height
        ) = get_detection_information(
            detection
        )


        print(
            f"{index + 1}. "
            f"Class: {category_name}"
        )

        print(
            f"   Confidence: "
            f"{confidence:.4f}"
        )

        print(
            f"   Bounding Box:"
        )

        print(
            f"      x      = {x}"
        )

        print(
            f"      y      = {y}"
        )

        print(
            f"      width  = {width}"
        )

        print(
            f"      height = {height}"
        )

        print()


    print("=" * 70)


# ============================================================
# FUNCTION 13
# Cleanup
# ============================================================

def cleanup(
    cap,
    detector
):

    cap.release()

    detector.close()

    cv2.destroyAllWindows()

    print()
    print("Camera released.")
    print("Object Detector closed.")
    print("Program finished.")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MediaPipe Object Detector Practice")
    print("=" * 70)
    print()


    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    print("Loading Object Detector...")

    detector = initialize_object_detector()

    print(
        "Object Detector loaded successfully."
    )


    # --------------------------------------------------------
    # Open phone camera
    # --------------------------------------------------------

    cap = open_camera()

    if cap is None:

        detector.close()

        return


    print()
    print("Object detection started.")
    print()
    print("Controls:")
    print("Q → Quit")
    print("P → Print current detections")
    print()


    # --------------------------------------------------------
    # FPS initialization
    # --------------------------------------------------------

    previous_time = time.time()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:


        # ----------------------------------------------------
        # Read camera frame
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

        mp_image = process_frame(
            frame
        )


        # ----------------------------------------------------
        # Run detection
        # ----------------------------------------------------

        result = detect_objects(
            detector,
            mp_image
        )


        # ----------------------------------------------------
        # Store detected classes
        # ----------------------------------------------------

        detected_classes = []


        # ----------------------------------------------------
        # Process each detection
        # ----------------------------------------------------

        for detection in result.detections:


            (
                category_name,
                confidence,
                x,
                y,
                width,
                height
            ) = get_detection_information(
                detection
            )


            # ------------------------------------------------
            # Apply class filter
            # ------------------------------------------------

            if not is_allowed_class(
                category_name
            ):

                continue


            # ------------------------------------------------
            # Store class
            # ------------------------------------------------

            detected_classes.append(
                category_name
            )


            # ------------------------------------------------
            # Draw detection
            # ------------------------------------------------

            draw_detection(
                frame,
                category_name,
                confidence,
                x,
                y,
                width,
                height
            )


        # ----------------------------------------------------
        # Count objects
        # ----------------------------------------------------

        object_counts = count_objects(
            detected_classes
        )


        # ----------------------------------------------------
        # Draw object counts
        # ----------------------------------------------------

        draw_object_counts(
            frame,
            object_counts
        )


        # ----------------------------------------------------
        # FPS
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
            len(detected_classes)
        )


        # ----------------------------------------------------
        # Display frame
        # ----------------------------------------------------

        cv2.imshow(
            "MediaPipe Object Detector",
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
        # Print detections
        # ----------------------------------------------------

        if (
            key == ord("p")
        ):

            print_detections(
                result
            )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cleanup(
        cap,
        detector
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()