import cv2

PHONE_CAMERA_URL = "http://192.168.1.110:8080/video"

cap = cv2.VideoCapture(PHONE_CAMERA_URL)

if not cap.isOpened():
    print("ERROR: Could not connect to phone camera.")
    exit()

print("Phone camera connected.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Failed to receive frame.")
        break

    cv2.imshow("Phone Camera - OpenCV", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()