import cv2
import os

OUTPUT_PATH = "selfie_face.jpg"

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

print("📸 Webcam opened")
print("Press SPACE to capture selfie")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Selfie Capture", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("Capture cancelled")
        break

    elif key == 32:  # SPACE
        cv2.imwrite(OUTPUT_PATH, frame)
        print("Selfie captured and saved as", OUTPUT_PATH)
        break

cap.release()
cv2.destroyAllWindows()
