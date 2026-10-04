import sys
import math

import cv2
import cvzone
from ultralytics import YOLO

# Video file path, or a camera index such as 0 (pass on the command line)
source = sys.argv[1] if len(sys.argv) > 1 else "Media/garbage.mp4"
cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
if not cap.isOpened():
    sys.exit(f"Could not open video source: {source}")

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Define class names
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

try:
    while True:
        success, img = cap.read()
        if not success:
            break  # end of video or camera disconnected

        results = model(img, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf > 0.3:
                    cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
