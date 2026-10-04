import sys
import math

import cv2
import cvzone
from ultralytics import YOLO

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Define class names
class_labels = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# Load the image (pass a path on the command line, or use the sample image)
image_path = sys.argv[1] if len(sys.argv) > 1 else "Media/garbage_6.jpg"
img = cv2.imread(image_path)
if img is None:
    sys.exit(f"Could not read image: {image_path}")

# Perform object detection
results = yolo_model(img)

# Loop through the detections and draw bounding boxes
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.3:
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (max(0, x1), max(20, y1 - 10)),
                               scale=0.8, thickness=1, colorR=(255, 0, 0))

# Display the image with detections; press 'q' (or close the window) to quit
cv2.imshow("Image", img)
while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
