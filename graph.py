import cv2
import math
import cvzone
from ultralytics import YOLO
import matplotlib.pyplot as plt
import collections
import os

# ----------------------------
# Load YOLO model and class labels
# ----------------------------
yolo_model = YOLO("Weights/best.pt")
class_labels = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# ----------------------------
# Folder containing garbage images
# ----------------------------
image_folder = "Media/"   # put all your garbage images here
detections = []           # to store results for ALL images

# ----------------------------
# Process each image in folder
# ----------------------------
for file in os.listdir(image_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(image_folder, file)
        img = cv2.imread(img_path)

        results = yolo_model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > 0.3:  # filter low confidence
                    detections.append({
                        "class": class_labels[cls],
                        "conf": conf,
                        "width": w,
                        "height": h,
                        "cx": x1 + w // 2,
                        "cy": y1 + h // 2,
                        "image": file
                    })

        # OPTIONAL: show detections one by one
        # cv2.imshow("Detections", img)
        # cv2.waitKey(300)  # show for 300 ms

cv2.destroyAllWindows()

# ----------------------------
# PLOTS FOR WHOLE DATASET
# ----------------------------
if detections:
    # 1. Confidence Histogram
    plt.figure(figsize=(6, 4))
    plt.hist([d["conf"] for d in detections], bins=10, edgecolor='black')
    plt.title("Confidence Distribution (All Images)")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.show()

    # 2. Scatter Plot: Width vs Height
    plt.figure(figsize=(6, 4))
    plt.scatter([d["width"] for d in detections],
                [d["height"] for d in detections],
                alpha=0.5)
    plt.title("Bounding Box Width vs Height")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.show()

    # 3. Bar Chart: Class Distribution
    counter = collections.Counter(d["class"] for d in detections)
    plt.figure(figsize=(6, 4))
    plt.bar(counter.keys(), counter.values())
    plt.title("Class Distribution (All Images)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # 4. Scatter Plot: Object centers (combined across dataset)
    plt.figure(figsize=(6, 4))
    plt.scatter([d["cx"] for d in detections],
                [d["cy"] for d in detections],
                c='red', s=15, alpha=0.5)
    plt.title("Object Centers (All Images)")
    plt.xlabel("X position (px)")
    plt.ylabel("Y position (px)")
    plt.gca().invert_yaxis()  # match image coordinates
    plt.show()

else:
    print("⚠️ No objects detected in dataset.")
