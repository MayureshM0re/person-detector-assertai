import cv2
import torch
from ultralytics import YOLO
import os

# -----------------------------
# Paths 
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(BASE_DIR, "person_detector.pt")
video_path = os.path.join(BASE_DIR, "video.mp4")
output_path = os.path.join(BASE_DIR, "output_video.mp4")

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# Load video
# -----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Could not open video at {video_path}")
    exit()

# -----------------------------
# Video properties
# -----------------------------
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

confidence_threshold = 0.40

# -----------------------------
# Process frames
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    results = model(frame)

    # Draw detections
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"Person: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)

    # Show live window 
    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved at {output_path}")
