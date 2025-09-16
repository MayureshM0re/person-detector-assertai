# Person Detector — README

Overview
- Simple video person detector using a YOLOv8 model in [main.py](main.py).
- Reads a video, runs detections, draws boxes, and writes an output video.

Files
- [main.py](main.py) — main script with configuration and processing loop.
- [person_detector.pt](person_detector.pt) — YOLO model used by the script.
- [video.mp4](video.mp4) — input video used by default.
- [output_video.mp4](output_video.mp4) — default output file produced.

Requirements
- Python 3.10+
- Packages:
  - ultralytics
  - torch
  - opencv-python

Install
```sh
pip install ultralytics torch opencv-python