import os
import cv2
# Base directory for building absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Paths ----------------
WANTED_DIR = os.path.join(BASE_DIR, "wanted")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
FACE_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "faces")
OBJECT_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "objects")

# ---------------- Camera & YOLO ----------------
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_DSHOW  # This will be set based on troubleshooting
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

YOLO_WEIGHTS = "yolov8n.pt"
YOLO_CONF = 0.5
YOLO_IMGSZ = 320

# ---------------- Alarm & snapshots ----------------
ALARM_COOLDOWN_S = 5.0
SNAPSHOT_COOLDOWN_S = 2.0

# ---------------- Face Recognition ----------------
FACE_TOLERANCE = 0.6

# ---------------- MongoDB ----------------
MONGO_ENABLED = True
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "facedetection"
MONGO_COLLECTION = "detections"