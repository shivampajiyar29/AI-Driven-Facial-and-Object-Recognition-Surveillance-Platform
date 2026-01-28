# import os
# import time
# import threading
# import cv2
# import numpy as np
# import face_recognition
# from ultralytics import YOLO
# from PIL import Image

# # Optional alarm backends
# try:
#     from playsound import playsound
# except Exception:
#     playsound = None

# try:
#     import winsound  # Windows beep fallback
# except Exception:
#     winsound = None

# # ----------------------
# # Configuration
# # ----------------------
# WANTED_DIR = "wanted"
# CAPTURES_DIR = "captures"
# FACE_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "faces")
# OBJECT_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "objects")
# ALARM_FILE = "alarm.mp3"
# ALARM_COOLDOWN_S = 5.0
# SNAPSHOT_COOLDOWN_S = 5.0
# CAMERA_WIDTH = 1280
# CAMERA_HEIGHT = 720
# YOLO_CONF = 0.4  # confidence threshold for object detection

# def ensure_dir(path: str):
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)

# # Prepare output folders
# ensure_dir(CAPTURES_DIR)
# ensure_dir(FACE_CAPTURES_DIR)
# ensure_dir(OBJECT_CAPTURES_DIR)

# # ----------------------
# # Load YOLO model (objects)
# # ----------------------
# yolo_model = YOLO("yolov8n.pt")  # auto-downloads if missing

# # ----------------------
# # Load known faces (wanted)
# # ----------------------
# known_faces = []
# known_names = []

# if not os.path.exists(WANTED_DIR):
#     print(f"[ERROR] Wanted folder '{WANTED_DIR}' not found! Create it and add images.")
# else:
#     for filename in os.listdir(WANTED_DIR):
#         if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             image_path = os.path.join(WANTED_DIR, filename)
#             try:
#                 pil_img = Image.open(image_path).convert("RGB")
#                 image = np.array(pil_img)
#                 encodings = face_recognition.face_encodings(image)
#                 if encodings:
#                     known_faces.append(encodings[0])
#                     known_names.append(os.path.splitext(filename)[0])
#                     print(f"[INFO] Loaded face from {filename}")
#                 else:
#                     print(f"[WARNING] No face found in {filename}, skipping.")
#             except Exception as e:
#                 print(f"[ERROR] Could not load {filename}: {e}")

# if not known_faces:
#     print("[WARNING] No wanted faces loaded. Face recognition will label everyone as 'Unknown'.")

# # ----------------------
# # Camera initialization with fallback indices
# # ----------------------
# def init_camera(indices=(0, 1, 2, 3)):
#     for idx in indices:
#         cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
#         if not cap.isOpened():
#             cap.release()
#             continue
#         # Set preferred resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
#         # Test a frame
#         ok, test = cap.read()
#         if not ok or test is None or test.size == 0:
#             cap.release()
#             continue
#         print(f"[INFO] Using camera index {idx} at ~{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
#         return cap, idx
#     return None, None

# cap, cam_index = init_camera()
# if cap is None:
#     print("[ERROR] Could not access any webcam (tried indices 0-3). Check camera and drivers.")
#     raise SystemExit(1)

# print("[INFO] Press 'Q' to quit the program.")

# last_alarm_time = 0.0
# last_saved_time = {}  # key: label or person name -> last timestamp saved

# def now():
#     return time.time()

# def should_cooldown(key: str, cooldown_s: float) -> bool:
#     t = last_saved_time.get(key, 0.0)
#     if now() - t >= cooldown_s:
#         last_saved_time[key] = now()
#         return True
#     return False

# def trigger_alarm():
#     global last_alarm_time
#     if now() - last_alarm_time < ALARM_COOLDOWN_S:
#         return
#     last_alarm_time = now()

#     if playsound and os.path.exists(ALARM_FILE):
#         def _play():
#             try:
#                 playsound(ALARM_FILE)
#             except Exception as e:
#                 print(f"[WARNING] Alarm playback failed: {e}")
#                 if winsound:
#                     try:
#                         winsound.Beep(2500, 500)
#                     except Exception:
#                         pass
#         threading.Thread(target=_play, daemon=True).start()
#     elif winsound:
#         try:
#             winsound.Beep(2500, 500)
#         except Exception:
#             pass
#     else:
#         print("[INFO] Alarm triggered (no audio backend available)")

# def save_snapshot(dir_path: str, prefix: str, frame):
#     ensure_dir(dir_path)
#     ts = time.strftime("%Y%m%d-%H%M%S")
#     filename = f"{prefix}_{ts}.jpg"
#     out_path = os.path.join(dir_path, filename)
#     try:
#         cv2.imwrite(out_path, frame)
#         print(f"[INFO] Saved snapshot: {out_path}")
#     except Exception as e:
#         print(f"[WARNING] Failed to save snapshot {out_path}: {e}")

# # ----------------------
# # Main loop
# # ----------------------
# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None or frame.size == 0:
#         print("[WARNING] Failed to grab a valid frame from webcam.")
#         continue

#     # FACE DETECTION & RECOGNITION
#     try:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     except Exception as e:
#         print(f"[ERROR] Could not convert frame to RGB: {e}")
#         continue

#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

#     for face_enc, (top, right, bottom, left) in zip(face_encs, face_locations):
#         name = "Unknown"
#         if known_faces:
#             matches = face_recognition.compare_faces(known_faces, face_enc)
#             if True in matches:
#                 distances = face_recognition.face_distance(known_faces, face_enc)
#                 best_idx = int(np.argmin(distances))
#                 if matches[best_idx]:
#                     name = known_names[best_idx]
#                     # Mark wanted on screen
#                     cv2.putText(frame, f"WANTED: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     trigger_alarm()
#                     # Save snapshot with cooldown per person
#                     if should_cooldown(f"face:{name}", SNAPSHOT_COOLDOWN_S):
#                         save_snapshot(FACE_CAPTURES_DIR, f"{name}", frame)
#                     # Optional: show details from wanted/<name>.txt
#                     details_file = os.path.join(WANTED_DIR, f"{name}.txt")
#                     if os.path.exists(details_file):
#                         try:
#                             with open(details_file, "r", encoding="utf-8", errors="ignore") as f:
#                                 details = f.read().strip().splitlines()[:3]
#                             for i, line in enumerate(details):
#                                 cv2.putText(frame, line, (50, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                         except Exception:
#                             pass

#         # Draw face box and label
#         cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
#         cv2.putText(frame, name, (left, max(20, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     # OBJECT DETECTION (YOLO)
#     results = yolo_model(frame, conf=YOLO_CONF, stream=True, verbose=False)
#     for result in results:
#         names = getattr(result, "names", None) or getattr(yolo_model, "names", {})
#         boxes = getattr(result, "boxes", None)
#         if boxes is None:
#             continue
#         for box in boxes:
#             try:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 cls_id = int(box.cls[0]) if box.cls is not None else -1
#                 conf = float(box.conf[0]) if box.conf is not None else 0.0
#                 label = names.get(cls_id, str(cls_id))
#             except Exception:
#                 continue

#             # Draw object box and label with confidence
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Save snapshot per class with cooldown
#             key = f"obj:{label}"
#             if should_cooldown(key, SNAPSHOT_COOLDOWN_S):
#                 save_snapshot(OBJECT_CAPTURES_DIR, label, frame)

#     # Show the video
#     cv2.imshow("Face & Object Detection", frame)

#     # Exit on 'Q' key
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()











# import os
# import time
# import threading
# import cv2
# import numpy as np
# import face_recognition
# from ultralytics import YOLO
# from PIL import Image

# # Optional alarm backends
# try:
#     from playsound import playsound
# except Exception:
#     playsound = None

# try:
#     import winsound  # Windows beep fallback
# except Exception:
#     winsound = None

# # ----------------------
# # Configuration
# # ----------------------
# WANTED_DIR = "wanted"   # Put multiple wanted images here: person1.jpg, person2.png, etc.
# CAPTURES_DIR = "captures"
# FACE_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "faces")
# OBJECT_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "objects")
# ALARM_FILE = "alarm.mp3"
# ALARM_COOLDOWN_S = 5.0
# SNAPSHOT_COOLDOWN_S = 5.0
# CAMERA_WIDTH = 1280
# CAMERA_HEIGHT = 720
# YOLO_CONF = 0.4  # confidence threshold for object detection

# def ensure_dir(path: str):
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)

# # Prepare output folders
# ensure_dir(CAPTURES_DIR)
# ensure_dir(FACE_CAPTURES_DIR)
# ensure_dir(OBJECT_CAPTURES_DIR)

# # ----------------------
# # Load YOLO model (objects)
# # ----------------------
# yolo_model = YOLO("yolov8n.pt")  # auto-downloads if missing

# # ----------------------
# # Load known faces (wanted)
# # ----------------------
# known_faces = []
# known_names = []

# if not os.path.exists(WANTED_DIR):
#     print(f"[ERROR] Wanted folder '{WANTED_DIR}' not found! Create it and add images.")
# else:
#     for filename in os.listdir(WANTED_DIR):
#         if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             image_path = os.path.join(WANTED_DIR, filename)
#             try:
#                 pil_img = Image.open(image_path).convert("RGB")
#                 image = np.array(pil_img)
#                 encodings = face_recognition.face_encodings(image)
#                 if encodings:
#                     known_faces.append(encodings[0])
#                     # Store the person's name (without extension)
#                     known_names.append(os.path.splitext(filename)[0])
#                     print(f"[INFO] Loaded wanted face: {filename}")
#                 else:
#                     print(f"[WARNING] No face found in {filename}, skipping.")
#             except Exception as e:
#                 print(f"[ERROR] Could not load {filename}: {e}")

# if not known_faces:
#     print("[WARNING] No wanted faces loaded. Everyone will be 'Unknown'.")

# # ----------------------
# # Camera initialization
# # ----------------------
# def init_camera(indices=(0, 1, 2, 3)):
#     for idx in indices:
#         cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
#         if not cap.isOpened():
#             cap.release()
#             continue
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
#         ok, test = cap.read()
#         if not ok or test is None or test.size == 0:
#             cap.release()
#             continue
#         print(f"[INFO] Using camera index {idx}")
#         return cap, idx
#     return None, None

# cap, cam_index = init_camera()
# if cap is None:
#     print("[ERROR] No webcam found (indices 0-3).")
#     raise SystemExit(1)

# print("[INFO] Press 'Q' to quit.")

# last_alarm_time = 0.0
# last_saved_time = {}

# def now():
#     return time.time()

# def should_cooldown(key: str, cooldown_s: float) -> bool:
#     t = last_saved_time.get(key, 0.0)
#     if now() - t >= cooldown_s:
#         last_saved_time[key] = now()
#         return True
#     return False

# def trigger_alarm():
#     global last_alarm_time
#     if now() - last_alarm_time < ALARM_COOLDOWN_S:
#         return
#     last_alarm_time = now()

#     if playsound and os.path.exists(ALARM_FILE):
#         def _play():
#             try:
#                 playsound(ALARM_FILE)
#             except Exception as e:
#                 print(f"[WARNING] Alarm playback failed: {e}")
#                 if winsound:
#                     try:
#                         winsound.Beep(2500, 500)
#                     except Exception:
#                         pass
#         threading.Thread(target=_play, daemon=True).start()
#     elif winsound:
#         try:
#             winsound.Beep(2500, 500)
#         except Exception:
#             pass
#     else:
#         print("[INFO] Alarm triggered (no audio backend available)")

# def save_snapshot(dir_path: str, prefix: str, frame):
#     ensure_dir(dir_path)
#     ts = time.strftime("%Y%m%d-%H%M%S")
#     filename = f"{prefix}_{ts}.jpg"
#     out_path = os.path.join(dir_path, filename)
#     try:
#         cv2.imwrite(out_path, frame)
#         print(f"[INFO] Saved snapshot: {out_path}")
#     except Exception as e:
#         print(f"[WARNING] Failed to save snapshot {out_path}: {e}")

# # ----------------------
# # Main loop
# # ----------------------
# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None or frame.size == 0:
#         print("[WARNING] Failed to grab frame.")
#         continue

#     # Face recognition
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

#     for face_enc, (top, right, bottom, left) in zip(face_encs, face_locations):
#         name = "Unknown"
#         if known_faces:
#             matches = face_recognition.compare_faces(known_faces, face_enc)
#             if True in matches:
#                 distances = face_recognition.face_distance(known_faces, face_enc)
#                 best_idx = int(np.argmin(distances))
#                 if matches[best_idx]:
#                     name = known_names[best_idx]
#                     # Mark WANTED
#                     cv2.putText(frame, f"WANTED: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     trigger_alarm()
#                     if should_cooldown(f"face:{name}", SNAPSHOT_COOLDOWN_S):
#                         save_snapshot(FACE_CAPTURES_DIR, f"{name}", frame)

#         cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
#         cv2.putText(frame, name, (left, max(20, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     # YOLO object detection
#     results = yolo_model(frame, conf=YOLO_CONF, stream=True, verbose=False)
#     for result in results:
#         names = getattr(result, "names", None) or getattr(yolo_model, "names", {})
#         boxes = getattr(result, "boxes", None)
#         if boxes is None:
#             continue
#         for box in boxes:
#             try:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 cls_id = int(box.cls[0]) if box.cls is not None else -1
#                 conf = float(box.conf[0]) if box.conf is not None else 0.0
#                 label = names.get(cls_id, str(cls_id))
#             except Exception:
#                 continue

#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(20, int(y1) - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             if should_cooldown(f"obj:{label}", SNAPSHOT_COOLDOWN_S):
#                 save_snapshot(OBJECT_CAPTURES_DIR, label, frame)

#     cv2.imshow("Face & Object Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()













import os
import time
import threading
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
from PIL import Image
from pymongo import MongoClient
import yagmail
import traceback

# ---------------- Configuration ----------------
WANTED_DIR = "wanted"
CAPTURES_DIR = "captures"
FACE_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "faces")
OBJECT_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "objects")

ALARM_COOLDOWN_S = 5.0
SNAPSHOT_COOLDOWN_S = 5.0
YOLO_CONF = 0.4
YOLO_IMGSZ = 320

# Email configuration
EMAIL_ENABLED = False
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_APP_PASSWORD = "your_app_password"
EMAIL_RECIPIENT = "recipient_email@gmail.com"

# MongoDB configuration
MONGO_ENABLED = True
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "facedetection"
MONGO_COLLECTION = "wanted"


# ---------------- Setup directories ----------------
for path in (CAPTURES_DIR, FACE_CAPTURES_DIR, OBJECT_CAPTURES_DIR):
    os.makedirs(path, exist_ok=True)

# ---------------- Optional audio ----------------
try:
    from playsound import playsound
except Exception:
    playsound = None

try:
    import winsound
except Exception:
    winsound = None

def trigger_alarm(file=None):
    if file and os.path.exists(file) and playsound:
        threading.Thread(target=lambda: playsound(file), daemon=True).start()
    elif playsound and os.path.exists("alarm.mp3"):
        threading.Thread(target=lambda: playsound("alarm.mp3"), daemon=True).start()
    elif winsound:
        try:
            winsound.Beep(2500, 500)
        except Exception:
            pass
    else:
        print("[ALARM] WANTED detected!")

# ---------------- Optional Mongo ----------------
if MONGO_ENABLED:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[MONGO_DB]
        detections_collection = db[MONGO_COLLECTION]
        mongo_client.server_info()
        print("[MONGO] Connected.")
    except Exception as e:
        print("[MONGO] Connection failed, disabling Mongo logging:", e)
        MONGO_ENABLED = False
else:
    detections_collection = None

# ---------------- Optional email ----------------
if EMAIL_ENABLED:
    try:
        yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        print("[EMAIL] Email configured.")
    except Exception as e:
        print("[EMAIL] Failed to configure email client, disabling email alerts:", e)
        EMAIL_ENABLED = False

# ---------------- Load YOLO model ----------------
try:
    yolo_person_object = YOLO("yolov8n.pt")
    print("[YOLO] Model loaded.")
except Exception as e:
    print("[YOLO] Failed to load YOLO model:", e)
    yolo_person_object = None

# ---------------- Load known faces ----------------
known_faces = []
known_names = []
if not os.path.exists(WANTED_DIR):
    print(f"[WARN] Wanted directory '{WANTED_DIR}' does not exist.")
else:
    for f in os.listdir(WANTED_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img = np.array(Image.open(os.path.join(WANTED_DIR, f)).convert("RGB"))
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_faces.append(enc[0])
                    known_names.append(os.path.splitext(f)[0])
                    print(f"[INFO] Loaded WANTED face: {f}")
                else:
                    print(f"[WARN] No face found in {f}, skipping.")
            except Exception as e:
                print(f"[WARN] Failed to load {f}: {e}")

# ---------------- Threaded Video Stream ----------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera (checked default and CAP_DSHOW)")
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# ---------------- Snapshot Utility ----------------
last_saved = {}
def save_snapshot(folder, prefix, frame):
    ts = time.strftime("%Y%m%d-%H%M%S")
    key = prefix
    if last_saved.get(key, 0) + SNAPSHOT_COOLDOWN_S > time.time():
        return None
    last_saved[key] = time.time()
    filename = f"{prefix}_{ts}.jpg"
    path = os.path.join(folder, filename)
    try:
        cv2.imwrite(path, frame)
        print(f"[CAPTURED] {path}")
        return path
    except Exception as e:
        print("[CAPTURE] Failed to write image:", e)
        return None

# ---------------- Detection Thread ----------------
class DetectionThread:
    def __init__(self):
        self.faces = []
        self.objects = []
        self.lock = threading.Lock()
        self.last_alarm = 0
        self.alert_active = False
        self.flash_state = True
        self.wanted_count = 0
        self.object_count = 0

    def detect(self, frame):
        try:
            small_frame = cv2.resize(frame, (640, 360))
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("[DETECT] Frame preprocessing failed:", e)
            return

        faces_detected = []
        objects_detected = []
        wanted_count = 0
        self.alert_active = False

        # --- Face Recognition ---
        try:
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
            for enc, (top, right, bottom, left) in zip(face_encodings, face_locations):
                name = "Unknown"
                if known_faces:
                    matches = face_recognition.compare_faces(known_faces, enc)
                    if True in matches:
                        dists = face_recognition.face_distance(known_faces, enc)
                        idx = int(np.argmin(dists))
                        if matches[idx]:
                            name = known_names[idx]
                            wanted_count += 1
                            self.alert_active = True

                            if time.time() - self.last_alarm > ALARM_COOLDOWN_S:
                                audio_file = os.path.join(WANTED_DIR, f"{name}.mp3")
                                trigger_alarm(audio_file if os.path.exists(audio_file) else None)
                                self.last_alarm = time.time()

                            snapshot_path = save_snapshot(FACE_CAPTURES_DIR, name, frame)
                            ts = time.strftime("%Y%m%d-%H%M%S")

                            if MONGO_ENABLED and detections_collection is not None:
                                try:
                                    detections_collection.insert_one({
                                        "timestamp": ts,
                                        "type": "wanted_face",
                                        "name": name,
                                        "snapshot": snapshot_path
                                    })
                                except Exception as e:
                                    print("[MONGO] Insert failed:", e)

                            if EMAIL_ENABLED:
                                subject = f"WANTED ALERT: {name} detected"
                                body = f"{name} detected at {ts}.\nSnapshot: {snapshot_path}"
                                try:
                                    yag.send(EMAIL_RECIPIENT, subject, body, attachments=[snapshot_path] if snapshot_path else None)
                                    print("[EMAIL] Notification sent.")
                                except Exception as e:
                                    print("[EMAIL] Failed to send:", e)

                h_ratio = frame.shape[0] / 360
                w_ratio = frame.shape[1] / 640
                faces_detected.append((int(top * h_ratio), int(right * w_ratio),
                                       int(bottom * h_ratio), int(left * w_ratio), name))
        except Exception as e:
            print("[DETECT] Face recognition error:", e)

        # --- Object Detection ---
        if yolo_person_object is not None:
            try:
                results = yolo_person_object(small_frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, stream=True, verbose=False)
                for r in results:
                    boxes = getattr(r, "boxes", None)
                    if boxes is None:
                        continue
                    for b in boxes:
                        try:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            cls_id = int(b.cls[0])
                            label = yolo_person_object.names.get(cls_id, f"cls_{cls_id}")
                            x1 = int(x1 * frame.shape[1] / 640)
                            y1 = int(y1 * frame.shape[0] / 360)
                            x2 = int(x2 * frame.shape[1] / 640)
                            y2 = int(y2 * frame.shape[0] / 360)
                            objects_detected.append((x1, y1, x2, y2, label))
                            save_snapshot(OBJECT_CAPTURES_DIR, label, frame)
                        except Exception:
                            continue
            except Exception as e:
                print("[DETECT] YOLO object detection error:", e)

        with self.lock:
            self.faces = faces_detected
            self.objects = objects_detected
            self.flash_state = not self.flash_state
            self.wanted_count = wanted_count
            self.object_count = len(objects_detected)

# ---------------- Main ----------------
def main():
    try:
        stream = VideoStream(0)
    except Exception as e:
        print("[ERROR] Camera open failed:", e)
        return

    detector = DetectionThread()

    def detection_worker():
        while True:
            ret, frame = stream.read()
            if ret and frame is not None:
                detector.detect(frame)
            else:
                time.sleep(0.01)

    t = threading.Thread(target=detection_worker, daemon=True)
    t.start()

    window_name = "Smart Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            with detector.lock:
                for top, right, bottom, left, name in detector.faces:
                    color = (0, 0, 255) if name != "Unknown" and detector.flash_state else (255, 255, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, max(20, top - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                for x1, y1, x2, y2, label in detector.objects:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if detector.alert_active and detector.flash_state:
                    cv2.putText(frame, "!!! WANTED ALERT !!!", (50, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 5)

                cv2.putText(frame, f"WANTED: {detector.wanted_count}", (50, frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                cv2.putText(frame, f"Objects: {detector.object_count}", (50, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Exiting by user.")
    except Exception as e:
        print("[ERROR] Main loop exception:", e)
        traceback.print_exc()
    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
