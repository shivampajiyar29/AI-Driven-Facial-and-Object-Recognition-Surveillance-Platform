import os
import time
import threading
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
from PIL import Image
from pymongo import MongoClient
import traceback

# ---------------- Configuration ----------------
WANTED_DIR = "wanted"
CAPTURES_DIR = "captures"
FACE_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "faces")
OBJECT_CAPTURES_DIR = os.path.join(CAPTURES_DIR, "objects")

# Detection settings
FACE_TOLERANCE = 0.6  # Lower = more strict matching
ALARM_COOLDOWN_S = 5.0
SNAPSHOT_COOLDOWN_S = 3.0
YOLO_CONF = 0.5
YOLO_IMGSZ = 320

# MongoDB configuration
MONGO_ENABLED = True
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "facedetection"
MONGO_COLLECTION = "detections"

# ---------------- Setup directories ----------------
for path in (CAPTURES_DIR, FACE_CAPTURES_DIR, OBJECT_CAPTURES_DIR, WANTED_DIR):
    os.makedirs(path, exist_ok=True)

print("🚀 Starting Enhanced Face Detection System...")
print(f"📁 Wanted faces directory: {WANTED_DIR}")

# ---------------- Optional audio ----------------
try:
    from playsound import playsound
    AUDIO_AVAILABLE = True
    print("🔊 Audio: playsound available")
except Exception:
    playsound = None
    AUDIO_AVAILABLE = False
    print("🔇 Audio: playsound not available")

try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception:
    winsound = None
    WINSOUND_AVAILABLE = False

def trigger_alarm(file=None):
    """Improved alarm function"""
    print("🚨 🔴 🚨 WANTED PERSON DETECTED! 🚨 🔴 🚨")
    
    def play_alarm():
        audio_files_to_try = []
        
        # Try custom person audio first
        if file and os.path.exists(file):
            audio_files_to_try.append(file)
        
        # Try default alarm files
        audio_files_to_try.extend([
            "alarm.wav",
            "simple_alarm.wav", 
            "alarm.mp3"
        ])
        
        # Try each audio file
        for audio_file in audio_files_to_try:
            if os.path.exists(audio_file) and playsound:
                try:
                    print(f"🔊 Playing: {audio_file}")
                    playsound(audio_file)
                    return
                except Exception as e:
                    print(f"🔇 Failed to play {audio_file}: {e}")
        
        # Fallback to system beep
        if winsound:
            try:
                winsound.Beep(1000, 800)
                print("🔊 Played system beep")
            except Exception as e:
                print(f"🔇 System beep failed: {e}")
    
    threading.Thread(target=play_alarm, daemon=True).start()

# ---------------- MongoDB Setup ----------------
if MONGO_ENABLED:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[MONGO_DB]
        detections_collection = db[MONGO_COLLECTION]
        mongo_client.admin.command('ismaster')
        print("✅ MongoDB: Connected successfully")
    except Exception as e:
        print(f"❌ MongoDB: Connection failed - {e}")
        MONGO_ENABLED = False
        detections_collection = None
else:
    detections_collection = None
    print("📊 MongoDB: Disabled")

# ---------------- Load YOLO model ----------------
try:
    yolo_model = YOLO("yolov8n.pt")
    print("✅ YOLO: Model loaded successfully")
except Exception as e:
    print(f"❌ YOLO: Failed to load model - {e}")
    yolo_model = None

# ---------------- Load known faces ----------------
known_faces = []
known_names = []

if not os.path.exists(WANTED_DIR):
    print(f"📁 Creating wanted directory: {WANTED_DIR}")
    os.makedirs(WANTED_DIR, exist_ok=True)

wanted_files = [f for f in os.listdir(WANTED_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"📸 Found {len(wanted_files)} images in wanted directory")

for f in wanted_files:
    try:
        image_path = os.path.join(WANTED_DIR, f)
        pil_img = Image.open(image_path).convert("RGB")
        image = np.array(pil_img)
        
        # Detect faces in the image
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"⚠️ No face found in {f}, skipping")
            continue
            
        encodings = face_recognition.face_encodings(image, face_locations)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(os.path.splitext(f)[0])
            print(f"✅ Loaded face: {f} (found {len(face_locations)} face(s))")
        else:
            print(f"⚠️ Could not encode face in {f}")
            
    except Exception as e:
        print(f"❌ Error loading {f}: {e}")

print(f"👤 Total wanted faces loaded: {len(known_faces)}")

# ---------------- Enhanced Video Stream ----------------
class VideoStream:
    def __init__(self, src=0):
        # Try different backends for better compatibility
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(src)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {src}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        print("✅ Camera: Stream started successfully")

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
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

# ---------------- Snapshot Utility ----------------
last_saved = {}
def save_snapshot(folder, prefix, frame):
    ts = time.strftime("%Y%m%d-%H%M%S")
    key = f"{folder}_{prefix}"
    if last_saved.get(key, 0) + SNAPSHOT_COOLDOWN_S > time.time():
        return None
    last_saved[key] = time.time()
    filename = f"{prefix}_{ts}.jpg"
    path = os.path.join(folder, filename)
    try:
        cv2.imwrite(path, frame)
        print(f"💾 Saved: {path}")
        return path
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return None

# ---------------- Enhanced Detection Thread ----------------
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
        self.frame_count = 0

    def detect(self, frame):
        self.frame_count += 1
        
        # Process every 2nd frame for better performance
        if self.frame_count % 2 != 0:
            return

        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (640, 360))
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return

        faces_detected = []
        objects_detected = []
        wanted_count = 0
        alert_active = False

        # --- Face Recognition ---
        try:
            face_locations = face_recognition.face_locations(rgb_small, model="hog")  # Use hog for faster processing
            print(f"🔍 Found {len(face_locations)} face(s) in frame")
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                
                for enc, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    name = "Unknown"
                    
                    if known_faces:
                        # Calculate distances to known faces
                        distances = face_recognition.face_distance(known_faces, enc)
                        if len(distances) > 0:
                            best_match_idx = np.argmin(distances)
                            best_distance = distances[best_match_idx]
                            
                            # Use tolerance for matching
                            if best_distance <= FACE_TOLERANCE:
                                name = known_names[best_match_idx]
                                wanted_count += 1
                                alert_active = True
                                print(f"🎯 MATCH: {name} (distance: {best_distance:.3f})")
                                
                                # Trigger alarm if cooldown has passed
                                if time.time() - self.last_alarm > ALARM_COOLDOWN_S:
                                    # Look for custom audio for this person
                                    audio_file_mp3 = os.path.join(WANTED_DIR, f"{name}.mp3")
                                    audio_file_wav = os.path.join(WANTED_DIR, f"{name}.wav")
                                    
                                    if os.path.exists(audio_file_mp3):
                                        trigger_alarm(audio_file_mp3)
                                    elif os.path.exists(audio_file_wav):
                                        trigger_alarm(audio_file_wav)
                                    else:
                                        trigger_alarm()  # Use default alarm
                                    
                                    self.last_alarm = time.time()
                                
                                # Save snapshot
                                snapshot_path = save_snapshot(FACE_CAPTURES_DIR, name, frame)
                                
                                # Save to MongoDB
                                if MONGO_ENABLED and detections_collection is not None:
                                    try:
                                        detections_collection.insert_one({
                                            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                                            "type": "wanted_face",
                                            "name": name,
                                            "distance": float(best_distance),
                                            "snapshot": snapshot_path,
                                            "alert": True
                                        })
                                        print(f"📊 Saved to MongoDB: {name}")
                                    except Exception as e:
                                        print(f"❌ MongoDB save failed: {e}")

                    # Scale coordinates back to original frame size
                    h_ratio = frame.shape[0] / 360
                    w_ratio = frame.shape[1] / 640
                    faces_detected.append((
                        int(top * h_ratio), int(right * w_ratio),
                        int(bottom * h_ratio), int(left * w_ratio), name
                    ))
                    
                    # Save every detected face (throttled by cooldown)
                    save_snapshot(FACE_CAPTURES_DIR, "face", frame)
                    
        except Exception as e:
            print(f"❌ Face detection error: {e}")

        # --- Object Detection ---
        if yolo_model is not None:
            try:
                results = yolo_model(rgb_small, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
                
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            try:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                cls_id = int(box.cls[0])
                                label = yolo_model.names.get(cls_id, f"obj_{cls_id}")
                                conf = float(box.conf[0])
                                
                                # Scale coordinates
                                x1 = int(x1 * frame.shape[1] / 640)
                                y1 = int(y1 * frame.shape[0] / 360)
                                x2 = int(x2 * frame.shape[1] / 640)
                                y2 = int(y2 * frame.shape[0] / 360)
                                
                                objects_detected.append((x1, y1, x2, y2, label, conf))
                                save_snapshot(OBJECT_CAPTURES_DIR, label, frame)
                                
                                # Save to MongoDB
                                if MONGO_ENABLED and detections_collection is not None:
                                    try:
                                        detections_collection.insert_one({
                                            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                                            "type": "object",
                                            "name": label,
                                            "confidence": conf,
                                            "alert": False
                                        })
                                    except Exception as e:
                                        print(f"❌ MongoDB object save failed: {e}")
                                        
                            except Exception as e:
                                print(f"❌ Object box error: {e}")
            except Exception as e:
                print(f"❌ Object detection error: {e}")

        with self.lock:
            self.faces = faces_detected
            self.objects = objects_detected
            self.alert_active = alert_active
            self.flash_state = not self.flash_state
            self.wanted_count = wanted_count
            self.object_count = len(objects_detected)

# ---------------- Main Loop ----------------
def main():
    print("🎬 Starting main detection loop...")
    
    try:
        stream = VideoStream(0)
        # Let camera initialize
        time.sleep(2.0)
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        print("💡 Try changing camera index or check camera permissions")
        return

    detector = DetectionThread()

    def detection_worker():
        while True:
            ret, frame = stream.read()
            if ret and frame is not None:
                detector.detect(frame)
            else:
                time.sleep(0.01)

    # Start detection thread
    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()

    window_name = "Smart Face Detection - Press 'Q' to quit, 'S' for snapshot"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("✅ System ready! Detection running...")
    print("💡 Press 'Q' to quit, 'S' to take manual snapshot")
    
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("⚠️ No frame from camera")
                time.sleep(0.1)
                continue

            # Draw detections on frame
            with detector.lock:
                # Draw face bounding boxes
                for top, right, bottom, left, name in detector.faces:
                    color = (0, 0, 255) if name != "Unknown" else (255, 255, 255)
                    thickness = 3 if name != "Unknown" else 2
                    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                    
                    # Add label with background for better visibility
                    label = f"{name}" if name != "Unknown" else "Unknown"
                    label_bg = (0, 0, 0) if name != "Unknown" else (50, 50, 50)
                    
                    # Draw background for text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (left, top - text_size[1] - 10), 
                                 (left + text_size[0], top), label_bg, -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw object bounding boxes
                for x1, y1, x2, y2, label, conf in detector.objects:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Object label with confidence
                    obj_label = f"{label} {conf:.2f}"
                    text_size = cv2.getTextSize(obj_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                                 (x1 + text_size[0], y1), (0, 0, 0), -1)
                    cv2.putText(frame, obj_label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Alert overlay (flashing)
                if detector.alert_active and detector.flash_state:
                    cv2.putText(frame, "!!! WANTED ALERT !!!", (50, 50),
                               cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

                # Status overlay
                cv2.putText(frame, f"WANTED: {detector.wanted_count}", (20, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f"Objects: {detector.object_count}", (20, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # FPS counter (approximate)
                fps_text = f"FPS: {int(1/(time.time() - getattr(main, 'last_time', time.time()) + 0.001))}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                main.last_time = time.time()

            # Show frame
            cv2.imshow(window_name, frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = save_snapshot(CAPTURES_DIR, "manual", frame)
                if path:
                    print(f"📸 Manual snapshot: {path}")
                    
    except KeyboardInterrupt:
        print("\n⏹️ Stopping by user request...")
    except Exception as e:
        print(f"❌ Main loop error: {e}")
        traceback.print_exc()
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("🛑 System stopped successfully.")

if __name__ == "__main__":
    main()