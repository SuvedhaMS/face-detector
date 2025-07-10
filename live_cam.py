import cv2
import os
import pickle
import numpy as np
import logging
import time
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis

# === Setup logging ===
logging.basicConfig(filename='events.log', level=logging.INFO, format='%(asctime)s - %(message)s')
def log_event(event: str):
    logging.info(event)

# === Create required directories ===
os.makedirs("static/logs", exist_ok=True)

# === Load YOLOv8 model ===
print("🔄 Loading YOLOv8s...")
yolo = YOLO("yolov8s.pt")

# === Load DeepSort tracker ===
print("🔄 Initializing DeepSort...")
tracker = DeepSort(max_age=30)

# === Load InsightFace ===
print("🔄 Loading InsightFace...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# === Load or initialize face database ===
db_path = "faces.pkl"
if os.path.exists(db_path):
    with open(db_path, "rb") as f:
        face_db = pickle.load(f)
    print(f"✅ Loaded {len(face_db)} known faces from database.")
else:
    face_db = {}

# === Helper: cosine similarity ===
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# === Cooldown timer ===
last_seen = {}
cooldown = 10  # seconds

# === Start webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()
print("✅ Webcam started. Press 'r' to register, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == 0:  # Only 'person' class
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    faces = face_app.get(frame)
    for face in faces:
        emb = face.embedding
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        name = "Unknown"
        max_sim = 0.0
        for db_name, db_emb in face_db.items():
            sim = cosine_similarity(emb, db_emb)
            if sim > max_sim and sim > 0.6:
                name = db_name
                max_sim = sim

        # Log only if cooldown passed
        now = time.time()
        if name != "Unknown":
            if name in last_seen and (now - last_seen[name]) < cooldown:
                continue  # Skip repeated logging
            last_seen[name] = now

        # Draw box & label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Save snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        img_path = os.path.join("static", "logs", filename)
        cv2.imwrite(img_path, frame)

        # Log to CSV
        with open("visitor_log.csv", "a") as log:
            log.write(f"{datetime.now()},{name},{img_path}\n")

        # Log event
        if name == "Unknown":
            log_event("Detected unknown person")
        else:
            log_event(f"Recognized {name}")

    # Display DeepSort IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltwh()
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    cv2.imshow("Live Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        print("✍️  Enter name for the new face:")
        name = input("Name: ")
        if faces:
            face_db[name] = faces[0].embedding
            with open(db_path, "wb") as f:
                pickle.dump(face_db, f)
            print(f"✅ Registered face as {name}")
            log_event(f"Registered new face: {name}")
        else:
            print("❌ No face detected to register.")

cap.release()
cv2.destroyAllWindows()
