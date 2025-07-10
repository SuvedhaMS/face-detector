import os
import cv2
import numpy as np
import pickle
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import json
import csv

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

frame_skip = config.get("frame_skip", 5)
face_db_path = config.get("face_db_path", "database/faces.pkl")

# Load models
print("🔄 Loading YOLOv8 model...")
yolo = YOLO("yolov8s.pt")

print("🔄 Loading InsightFace model...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# Create folders if missing
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("database", exist_ok=True)

# Initialize face database
if os.path.exists(face_db_path):
    with open(face_db_path, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {"embeddings": [], "ids": []}

# ✅ Initialize CSV file if it doesn't exist
csv_path = "visitor_log.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Person ID", "Image File"])

# ✅ Function to log detection to CSV
def log_to_csv(person_id, image_filename):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, person_id, image_filename])

# Helper: match face to known database
def match_face(embedding, threshold=0.5):
    if not face_db["embeddings"]:
        return None
    sims = np.dot(face_db["embeddings"], embedding)
    best_idx = np.argmax(sims)
    if sims[best_idx] >= threshold:
        return face_db["ids"][best_idx]
    return None

# ✅ Register and log new face
def register_new_face(embedding, face_crop):
    new_id = f"person_{len(face_db['ids']) + 1}"
    face_db["embeddings"].append(embedding)
    face_db["ids"].append(new_id)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/{new_id}_{now}.jpg"
    cv2.imwrite(filename, face_crop)

    # Save updated face DB
    with open(face_db_path, "wb") as f:
        pickle.dump(face_db, f)

    # Log to CSV
    log_to_csv(new_id, filename)

    return new_id

# Process each video
video_folder = "videos"
for filename in os.listdir(video_folder):
    if not filename.lower().endswith((".mp4", ".avi")):
        continue

    video_path = os.path.join(video_folder, filename)
    cap = cv2.VideoCapture(video_path)
    print(f"\n🎥 Processing {filename}...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            results = yolo(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]

                faces = face_app.get(face_crop)
                if not faces:
                    continue

                embedding = faces[0]["embedding"]
                match_id = match_face(embedding)

                if match_id is None:
                    match_id = register_new_face(embedding, face_crop)
                    print(f"🆕 New face registered: {match_id}")
                else:
                    print(f"✅ Recognized: {match_id}")
                    # ✅ Log recognized face too (optional)
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"logs/{match_id}_{now}.jpg"
                    cv2.imwrite(image_filename, face_crop)
                    log_to_csv(match_id, image_filename)

        frame_count += 1

    cap.release()

# ✅ Final Summary
print("\n✅ All videos processed and logged.")
print(f"👥 Total unique persons registered: {len(face_db['ids'])}")
