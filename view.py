import pickle

# Load the face database
with open("database/faces.pkl", "rb") as f:
    face_db = pickle.load(f)

# Print number of persons
print(f"👥 Total persons in DB: {len(face_db['ids'])}\n")

# Print person IDs
for i, person_id in enumerate(face_db["ids"], 1):
    print(f"{i}. {person_id}")
