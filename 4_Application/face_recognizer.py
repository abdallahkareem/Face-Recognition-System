import cv2
import numpy as np
import pickle
from deepface import DeepFace
from numpy import dot
from numpy.linalg import norm

# =========================
# Load trained embeddings
# =========================
with open("facenet_trained.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_names = data["names"]

# =========================
# Cosine distance
# =========================
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - dot(a, b) / (norm(a) * norm(b))

# =========================
# Find best match
# =========================
def find_match(face_embedding):

    min_distance = float("inf")
    matched_name = "Unknown"

    for i, known_embedding in enumerate(known_embeddings):

        distance = cosine_distance(face_embedding, known_embedding)

        if distance < min_distance:
            min_distance = distance
            matched_name = known_names[i]

    # Threshold (IMPORTANT)
    if min_distance > 0.4:
        return "Unknown", min_distance

    return matched_name, min_distance


# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    try:

        # resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        results = DeepFace.represent(
            img_path=small_frame,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )

        for face in results:

            embedding = face["embedding"]

            x = face["facial_area"]["x"] * 2
            y = face["facial_area"]["y"] * 2
            w = face["facial_area"]["w"] * 2
            h = face["facial_area"]["h"] * 2

            name, distance = find_match(embedding)

            # color
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # label
            text = f"{name} ({distance:.2f})"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Face Recognition - FaceNet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
