import cv2
import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_distances
from db import get_connection

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")

# -------------------------------
# Load trained classifier + names
# -------------------------------
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    names = saved["names"]

# -------------------------------
# Setup InsightFace
# -------------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# -------------------------------
# Load raw training embeddings for distance-based "Unknown" rejection
# (KNN's predict_proba alone isn't reliable for open-set rejection,
# so we check real cosine distance to the nearest known face)
# -------------------------------
def load_training_data():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, embedding FROM face_embeddings")
    rows = cursor.fetchall()
    conn.close()

    X, y = [], []
    for student_id, embedding_blob in rows:
        X.append(np.frombuffer(embedding_blob, dtype=np.float32))
        y.append(student_id)
    return np.array(X), np.array(y)


TRAIN_X, TRAIN_Y = load_training_data()

# Tune this after testing:
# - Lower it if strangers get misidentified as students
# - Raise it if real students get rejected as "Unknown" too often
DISTANCE_THRESHOLD = 0.4

already_marked_today = set()


def already_marked(student_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 1 FROM attendance
        WHERE student_id = ? AND DATE(timestamp) = DATE('now', 'localtime')
    """, (student_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def mark_attendance(student_id, name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attendance (student_id, name, status)
        VALUES (?, ?, 'Present')
    """, (student_id, name))
    conn.commit()
    conn.close()
    print(f"📝 Attendance marked: {name} ({student_id})")


def run_recognition():
    cv2.namedWindow("Attendance", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Attendance", 1000, 750)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("Live attendance running. Press ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        faces = face_app.get(frame)

        for face in faces:
            box = face.bbox.astype(int)
            embedding = face.embedding.reshape(1, -1)

            # ---- Distance-based recognition (the real "known vs unknown" check) ----
            distances = cosine_distances(embedding, TRAIN_X)[0]
            min_dist = np.min(distances)
            nearest_idx = np.argmin(distances)
            nearest_student_id = TRAIN_Y[nearest_idx]
            similarity = 1 - min_dist

            if min_dist <= DISTANCE_THRESHOLD:
                pred_id = nearest_student_id
                name = names.get(pred_id, "Unknown")
                label = f"{name} ({similarity:.2f})"
                color = (0, 255, 0)

                # Mark attendance once per day per student
                if pred_id not in already_marked_today:
                    if not already_marked(pred_id):
                        mark_attendance(pred_id, name)
                    already_marked_today.add(pred_id)
            else:
                label = f"Unknown ({similarity:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(display, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(display, "[ESC] Quit", (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Attendance", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()