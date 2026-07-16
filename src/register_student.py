import cv2
import numpy as np
import os
import sqlite3
from insightface.app import FaceAnalysis
from db import get_connection

# -------------------------------
# Setup InsightFace
# -------------------------------
face_app = FaceAnalysis(name="buffalo_l")   # ArcFace 512-D model
face_app.prepare(ctx_id=0, det_size=(640, 640))   # GPU if available

# -------------------------------
# Paths
# -------------------------------
PHOTO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "student_faces")
os.makedirs(PHOTO_DIR, exist_ok=True)

NUM_SAMPLES = 5


def capture_and_register(student_id, name):





    # -------------------------------
    # Create Resizable Window
    # -------------------------------


    embeddings = []
    count = 0
    profile_photo_path = None

    pose_prompts = [
        "Look straight at the camera",
        "Turn your head slightly LEFT",
        "Turn your head slightly RIGHT",
        "Tilt your head slightly UP",
        "Tilt your head slightly DOWN"
    ]

    print(f"\nRegistering {name} ({student_id})")
    print("Press SPACE to capture.")
    print("Press ESC to cancel.\n")
    cv2.namedWindow("Register Student", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Register Student", 1000, 750)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return


    while count < NUM_SAMPLES:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read frame.")
            break

        display = frame.copy()

        faces = face_app.get(frame)

        # --------------------------------
        # Draw Face Box
        # --------------------------------
        if len(faces) == 1:

            face = faces[0]
            box = face.bbox.astype(int)

            cv2.rectangle(
                display,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),
                2,
            )

            status_text = "Face Detected - Press SPACE"
            status_color = (0, 255, 0)

        elif len(faces) == 0:

            status_text = "No Face Detected"
            status_color = (0, 0, 255)

        else:

            status_text = "Multiple Faces Detected"

            status_color = (0, 0, 255)

            for f in faces:
                box = f.bbox.astype(int)
                cv2.rectangle(
                    display,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 0, 255),
                    2,
                )

        h, w = display.shape[:2]
        # --------------------------------
        # Instructions
        # --------------------------------
        cv2.putText(
            display,
            f"Student : {name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"Sample : {count + 1}/{NUM_SAMPLES}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"Pose : {pose_prompts[count]}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            display,
            status_text,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        cv2.putText(
            display,
            "[SPACE] Capture   [ESC] Cancel",
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2,
        )

        cv2.imshow("Register Student", display)

        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:
            print("Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

        # SPACE
        if key == 32:

            if len(faces) == 0:
                print("❌ No face detected.")
                continue

            if len(faces) > 1:
                print("❌ Multiple faces detected.")
                continue

            face = faces[0]

            embeddings.append(face.embedding)

            count += 1

            if profile_photo_path is None:
                profile_photo_path = os.path.join(
                    PHOTO_DIR,
                    f"{student_id}.jpg",
                )
                cv2.imwrite(profile_photo_path, frame)

            print(f"✅ Captured {count}/{NUM_SAMPLES}")

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("No embeddings captured. Registration aborted.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Insert profile once
        cursor.execute("""
                       INSERT INTO student_profiles (student_id, name, photo_path)
                       VALUES (?, ?, ?)
                       """, (student_id, name, profile_photo_path))

        # Insert each of the 5 embeddings as a separate row
        for emb in embeddings:
            cursor.execute("""
                           INSERT INTO face_embeddings (student_id, embedding)
                           VALUES (?, ?)
                           """, (student_id, emb.astype(np.float32).tobytes()))

        conn.commit()
        print(f"✅ {name} registered successfully with {len(embeddings)} samples.")
    except sqlite3.IntegrityError:
        print(f"❌ Student ID '{student_id}' already exists.")
    finally:
        conn.close()





if __name__ == "__main__":

    sid = input("Enter Student ID : ").strip()

    sname = input("Enter Student Name : ").strip()

    capture_and_register(sid, sname)