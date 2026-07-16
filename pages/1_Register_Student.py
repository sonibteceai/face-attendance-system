import streamlit as st
import cv2
import numpy as np
import sqlite3
import os
from insightface.app import FaceAnalysis

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "attendance.db")
PHOTO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "student_faces")
os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profiles (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            photo_path TEXT,
            registered_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (student_id) REFERENCES student_profiles(student_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES student_profiles(student_id)
        )
    """)
    conn.commit()
    conn.close()


init_db()

NUM_SAMPLES = 5
POSE_PROMPTS = [
    "Look straight at the camera",
    "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT",
    "Tilt your head slightly UP",
    "Tilt your head slightly DOWN",
]

st.set_page_config(page_title="Register Student", layout="centered")
st.title("🧑‍🎓 Register New Student")


# -------------------------------
# Load InsightFace once, cached across reruns
# -------------------------------
@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


face_app = load_face_app()


def get_connection():
    return sqlite3.connect(DB_PATH)


# -------------------------------
# Session state to accumulate captures across reruns
# -------------------------------
if "captures" not in st.session_state:
    st.session_state.captures = []  # list of (embedding, frame_bgr)
if "registered" not in st.session_state:
    st.session_state.registered = False


# -------------------------------
# Student info form
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("Student ID", key="reg_student_id")
with col2:
    student_name = st.text_input("Student Name", key="reg_student_name")

if not student_id or not student_name:
    st.info("Enter Student ID and Name to begin capturing photos.")
    st.stop()

# Check if ID already exists
conn = get_connection()
existing = conn.execute(
    "SELECT 1 FROM student_profiles WHERE student_id = ?", (student_id,)
).fetchone()
conn.close()

if existing:
    st.error(f"Student ID '{student_id}' already exists. Choose a different ID.")
    st.stop()

st.divider()

# -------------------------------
# Capture loop
# -------------------------------
progress = len(st.session_state.captures)

if progress < NUM_SAMPLES:
    st.subheader(f"Sample {progress + 1}/{NUM_SAMPLES}")
    st.markdown(f"**Pose:** {POSE_PROMPTS[progress]}")

    img_file = st.camera_input("Take photo", key=f"camera_{progress}")

    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        faces = face_app.get(frame)

        if len(faces) == 0:
            st.warning("❌ No face detected in this photo. Try again with better lighting/framing.")
        elif len(faces) > 1:
            st.warning("❌ Multiple faces detected. Make sure only one person is in frame.")
        else:
            face = faces[0]
            st.session_state.captures.append((face.embedding, frame))
            st.success(f"✅ Captured sample {progress + 1}/{NUM_SAMPLES}")
            st.rerun()

else:
    st.success("All 5 samples captured!")

    # Show thumbnails of all captures
    thumb_cols = st.columns(NUM_SAMPLES)
    for i, (emb, frame) in enumerate(st.session_state.captures):
        with thumb_cols[i]:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Pose {i+1}", use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🔄 Retake All", use_container_width=True):
            st.session_state.captures = []
            st.rerun()

    with col_b:
        if st.button("💾 Save Registration", type="primary", use_container_width=True):
            # Save profile photo (first capture)
            profile_photo_path = os.path.join(PHOTO_DIR, f"{student_id}.jpg")
            cv2.imwrite(profile_photo_path, st.session_state.captures[0][1])

            conn = get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO student_profiles (student_id, name, photo_path) VALUES (?, ?, ?)",
                    (student_id, student_name, profile_photo_path),
                )
                for emb, _ in st.session_state.captures:
                    cursor.execute(
                        "INSERT INTO face_embeddings (student_id, embedding) VALUES (?, ?)",
                        (student_id, emb.astype(np.float32).tobytes()),
                    )
                conn.commit()
                st.session_state.registered = True
            except sqlite3.IntegrityError:
                st.error(f"Student ID '{student_id}' already exists.")
            finally:
                conn.close()

if st.session_state.registered:
    st.balloons()
    st.success(f"🎉 {student_name} registered successfully with 5 samples!")

    st.divider()
    st.subheader("🎯 Retrain the classifier")
    st.markdown(
        f"**{student_name}** won't be recognized in live attendance until the classifier is retrained "
        "on the updated dataset. Do it now:"
    )

    if st.button("🎯 Retrain Now (runs full model comparison)", type="primary", key="retrain_after_reg"):
        with st.spinner("Training KNN / SVM / LogisticRegression / RandomForest and picking the best..."):
            from collections import Counter
            import pickle as pkl
            from sklearn.model_selection import GridSearchCV, StratifiedKFold
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(MODEL_DIR, exist_ok=True)
            MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")

            conn = get_connection()
            emb_rows = conn.execute("SELECT student_id, embedding FROM face_embeddings").fetchall()
            name_rows = conn.execute("SELECT student_id, name FROM student_profiles").fetchall()
            conn.close()

            X, y = [], []
            for sid, emb_blob in emb_rows:
                X.append(np.frombuffer(emb_blob, dtype=np.float32))
                y.append(sid)
            X, y = np.array(X), np.array(y)
            name_lookup = dict(name_rows)

            n_classes = len(set(y))
            if n_classes < 2 or len(X) == 0:
                st.error("Need at least 2 registered students with samples to train.")
            else:
                smallest_class = min(Counter(y).values())
                cv_folds = max(2, min(3, smallest_class))
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

                candidates = {
                    "KNN": (Pipeline([("clf", KNeighborsClassifier())]),
                            {"clf__n_neighbors": [1, 3, 5], "clf__metric": ["cosine", "euclidean"], "clf__weights": ["uniform", "distance"]}),
                    "SVM_Linear": (Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", probability=True))]),
                                   {"clf__C": [0.1, 1, 10]}),
                    "LogisticRegression": (Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
                                            {"clf__C": [0.1, 1, 10]}),
                    "RandomForest": (Pipeline([("clf", RandomForestClassifier(random_state=42))]),
                                      {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]}),
                }

                best_overall, best_score, best_name = None, -1, None
                results_table = []

                for cname, (pipe, params) in candidates.items():
                    search = GridSearchCV(pipe, params, cv=cv, scoring="accuracy", n_jobs=-1)
                    search.fit(X, y)
                    results_table.append({"Model": cname, "CV Accuracy": round(search.best_score_, 4)})
                    if search.best_score_ > best_score:
                        best_score, best_overall, best_name = search.best_score_, search.best_estimator_, cname

                with open(MODEL_PATH, "wb") as f:
                    pkl.dump({"model": best_overall, "names": name_lookup, "model_type": best_name}, f)

                st.success(f"✅ Retrained. Best model: **{best_name}** (CV accuracy: {best_score:.2%})")
                st.dataframe(pd.DataFrame(results_table), hide_index=True)
                st.info("Go to the Mark Attendance page and click 'Reload cache' to pick up the new model.")

    if st.button("Register Another Student"):
        st.session_state.captures = []
        st.session_state.registered = False
        st.rerun()