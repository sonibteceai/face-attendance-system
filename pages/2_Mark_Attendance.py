import streamlit as st
import cv2
import numpy as np
import pickle
import sqlite3
import os
import av
import time
import pandas as pd
from datetime import date
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_distances
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "attendance.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _init_db():
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


_init_db()

st.set_page_config(page_title="Mark Attendance", layout="wide")
st.title("📸 Live Attendance")

DISTANCE_THRESHOLD = 0.4

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# -------------------------------
# Cached resources (loaded once)
# -------------------------------
@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


@st.cache_resource
def load_classifier():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
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


if not os.path.exists(MODEL_PATH):
    st.error("No trained classifier found. Register students and train the classifier first.")
    st.stop()

face_app = load_face_app()
saved = load_classifier()
names = saved["names"]
TRAIN_X, TRAIN_Y = load_training_data()

if st.button("🔄 Reload cache (after retraining on the Register Student page)"):
    load_classifier.clear()
    load_training_data.clear()
    st.rerun()

st.caption(f"Loaded {len(TRAIN_X)} face samples across {len(set(TRAIN_Y))} students.")


# -------------------------------
# Attendance helper functions
# -------------------------------
def already_marked_today(conn, student_id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM attendance WHERE student_id = ? AND DATE(timestamp) = DATE('now', 'localtime')",
        (student_id,),
    )
    return cursor.fetchone() is not None


def mark_attendance(conn, student_id, name):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO attendance (student_id, name, status) VALUES (?, ?, 'Present')",
        (student_id, name),
    )
    conn.commit()


# -------------------------------
# Video processor — runs per frame in a background thread
# -------------------------------
class AttendanceProcessor:
    def __init__(self):
        self.conn = get_connection()
        self.marked_today_cache = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face_app.get(img)

        for face in faces:
            box = face.bbox.astype(int)
            embedding = face.embedding.reshape(1, -1)

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

                if pred_id not in self.marked_today_cache:
                    if not already_marked_today(self.conn, pred_id):
                        mark_attendance(self.conn, pred_id, name)
                    self.marked_today_cache.add(pred_id)
            else:
                label = f"Unknown ({similarity:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------
# Start the live video stream
# -------------------------------
col_video, col_log = st.columns([2, 1])

with col_video:
    webrtc_streamer(
        key="attendance-stream",
        video_processor_factory=AttendanceProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col_log:
    st.subheader("✅ Marked Today")
    placeholder = st.empty()

    if st.button("Refresh list"):
        st.rerun()

    conn = get_connection()
    today_df = conn.execute(
        "SELECT name, timestamp FROM attendance WHERE DATE(timestamp) = DATE('now','localtime') ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    if today_df:
        for name, ts in today_df:
            ist_time = pd.to_datetime(ts, utc=True).tz_convert("Asia/Kolkata").strftime("%I:%M:%S %p")
            st.write(f"**{name}** — {ist_time}")
    else:
        st.info("No one marked yet today.")