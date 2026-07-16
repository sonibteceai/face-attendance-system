import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import cv2
import numpy as np
import threading
import av
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_distances
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from db import get_connection
from train_classifier import train_and_save, load_model_from_db

st.set_page_config(page_title="Mark Attendance")

DISTANCE_THRESHOLD = 0.4


@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


@st.cache_resource
def load_classifier():
    saved = load_model_from_db()
    if saved is None:
        return None, None
    return saved["model"], saved["names"]


@st.cache_resource
def load_training_data():
    conn = get_connection()
    conn.sync()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, embedding FROM face_embeddings")
    rows = cursor.fetchall()
    conn.close()
    X, y = [], []
    for student_id, embedding_blob in rows:
        X.append(np.frombuffer(embedding_blob, dtype=np.float32))
        y.append(student_id)
    return np.array(X), np.array(y)


st.title("🎥 Live Attendance")

model, names = load_classifier()

if model is None:
    st.warning(
        "⚠️ No trained model found yet. Register at least 2 students on the "
        "**Register Student** page, then click **Retrain Model Now** there before "
        "marking attendance."
    )
    if st.button("🔁 Try Training Now"):
        with st.spinner("Training classifier on all registered students..."):
            try:
                trained = train_and_save()
                if trained:
                    st.cache_resource.clear()
                    st.success("✅ Model trained. Reloading page...")
                    st.rerun()
                else:
                    st.error("❌ Training needs at least 2 registered students.")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
    st.stop()

TRAIN_X, TRAIN_Y = load_training_data()

if len(TRAIN_X) == 0:
    st.warning("⚠️ No face embeddings found in the database. Register students first.")
    st.stop()

face_app = load_face_app()


def already_marked(student_id):
    conn = get_connection()
    conn.sync()  # pull latest data from Turso before checking
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM attendance WHERE student_id = ? AND DATE(timestamp) = DATE('now', 'localtime')",
        (student_id,),
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None


def mark_attendance_db(student_id, name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO attendance (student_id, name, status) VALUES (?, ?, 'Present')",
        (student_id, name),
    )
    conn.commit()
    conn.sync()  # push/refresh local replica after write
    conn.close()
    print(f"📝 Attendance marked: {name} ({student_id})")


class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.marked_today = set()
        self.last_marked_name = None

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
                student_name = names.get(pred_id, "Unknown")
                label = f"{student_name} ({similarity:.2f})"
                color = (0, 255, 0)

                with self.lock:
                    seen_this_session = pred_id in self.marked_today
                if not seen_this_session:
                    if not already_marked(pred_id):
                        mark_attendance_db(pred_id, student_name)
                        with self.lock:
                            self.last_marked_name = student_name
                    with self.lock:
                        self.marked_today.add(pred_id)
            else:
                label = f"Unknown ({similarity:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.caption("Attendance is marked automatically, once per student per day.")

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    ctx = webrtc_streamer(
        key="attendance",
        video_processor_factory=AttendanceProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        video_html_attrs={
            "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #444", "border-radius": "8px"},
            "controls": False,
            "autoPlay": True,
        },
    )

if ctx.video_processor:
    with ctx.video_processor.lock:
        marked_names = list(ctx.video_processor.marked_today)
    if marked_names:
        st.success(f"Marked today (this session): {', '.join(str(m) for m in marked_names)}")