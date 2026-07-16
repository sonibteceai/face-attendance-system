import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import cv2
import numpy as np
import threading
import av
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from db import get_connection

st.set_page_config(page_title="Register Student")

PHOTO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "student_faces")
os.makedirs(PHOTO_DIR, exist_ok=True)

NUM_SAMPLES = 5
POSE_PROMPTS = [
    "Look straight at the camera",
    "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT",
    "Tilt your head slightly UP",
    "Tilt your head slightly DOWN",
]


@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


face_app = load_face_app()


class RegisterProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_embedding = None
        self.status = "No Face Detected"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face_app.get(img)

        with self.lock:
            if len(faces) == 1:
                face = faces[0]
                box = face.bbox.astype(int)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                self.status = "Face Detected"
                self.latest_embedding = face.embedding
                self.latest_frame = img.copy()
            elif len(faces) == 0:
                self.status = "No Face Detected"
                self.latest_embedding = None
            else:
                self.status = "Multiple Faces Detected"
                self.latest_embedding = None
                for f in faces:
                    box = f.bbox.astype(int)
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        cv2.putText(img, self.status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.status == "Face Detected" else (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("📸 Register New Student")

col1, col2 = st.columns(2)
student_id = col1.text_input("Student ID")
name = col2.text_input("Student Name")

if "samples" not in st.session_state:
    st.session_state.samples = []
if "profile_photo_path" not in st.session_state:
    st.session_state.profile_photo_path = None

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    ctx = webrtc_streamer(
        key="register",
        video_processor_factory=RegisterProcessor,
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

sample_count = len(st.session_state.samples)

if sample_count < NUM_SAMPLES:
    st.info(f"Pose: **{POSE_PROMPTS[sample_count]}**  (Sample {sample_count + 1}/{NUM_SAMPLES})")
else:
    st.success("All samples captured — click Save below.")

capture_col, reset_col = st.columns(2)

if capture_col.button("📷 Capture Sample", disabled=sample_count >= NUM_SAMPLES):
    if ctx.video_processor:
        with ctx.video_processor.lock:
            embedding = ctx.video_processor.latest_embedding
            frame = ctx.video_processor.latest_frame
        if embedding is not None:
            st.session_state.samples.append(embedding)
            if st.session_state.profile_photo_path is None and student_id:
                path = os.path.join(PHOTO_DIR, f"{student_id}.jpg")
                cv2.imwrite(path, frame)
                st.session_state.profile_photo_path = path
            st.success(f"Captured {len(st.session_state.samples)}/{NUM_SAMPLES}")
            st.rerun()
        else:
            st.error("No single face detected right now — try again.")
    else:
        st.warning("Start the camera above first.")

if reset_col.button("🔄 Reset Samples"):
    st.session_state.samples = []
    st.session_state.profile_photo_path = None
    st.rerun()

st.divider()

save_disabled = sample_count < NUM_SAMPLES or not student_id or not name
if st.button("💾 Save Registration", disabled=save_disabled):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO student_profiles (student_id, name, photo_path) VALUES (?, ?, ?)",
            (student_id, name, st.session_state.profile_photo_path),
        )
        for emb in st.session_state.samples:
            cursor.execute(
                "INSERT INTO face_embeddings (student_id, embedding) VALUES (?, ?)",
                (student_id, emb.astype(np.float32).tobytes()),
            )
        conn.commit()
        conn.sync()
        st.success(f"✅ {name} registered successfully with {len(st.session_state.samples)} samples.")
        st.session_state.samples = []
        st.session_state.profile_photo_path = None
    except Exception as e:
        if "constraint" in str(e).lower():
            st.error(f"❌ Student ID '{student_id}' already exists.")
        else:
            st.error(f"❌ Registration failed: {e}")
    finally:
        conn.close()