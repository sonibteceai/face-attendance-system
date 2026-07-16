import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import cv2
import numpy as np
import threading
import av
import re
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from db import get_connection
from train_classifier import train_and_save  # now persists model to Turso, not just local disk

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

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


face_app = load_face_app()


# -------------------------------
# ID suggestion + duplicate check
# -------------------------------
def get_existing_student_ids():
    conn = get_connection()
    conn.sync()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id FROM student_profiles")
    rows = cursor.fetchall()
    conn.close()
    return {r[0] for r in rows}


def suggest_next_id(existing_ids):
    numeric_ids = []
    for sid in existing_ids:
        if sid.isdigit():
            numeric_ids.append(int(sid))
    next_id = (max(numeric_ids) + 1) if numeric_ids else 1
    return str(next_id)


existing_ids = get_existing_student_ids()

if "suggested_id" not in st.session_state:
    st.session_state.suggested_id = suggest_next_id(existing_ids)


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
student_id_raw = col1.text_input(
    "Student ID",
    value=st.session_state.suggested_id,
    help="Auto-suggested — feel free to change it to whatever ID scheme you use.",
)
name_raw = col2.text_input("Student Name")

# ---- Clean + validate inputs ----
student_id = student_id_raw.strip()
name = name_raw.strip()

validation_error = None
if not student_id or not name:
    validation_error = "Enter both Student ID and Student Name to continue."
elif not ID_PATTERN.match(student_id):
    validation_error = "Student ID can only contain letters, numbers, hyphens, and underscores."
elif not any(c.isalpha() for c in name):
    validation_error = "Student Name must contain at least one letter."
elif student_id in existing_ids:
    validation_error = f"Student ID '{student_id}' is already taken. Choose a different one."

if validation_error:
    st.warning(f"⚠️ {validation_error}")

if "samples" not in st.session_state:
    st.session_state.samples = []
if "profile_photo_path" not in st.session_state:
    st.session_state.profile_photo_path = None
if "just_registered" not in st.session_state:
    st.session_state.just_registered = False

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

capture_disabled = sample_count >= NUM_SAMPLES or validation_error is not None

if capture_col.button("📷 Capture Sample", disabled=capture_disabled):
    if ctx.video_processor:
        with ctx.video_processor.lock:
            embedding = ctx.video_processor.latest_embedding
            frame = ctx.video_processor.latest_frame
        if embedding is not None:
            st.session_state.samples.append(embedding)
            if st.session_state.profile_photo_path is None:
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

save_disabled = sample_count < NUM_SAMPLES or validation_error is not None

if st.button("💾 Save Registration", disabled=save_disabled):
    # Re-check duplicates right before saving too, in case someone else
    # registered the same ID in another session moments ago
    fresh_ids = get_existing_student_ids()
    if student_id in fresh_ids:
        st.error(f"❌ Student ID '{student_id}' was just taken by someone else. Pick a different ID.")
    else:
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
            st.session_state.just_registered = True
            # Refresh the suggested next ID for whoever registers next
            st.session_state.suggested_id = suggest_next_id(fresh_ids | {student_id})
        except Exception as e:
            if "constraint" in str(e).lower() or "unique" in str(e).lower():
                st.error(f"❌ Student ID '{student_id}' already exists.")
            else:
                st.error(f"❌ Registration failed: {e}")
        finally:
            conn.close()

# Show retrain option right after a successful registration
if st.session_state.just_registered:
    st.divider()
    st.info("New student data was added. Retrain the model so attendance recognizes them.")
    if st.button("🔁 Retrain Model Now"):
        with st.spinner("Retraining classifier on all registered students..."):
            try:
                trained = train_and_save()
                if trained:
                    st.cache_resource.clear()  # so Mark Attendance page reloads the new model
                    st.success("✅ Model retrained and saved to the database.")
                    st.session_state.just_registered = False
                else:
                    st.error("❌ Need at least 2 registered students with samples to train.")
            except Exception as e:
                st.error(f"❌ Retraining failed: {e}")


# ─────────────────────────────────────────────────────────────
# 🗑️ Manage / Remove Students
# ─────────────────────────────────────────────────────────────
st.divider()
st.header("🗑️ Manage Students")


def get_all_students():
    conn = get_connection()
    conn.sync()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, photo_path FROM student_profiles ORDER BY name")
    rows = cursor.fetchall()
    conn.close()
    return rows  # list of (student_id, name, photo_path)


def delete_student(student_id, photo_path):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM student_profiles WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.sync()
    conn.close()

    if photo_path and os.path.exists(photo_path):
        try:
            os.remove(photo_path)
        except Exception:
            pass  # non-fatal — DB records are already gone


def delete_all_data():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    cursor.execute("DELETE FROM face_embeddings")
    cursor.execute("DELETE FROM student_profiles")
    cursor.execute("DELETE FROM model_store")
    conn.commit()
    conn.sync()
    conn.close()

    # Best-effort cleanup of saved photos and local model cache
    try:
        for fname in os.listdir(PHOTO_DIR):
            fpath = os.path.join(PHOTO_DIR, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
    except Exception:
        pass
    try:
        local_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")
        if os.path.exists(local_model_path):
            os.remove(local_model_path)
    except Exception:
        pass


students = get_all_students()

if not students:
    st.caption("No students registered yet.")
else:
    st.subheader("Remove a single student")
    options = {f"{name} ({sid})": (sid, name, photo_path) for sid, name, photo_path in students}
    selected_label = st.selectbox("Select a student to remove", list(options.keys()))
    selected_sid, selected_name, selected_photo = options[selected_label]

    confirm_single = st.checkbox(
        f"I understand this will permanently delete **{selected_name}**'s profile, "
        f"face samples, and attendance history."
    )
    if st.button("🗑️ Delete Selected Student", disabled=not confirm_single):
        with st.spinner(f"Deleting {selected_name}..."):
            delete_student(selected_sid, selected_photo)
        st.success(f"✅ {selected_name} ({selected_sid}) has been removed.")
        st.info("Remember to retrain the model so it forgets this student too.")
        st.session_state.suggested_id = suggest_next_id(get_existing_student_ids())
        st.rerun()

st.divider()
st.subheader("⚠️ Danger zone: remove all data")
st.caption("This permanently deletes every student, face sample, attendance record, and the trained model.")

danger_confirm_text = st.text_input(
    "Type DELETE ALL to confirm you want to wipe everything", value="", key="danger_confirm"
)
if st.button("🧨 Delete ALL Students & Data", disabled=danger_confirm_text.strip() != "DELETE ALL"):
    with st.spinner("Deleting all data..."):
        delete_all_data()
        st.cache_resource.clear()
    st.success("✅ All student data, attendance records, and the trained model have been removed.")
    st.session_state.suggested_id = "1"
    st.rerun()