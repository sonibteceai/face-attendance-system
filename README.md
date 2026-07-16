# 🎓 Face Attendance System

A Streamlit app that registers students by face, trains a recognition model, and marks attendance automatically from a live camera feed — backed by [Turso](https://turso.tech) (libSQL) as a cloud-synced database.

---

## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| UI / App framework | Streamlit (multipage) |
| Face detection & embeddings | InsightFace (`buffalo_l`) via ONNX Runtime |
| Classifier | scikit-learn KNN (cosine distance), tuned with `GridSearchCV` |
| Database | Turso (libSQL) — cloud-synced SQLite |
| Live video | `streamlit-webrtc` |
| Dashboard charts | Plotly + Pandas |

---

## 📁 Project Structure

```
face-attendance-system/
├── dashboard.py                  # Home page — attendance overview & charts
├── pages/
│   ├── 1_Register_Student.py     # Register new students + manage/delete students
│   └── 2_Mark_Attendance.py      # Live camera attendance marking
├── src/
│   ├── db.py                     # Turso connection + schema (init_db)
│   ├── train_classifier.py       # Loads embeddings, trains KNN, saves model to Turso
│   ├── check_db.py                # CLI: list registered students
│   └── check_attendance.py       # CLI: list attendance records
├── data/student_faces/           # Saved profile photos (one per student)
├── requirements.txt
├── packages.txt                  # System-level deps (libgl1, etc.) for Streamlit Cloud
└── .env                          # TURSO_DATABASE_URL, TURSO_AUTH_TOKEN (not committed)
```

---

## 🗄️ Database Schema (Turso)

| Table | Purpose |
|---|---|
| `student_profiles` | `student_id` (PK), `name`, `photo_path`, `registered_on` |
| `face_embeddings` | Multiple face samples (BLOB) per `student_id` |
| `attendance` | One row per check-in: `student_id`, `name`, `timestamp`, `status` |
| `model_store` | Single-row table holding the pickled, trained classifier — so it survives app restarts on ephemeral hosting |

Run once to create the schema:
```bash
python src/db.py
```

---

## ⚙️ Setup

1. **Clone and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the project root:
   ```
   TURSO_DATABASE_URL=libsql://your-db-name.turso.io
   TURSO_AUTH_TOKEN=your-auth-token
   ```

3. **Initialize the database** (creates all tables if they don't exist):
   ```bash
   python src/db.py
   ```

4. **Run the app locally**:
   ```bash
   streamlit run dashboard.py
   ```

### Deploying on Streamlit Community Cloud
- Add `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` under **App settings → Secrets**.
- Keep `opencv-python-headless` (not `opencv-python`) in `requirements.txt` and `packages.txt` containing `libgl1` / `libglib2.0-0` — avoids the common `libGL.so.1` crash on headless Linux hosts.
- Free-tier RAM is limited (~1GB). Training uses a single lightweight KNN model with `n_jobs=1` to stay within that budget.

---

## 🔄 Core Workflow

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────────┐     ┌───────────────┐
│ 1. Register       │ ──▶ │ 2. Train Model     │ ──▶ │ 3. Mark Attendance    │ ──▶ │ 4. Dashboard   │
│ Capture 5 face    │     │ KNN + cosine,      │     │ Live camera matches   │     │ View records,  │
│ samples per       │     │ GridSearchCV picks │     │ faces against the     │     │ charts, export │
│ student, save to  │     │ best params, saved │     │ trained model, once   │     │                │
│ Turso              │     │ to Turso           │     │ per student per day   │     │                │
└──────────────────┘     └───────────────────┘     └──────────────────────┘     └───────────────┘
```

### 1. Register a Student — `pages/1_Register_Student.py`
- Enter **Student ID** and **Name**.
- Start the camera, capture **5 pose samples** (straight, left, right, up, down).
- Click **💾 Save Registration** — writes the profile + embeddings to Turso.
- A **🔁 Retrain Model Now** button appears right after saving — click it so the new student is recognized.

### 2. Train the Classifier — `src/train_classifier.py`
- Pulls all face embeddings + student names from Turso.
- Trains a single **KNN (cosine distance)** classifier, using `GridSearchCV` to tune `n_neighbors` and `weights`.
- Saves the trained model as pickled bytes into the `model_store` table in Turso (persists across app restarts, unlike local disk on Streamlit Cloud).
- Can be triggered from the Register page button, or run manually:
  ```bash
  python src/train_classifier.py
  ```

### 3. Mark Attendance — `pages/2_Mark_Attendance.py`
- Loads the trained model from Turso.
- Live webcam feed detects faces and compares embeddings via cosine distance.
- If a match is confident enough (`DISTANCE_THRESHOLD = 0.4`) and the student hasn't already checked in today, it inserts a row into `attendance`.
- If no model exists yet, the page shows a warning with a **🔁 Try Training Now** button instead of crashing.

### 4. Dashboard — `dashboard.py`
- Overview of attendance records: counts, trends, and charts.

### 5. Manage Students — bottom of `1_Register_Student.py`
- **Remove one student**: deletes their profile, face samples, attendance history, and saved photo. Requires a confirmation checkbox.
- **Danger zone — delete everything**: wipes all students, embeddings, attendance, and the trained model. Requires typing `DELETE ALL` to confirm.
- ⚠️ After removing students, retrain the model so it stops recognizing them.

---

## 🧰 CLI Utilities (`src/`)
```bash
python src/check_db.py           # List all registered students + sample counts
python src/check_attendance.py   # List all attendance records
```

---

## 🩹 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: classifier.pkl` | No model trained yet, or ephemeral disk lost it on restart | Register ≥2 students and click **Retrain Model Now** — model now persists to Turso, not disk |
| `ImportError: libGL.so.1` | `opencv-python` needs graphics libs not present on headless hosts | Use `opencv-python-headless` in `requirements.txt`, and/or add `packages.txt` with `libgl1` |
| `ModuleNotFoundError: No module named 'libsql'` | Missing/removed from `requirements.txt` | Ensure `libsql` and `python-dotenv` are listed, with a trailing newline in the file |
| App crashes / `healthz EOF` during training | Likely OOM on Streamlit Cloud's free tier (face model + parallel GridSearchCV workers) | Training uses `n_jobs=1` and a single lightweight KNN model to minimize memory use |
| Attendance not reflecting recent registrations | Turso embedded replica not synced | All reads call `conn.sync()` before querying; all writes call `conn.sync()` after committing |

---

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `TURSO_DATABASE_URL` | Your Turso database URL (`libsql://...`) |
| `TURSO_AUTH_TOKEN` | Auth token for the Turso database |

Set locally in `.env`, and in Streamlit Cloud under **Settings → Secrets** for deployment.
