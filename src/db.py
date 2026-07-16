import os
import libsql
from dotenv import load_dotenv

load_dotenv()

TURSO_DATABASE_URL = os.environ["TURSO_DATABASE_URL"]
TURSO_AUTH_TOKEN = os.environ["TURSO_AUTH_TOKEN"]

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "attendance.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection():
    return libsql.connect(
        DB_PATH,
        sync_url=TURSO_DATABASE_URL,
        auth_token=TURSO_AUTH_TOKEN,
    )


def init_db():
    conn = get_connection()
    conn.sync()
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
    # Holds the trained classifier (pickled bytes) so it survives app restarts
    # even on ephemeral hosting like Streamlit Community Cloud.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_store (
            id INTEGER PRIMARY KEY,
            model_blob BLOB NOT NULL,
            model_type TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---- Migration: add photo_data BLOB column if it doesn't exist yet ----
    # Local disk (photo_path) is wiped on every Streamlit Cloud restart, so the
    # actual JPEG bytes need to live in Turso alongside everything else.
    cursor.execute("PRAGMA table_info(student_profiles)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "photo_data" not in existing_columns:
        cursor.execute("ALTER TABLE student_profiles ADD COLUMN photo_data BLOB")
        print("Migration: added photo_data column to student_profiles")

    conn.commit()
    conn.sync()
    conn.close()
    print("Database initialized (Turso):", TURSO_DATABASE_URL)


if __name__ == "__main__":
    init_db()