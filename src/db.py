import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "attendance.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # One row per student — profile info only
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profiles (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            photo_path TEXT,
            registered_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Multiple rows per student — one per captured pose
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
    print("Database initialized at:", DB_PATH)

if __name__ == "__main__":
    init_db()