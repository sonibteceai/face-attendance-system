import os
import shutil
from db import get_connection, DB_PATH

PHOTO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "student_faces")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")

def reset_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM student_profiles")
    cursor.execute("DELETE FROM face_embeddings")
    cursor.execute("DELETE FROM attendance")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('face_embeddings', 'attendance')")

    conn.commit()
    conn.close()
    print("✅ Cleared all tables.")





def reset_photos():
    if os.path.exists(PHOTO_DIR):
        shutil.rmtree(PHOTO_DIR)
        os.makedirs(PHOTO_DIR, exist_ok=True)
        print("✅ Cleared student photo folder.")
    else:
        os.makedirs(PHOTO_DIR, exist_ok=True)


def reset_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("✅ Removed old classifier.pkl.")


if __name__ == "__main__":
    confirm = input("This will DELETE all students, photos, and the trained model. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        reset_database()
        reset_photos()
        reset_model()
        print("\n🎉 Database reset complete. Ready for fresh registrations.")
    else:
        print("Cancelled. Nothing was deleted.")