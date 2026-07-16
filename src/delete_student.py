import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from db import get_connection


def delete_by_name(name):
    conn = get_connection()
    conn.sync()
    cursor = conn.cursor()

    # Find matching student(s) first, so we know exactly what we're deleting
    cursor.execute("SELECT student_id, name, photo_path FROM student_profiles WHERE name = ?", (name,))
    matches = cursor.fetchall()

    if not matches:
        print(f"❌ No student found with name '{name}'.")
        conn.close()
        return

    for student_id, matched_name, photo_path in matches:
        print(f"Deleting {matched_name} ({student_id})...")
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM student_profiles WHERE student_id = ?", (student_id,))

        if photo_path and isinstance(photo_path, str) and os.path.exists(photo_path):
            try:
                os.remove(photo_path)
                print(f"  Removed photo file: {photo_path}")
            except Exception as e:
                print(f"  ⚠️ Could not remove photo file (non-fatal): {e}")

    conn.commit()
    conn.sync()
    conn.close()
    print(f"✅ Deleted {len(matches)} record(s) for name '{name}'.")
    print("⚠️ Remember to retrain the model so it forgets this student too.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
    else:
        target_name = input("Enter the student name to delete: ").strip()

    delete_by_name(target_name)