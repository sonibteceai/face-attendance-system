import numpy as np
from db import get_connection

def check_students():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sp.student_id, sp.name, COUNT(fe.id) as sample_count
        FROM student_profiles sp
        LEFT JOIN face_embeddings fe ON sp.student_id = fe.student_id
        GROUP BY sp.student_id
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No students found in database.")
        return

    print(f"Found {len(rows)} student(s):\n")
    for student_id, name, sample_count in rows:
        print(f"Student ID : {student_id}  Name: {name}  Samples: {sample_count}")

if __name__ == "__main__":
    check_students()