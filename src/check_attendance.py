from db import get_connection

def check_attendance():
    conn = get_connection()
    conn.sync()  # pull latest data from Turso before reading
    cursor = conn.cursor()
    cursor.execute("""
        SELECT student_id, name, timestamp, status 
        FROM attendance 
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No attendance records found.")
        return

    print(f"Found {len(rows)} attendance record(s):\n")
    for student_id, name, timestamp, status in rows:
        print(f"{timestamp}  |  {name} ({student_id})  |  {status}")

if __name__ == "__main__":
    check_attendance()