import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from db import get_connection, init_db

init_db()

st.set_page_config(page_title="Attendance Dashboard", layout="wide")


def read_sql_df(conn, query):
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        cur = conn.cursor()
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


@st.cache_data(ttl=5)  # refresh cache every 5 seconds so new attendance shows up
def load_data():
    conn = get_connection()
    conn.sync()

    attendance_df = read_sql_df(
        conn,
        "SELECT student_id, name, timestamp, status FROM attendance ORDER BY timestamp DESC"
    )
    students_df = read_sql_df(
        conn,
        "SELECT student_id, name, photo_path, registered_on FROM student_profiles"
    )
    conn.close()

    if not attendance_df.empty:
        # SQLite CURRENT_TIMESTAMP is stored in UTC — convert to IST (UTC+5:30)
        attendance_df["timestamp"] = pd.to_datetime(attendance_df["timestamp"], utc=True)
        attendance_df["timestamp"] = attendance_df["timestamp"].dt.tz_convert("Asia/Kolkata")
        attendance_df["date"] = attendance_df["timestamp"].dt.date
        attendance_df["time"] = attendance_df["timestamp"].dt.strftime("%I:%M:%S %p")
        attendance_df["date_display"] = attendance_df["timestamp"].dt.strftime("%d-%m-%Y")
        attendance_df["year"] = attendance_df["timestamp"].dt.year
        attendance_df["month"] = attendance_df["timestamp"].dt.month
        attendance_df["month_name"] = attendance_df["timestamp"].dt.strftime("%b %Y")
        attendance_df["year_month"] = attendance_df["timestamp"].dt.to_period("M").astype(str)

    if not students_df.empty and "registered_on" in students_df:
        students_df["registered_on"] = pd.to_datetime(students_df["registered_on"], utc=True, errors="coerce")
        students_df["registered_on"] = students_df["registered_on"].dt.tz_convert("Asia/Kolkata")

    return attendance_df, students_df


st.title("📋 Attendance Dashboard")

attendance_df, students_df = load_data()

if attendance_df.empty:
    st.warning("No attendance records yet. Run mark_attendance.py to start logging.")
    st.stop()

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")

min_date = attendance_df["date"].min()
max_date = attendance_df["date"].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

all_students = sorted(attendance_df["name"].unique())
selected_students = st.sidebar.multiselect(
    "Students", options=all_students, default=all_students
)

# Apply filters
filtered = attendance_df.copy()
if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
filtered = filtered[filtered["name"].isin(selected_students)]

# -------------------------------
# Top metrics
# -------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered))
col2.metric("Unique Students Present", filtered["student_id"].nunique())
col3.metric("Total Registered Students", len(students_df))

st.divider()

# -------------------------------
# Daily attendance chart
# -------------------------------
st.subheader("Attendance Over Time")
daily_counts = filtered.groupby("date")["student_id"].nunique().reset_index()
daily_counts.columns = ["date", "students_present"]
daily_counts["date"] = pd.to_datetime(daily_counts["date"])
daily_counts = daily_counts.sort_values("date")

if not daily_counts.empty:
    fig = px.line(daily_counts, x="date", y="students_present",
                  title="Unique Students Present Per Day",
                  markers=True, text="students_present")
    fig.update_traces(textposition="top center", line=dict(width=2), marker=dict(size=8))
    fig.update_xaxes(
        title="Date",
        tickformat="%d-%m-%Y",
        type="date",
    )
    fig.update_yaxes(title="Students Present", dtick=1)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data for selected filters.")

st.divider()

# -------------------------------
# Today's attendance (quick view)
# -------------------------------
st.subheader(f"Today's Attendance ({date.today()})")
today_df = attendance_df[attendance_df["date"] == date.today()]

if not today_df.empty:
    present_ids = set(today_df["student_id"])
    absent_df = students_df[~students_df["student_id"].isin(present_ids)]

    c1, c2 = st.columns(2)
    with c1:
        st.success(f"✅ Present ({len(today_df)})")
        st.dataframe(today_df[["name", "time", "status"]], use_container_width=True, hide_index=True)
    with c2:
        st.error(f"❌ Absent ({len(absent_df)})")
        st.dataframe(absent_df[["name"]], use_container_width=True, hide_index=True)
else:
    st.info("No attendance marked yet today.")

st.divider()

# -------------------------------
# Full filtered table + export
# -------------------------------
st.subheader("All Records")
display_df = filtered[["date_display", "time", "name", "student_id", "status"]].sort_values(
    ["date_display", "time"], ascending=False
)
display_df = display_df.rename(columns={"date_display": "date"})
st.dataframe(display_df, use_container_width=True, hide_index=True)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered table as CSV",
    data=csv,
    file_name=f"attendance_filtered_{date.today()}.csv",
    mime="text/csv",
)

st.divider()

# -------------------------------
# Reports: Per-Day and Per-Student
# -------------------------------
st.subheader("📑 Reports")

report_tab1, report_tab2 = st.tabs(["Per-Day Report", "Per-Student Report"])

# ---- Per-Day Report ----
with report_tab1:
    st.markdown("Get a full Present/Absent report for **one specific day**, covering every registered student.")

    report_date = st.date_input(
        "Select date", value=date.today(), min_value=min_date, max_value=max_date, key="day_report_date"
    )

    day_df = attendance_df[attendance_df["date"] == report_date]
    present_ids = set(day_df["student_id"])

    day_report = students_df.copy()
    day_report["status"] = day_report["student_id"].apply(
        lambda sid: "Present" if sid in present_ids else "Absent"
    )
    day_report["time"] = day_report["student_id"].apply(
        lambda sid: day_df[day_df["student_id"] == sid]["time"].values[0] if sid in present_ids else "-"
    )
    day_report = day_report[["student_id", "name", "status", "time"]]

    st.dataframe(day_report, use_container_width=True, hide_index=True)

    p_count = (day_report["status"] == "Present").sum()
    a_count = (day_report["status"] == "Absent").sum()
    st.caption(f"Present: {p_count}  |  Absent: {a_count}")

    day_csv = day_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download report for {report_date}",
        data=day_csv,
        file_name=f"attendance_{report_date}.csv",
        mime="text/csv",
        key="day_download",
    )

# ---- Per-Student Report ----
with report_tab2:
    st.markdown("Get the **full attendance history** for one student across all recorded days.")

    student_names = sorted(students_df["name"].unique())
    selected_student = st.selectbox("Select student", options=student_names, key="student_report_select")

    student_row = students_df[students_df["name"] == selected_student].iloc[0]
    student_id = student_row["student_id"]
    photo_path = student_row.get("photo_path")

    student_history = attendance_df[attendance_df["student_id"] == student_id][
        ["date_display", "time", "status"]
    ].sort_values("date_display", ascending=False)
    student_history = student_history.rename(columns={"date_display": "date"})

    all_days = pd.date_range(min_date, max_date, freq="D").date
    present_days = set(student_history["date"])
    total_days = len(all_days)
    present_count = len(present_days)
    attendance_pct = (present_count / total_days * 100) if total_days > 0 else 0

    photo_col, metric_col = st.columns([1, 3])
    with photo_col:
        if photo_path and os.path.exists(photo_path):
            st.image(photo_path, caption=selected_student, width=150)
        else:
            st.info("No photo on file")
    with metric_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("Days Present", present_count)
        m2.metric("Total Days Tracked", total_days)
        m3.metric("Attendance %", f"{attendance_pct:.1f}%")

    st.dataframe(student_history, use_container_width=True, hide_index=True)

    student_csv = student_history.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download {selected_student}'s attendance report",
        data=student_csv,
        file_name=f"attendance_{selected_student}_{student_id}.csv",
        mime="text/csv",
        key="student_download",
    )

st.divider()

# =========================================================
# 📈 ANALYTICS
# =========================================================
st.header("📈 Analytics")

analytics_tabs = st.tabs([
    "Trends (Day/Month/Year)",
    "Student Comparison",
    "Distribution",
    "Per-Student Calendar",
    "Registrations",
    "Period Comparison",
])

# ---------------------------------------------------------
# TAB 1: Day-wise / Month-wise / Year-wise attendance trend
# ---------------------------------------------------------
with analytics_tabs[0]:
    st.markdown("Total unique students present, grouped by day, month, or year.")
    granularity = st.radio("Group by", ["Day", "Month", "Year"], horizontal=True, key="trend_granularity")

    if granularity == "Day":
        trend_df = attendance_df.groupby("date")["student_id"].nunique().reset_index()
        trend_df.columns = ["Period", "Students Present"]
        trend_df["Period"] = pd.to_datetime(trend_df["Period"])
        trend_df = trend_df.sort_values("Period")
        fig = px.line(trend_df, x="Period", y="Students Present", title="Day-wise Attendance",
                      markers=True, text="Students Present")
        fig.update_traces(textposition="top center", line=dict(width=2), marker=dict(size=8))
        fig.update_xaxes(tickformat="%d-%m-%Y", type="date")
        fig.update_yaxes(dtick=1)
    elif granularity == "Month":
        trend_df = attendance_df.groupby("year_month")["student_id"].nunique().reset_index()
        trend_df.columns = ["Period", "Students Present"]
        trend_df = trend_df.sort_values("Period")
        fig = px.line(trend_df, x="Period", y="Students Present", title="Month-wise Attendance",
                      markers=True, text="Students Present")
        fig.update_traces(textposition="top center", line=dict(width=2), marker=dict(size=8))
        fig.update_yaxes(dtick=1)
    else:
        trend_df = attendance_df.groupby("year")["student_id"].nunique().reset_index()
        trend_df.columns = ["Period", "Students Present"]
        trend_df["Period"] = trend_df["Period"].astype(str)
        trend_df = trend_df.sort_values("Period")
        fig = px.line(trend_df, x="Period", y="Students Present", title="Year-wise Attendance",
                      markers=True, text="Students Present")
        fig.update_traces(textposition="top center", line=dict(width=2), marker=dict(size=10))
        fig.update_yaxes(dtick=1)

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Compare students — monthly & annual bar charts
# ---------------------------------------------------------
with analytics_tabs[1]:
    st.markdown("Compare how many days each student was present, for a chosen month or year.")
    compare_mode = st.radio("Compare by", ["Month", "Year"], horizontal=True, key="compare_mode")

    if compare_mode == "Month":
        month_options = sorted(attendance_df["year_month"].unique(), reverse=True)
        selected_period = st.selectbox("Select month", month_options, key="compare_month_select")
        period_df = attendance_df[attendance_df["year_month"] == selected_period]
    else:
        year_options = sorted(attendance_df["year"].unique(), reverse=True)
        selected_period = st.selectbox("Select year", year_options, key="compare_year_select")
        period_df = attendance_df[attendance_df["year"] == selected_period]

    student_days = period_df.groupby("name")["date"].nunique().reset_index()
    student_days.columns = ["Student", "Days Present"]
    student_days = student_days.sort_values("Days Present", ascending=False)

    if not student_days.empty:
        fig = px.bar(student_days, x="Student", y="Days Present",
                     title=f"Days Present — {selected_period}", color="Student")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attendance data for this period.")

# ---------------------------------------------------------
# TAB 3: Pie charts — present vs absent for a month/year
# ---------------------------------------------------------
with analytics_tabs[2]:
    st.markdown("Overall Present vs Absent split (across all students) for a chosen month or year.")
    dist_mode = st.radio("View by", ["Month", "Year"], horizontal=True, key="dist_mode")

    total_students = len(students_df)

    if dist_mode == "Month":
        month_options = sorted(attendance_df["year_month"].unique(), reverse=True)
        selected_period = st.selectbox("Select month", month_options, key="dist_month_select")
        period_df = attendance_df[attendance_df["year_month"] == selected_period]
        days_in_period = period_df["date"].nunique()
    else:
        year_options = sorted(attendance_df["year"].unique(), reverse=True)
        selected_period = st.selectbox("Select year", year_options, key="dist_year_select")
        period_df = attendance_df[attendance_df["year"] == selected_period]
        days_in_period = period_df["date"].nunique()

    possible_slots = total_students * days_in_period
    present_slots = len(period_df)
    absent_slots = max(possible_slots - present_slots, 0)

    pie_df = pd.DataFrame({
        "Status": ["Present", "Absent"],
        "Count": [present_slots, absent_slots],
    })
    fig = px.pie(pie_df, names="Status", values="Count",
                 title=f"Present vs Absent — {selected_period}",
                 color="Status", color_discrete_map={"Present": "#2ecc71", "Absent": "#e74c3c"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Absent is estimated as (registered students × active attendance days) − present records.")

# ---------------------------------------------------------
# TAB 4: Per-student per-day calendar bar chart
# ---------------------------------------------------------
with analytics_tabs[3]:
    st.markdown("Daily Present/Absent pattern for one student in a chosen month.")

    cal_student = st.selectbox("Select student", sorted(students_df["name"].unique()), key="cal_student_select")
    cal_month_options = sorted(attendance_df["year_month"].unique(), reverse=True)
    cal_month = st.selectbox("Select month", cal_month_options, key="cal_month_select")

    cal_student_id = students_df[students_df["name"] == cal_student]["student_id"].values[0]
    year_num, month_num = map(int, cal_month.split("-"))
    days_in_month = pd.Period(f"{year_num}-{month_num}").days_in_month
    month_days = pd.date_range(f"{year_num}-{month_num}-01", periods=days_in_month, freq="D").date

    present_days_set = set(
        attendance_df[
            (attendance_df["student_id"] == cal_student_id) & (attendance_df["year_month"] == cal_month)
        ]["date"]
    )

    cal_df = pd.DataFrame({
        "Day": [d.day for d in month_days],
        "Status": ["Present" if d in present_days_set else "Absent" for d in month_days],
    })
    fig = px.bar(cal_df, x="Day", y=[1] * len(cal_df), color="Status",
                 color_discrete_map={"Present": "#2ecc71", "Absent": "#e74c3c"},
                 title=f"{cal_student} — {cal_month}")
    fig.update_yaxes(visible=False, title=None)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: Registration growth (line graph)
# ---------------------------------------------------------
with analytics_tabs[4]:
    st.markdown("Cumulative student registrations over time.")

    if "registered_on" in students_df and students_df["registered_on"].notna().any():
        reg_df = students_df.dropna(subset=["registered_on"]).copy()
        reg_df["reg_date"] = reg_df["registered_on"].dt.date
        reg_counts = reg_df.groupby("reg_date").size().reset_index(name="new_registrations")
        reg_counts = reg_counts.sort_values("reg_date")
        reg_counts["cumulative"] = reg_counts["new_registrations"].cumsum()

        fig = px.line(reg_counts, x="reg_date", y="cumulative", markers=True,
                      title="Cumulative Registrations Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No registration date data available.")

# ---------------------------------------------------------
# TAB 6: Last month vs this month / last year vs this year
# ---------------------------------------------------------
with analytics_tabs[5]:
    st.markdown("Compare attendance activity between consecutive periods.")
    period_type = st.radio("Compare", ["This Month vs Last Month", "This Year vs Last Year"],
                            horizontal=True, key="period_compare_type")

    today = pd.Timestamp.now(tz="Asia/Kolkata")

    if period_type == "This Month vs Last Month":
        this_period = today.strftime("%Y-%m")
        last_period_ts = (today.replace(day=1) - pd.Timedelta(days=1))
        last_period = last_period_ts.strftime("%Y-%m")
        this_df = attendance_df[attendance_df["year_month"] == this_period]
        last_df = attendance_df[attendance_df["year_month"] == last_period]
        this_label, last_label = this_period, last_period
    else:
        this_period = today.year
        last_period = today.year - 1
        this_df = attendance_df[attendance_df["year"] == this_period]
        last_df = attendance_df[attendance_df["year"] == last_period]
        this_label, last_label = str(this_period), str(last_period)

    compare_data = pd.DataFrame({
        "Period": [last_label, this_label],
        "Total Records": [len(last_df), len(this_df)],
        "Unique Students": [last_df["student_id"].nunique(), this_df["student_id"].nunique()],
        "Active Days": [last_df["date"].nunique(), this_df["date"].nunique()],
    })

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", compare_data["Total Records"].iloc[1],
              delta=int(compare_data["Total Records"].iloc[1] - compare_data["Total Records"].iloc[0]))
    m2.metric("Unique Students", compare_data["Unique Students"].iloc[1],
              delta=int(compare_data["Unique Students"].iloc[1] - compare_data["Unique Students"].iloc[0]))
    m3.metric("Active Days", compare_data["Active Days"].iloc[1],
              delta=int(compare_data["Active Days"].iloc[1] - compare_data["Active Days"].iloc[0]))

    melted = compare_data.melt(id_vars="Period", var_name="Metric", value_name="Value")
    fig = px.bar(melted, x="Metric", y="Value", color="Period", barmode="group",
                 title=f"{last_label} vs {this_label}")
    st.plotly_chart(fig, use_container_width=True)