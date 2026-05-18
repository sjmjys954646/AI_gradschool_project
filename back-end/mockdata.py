# seed_mock_data.py

import sqlite3
from datetime import datetime

DB_PATH = "fall_detection.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


# -----------------------------
# USER
# -----------------------------
cur.execute("""
INSERT INTO user (
    name,
    age,
    profile_image,
    status,
    created_at
)
VALUES (?, ?, ?, ?, ?)
""", (
    "김영희",
    78,
    "images/soonja.png",
    "정상",
    datetime.now()
))

user_id = cur.lastrowid


# -----------------------------
# CURRENT STATUS
# -----------------------------
cur.execute("""
INSERT INTO current_status (
    user_id,

    walking_speed,
    walking_speed_status,

    steps,
    steps_goal,

    heart_rate,

    battery,

    last_active_at,
    updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    user_id,

    0.8,
    "느림",

    3847,
    5000,

    72,

    78,

    "2분 전 활동",
    datetime.now()
))


# -----------------------------
# LOCATION
# -----------------------------
cur.execute("""
INSERT INTO location (
    user_id,
    latitude,
    longitude,
    address,
    updated_at
)
VALUES (?, ?, ?, ?, ?)
""", (
    user_id,
    35.1751,
    126.9086,
    "전남대학교 생활관 B동",
    datetime.now()
))


# -----------------------------
# OPTIONAL MOCK ALARM
# -----------------------------
cur.execute("""
INSERT INTO alarm (
    user_id,
    type,
    message,
    status,
    slice_start_time,
    slice_end_time,
    detected_at,
    latitude,
    longitude,
    address
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    user_id,
    "FALL",
    "낙상 감지",
    "PENDING",
    datetime.now(),
    datetime.now(),
    datetime.now(),
    35.1751,
    126.9086,
    "전남대학교 생활관 B동"
))

conn.commit()
conn.close()

print("Mock data inserted successfully.")