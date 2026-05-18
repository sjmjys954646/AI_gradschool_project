import sqlite3

DB_PATH = "fall_detection.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        profile_image TEXT,
        status TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS current_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        walking_speed REAL,
        walking_speed_status TEXT,
        steps INTEGER,
        steps_goal INTEGER,
        heart_rate INTEGER,
        battery INTEGER,
        last_active_at DATETIME,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_reading (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        accel_x REAL,
        accel_y REAL,
        accel_z REAL,
        gyro_x REAL,
        gyro_y REAL,
        gyro_z REAL,
        pressure REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS location (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        latitude REAL,
        longitude REAL,
        address TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS alarm (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        type TEXT,
        message TEXT,
        status TEXT,
        slice_start_time DATETIME,
        slice_end_time DATETIME,
        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        confirmed_at DATETIME,
        latitude REAL,
        longitude REAL,
        address TEXT,
        FOREIGN KEY (user_id) REFERENCES user(id)
    )
    """)

    conn.commit()
    conn.close()