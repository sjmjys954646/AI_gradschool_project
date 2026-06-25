from fastapi import FastAPI
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from database import get_db, init_db
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import torch

import sys
sys.path.append(r"C:\Users\ECL1\PycharmProjects\Capstone")

from main2_sisfallandmobifall import DSCS

model = None

WINDOW_SECONDS = 2
TARGET_FS = 60
WINDOW_SIZE = WINDOW_SECONDS * TARGET_FS  # 400

class SensorReadingItem(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    pressure: float
    measured_at: Optional[datetime] = None


class SensorBatch(BaseModel):
    user_id: int
    readings: List[SensorReadingItem]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")

@app.on_event("startup")
def startup():
    global model

    init_db()

    model = DSCS()

    model.load_state_dict(
        torch.load(
            r"C:\Users\ECL1\PycharmProjects\Capstone\aihub_cane_adult2_window_outputs2_02\aihub-group-5fold-5_best_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    print("Model loaded")


@app.get("/status")
def get_status(user_id: int):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT 
        u.id AS user_id,
        u.name,
        u.age,
        u.profile_image,
        u.status,

        cs.walking_speed,
        cs.walking_speed_status,
        cs.steps,
        cs.steps_goal,
        cs.heart_rate,
        cs.battery,
        cs.last_active_at,

        l.address
    FROM user u
    LEFT JOIN current_status cs ON u.id = cs.user_id
    LEFT JOIN location l ON u.id = l.user_id
    WHERE u.id = ?
    """, (user_id,))

    row = cur.fetchone()

    cur.execute("""
    SELECT *
    FROM alarm
    WHERE user_id = ?
      AND status = 'PENDING'
    ORDER BY detected_at DESC
    LIMIT 1
    """, (user_id,))

    alarm = cur.fetchone()

    if alarm:
        profile_status = "확인 필요"
    else:
        profile_status = "정상"

    conn.close()

    if row is None:
        return {"message": "user not found"}

    return {
        "profile": {
            "name": row["name"],
            "age": row["age"],
            "status": profile_status,
            "lastActive": row["last_active_at"],
            "image": row["profile_image"],
        },
        "metrics": {
            "walkingSpeed": {
                "value": row["walking_speed"],
                "unit": "m/s",
                "status": row["walking_speed_status"],
            },
            "steps": {
                "value": row["steps"],
                "unit": row["steps_goal"],
                "status": "",
            },
            "heartRate": {
                "value": row["heart_rate"],
                "unit": "bpm",
                "status": "정상",
            },
            "battery": {
                "value": row["battery"],
                "unit": "%",
                "status": "충전 중",
            },
        },
        "location": {
            "realtime": True,
            "mapText": row["address"] or "Google Map 영역",
        },
        "emergency": {
            "showFallAlert": alarm is not None,
        },
    }


@app.get("/fall")
def detect_fall(user_id: int):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM sensor_reading
        WHERE user_id = ?
        ORDER BY measured_at DESC
        LIMIT ?
        """, (user_id, WINDOW_SIZE))

    readings = list(reversed(cur.fetchall()))

    acc, gyro = build_model_input(readings)

    if acc is None or gyro is None:
        conn.close()

        return {
            "message": "not enough sensor data",
            "required_window": WINDOW_SIZE,
            "current_count": len(readings)
        }

    with torch.no_grad():
        logits = model(acc, gyro)

        probs = torch.softmax(logits, dim=1)

        confidence = float(probs[0][1])

        pred = int(torch.argmax(probs, dim=1).item())

        fall_detected = pred == 1

    conn.close()

    return {
        "user_id": user_id,

        "fall_detected": fall_detected,

        "confidence": confidence,

        "window_size": WINDOW_SIZE,

        "sensor_count": len(readings),

        "message": (
            "Fall detected"
            if fall_detected
            else "No fall detected"
        )
    }

@app.post("/save")
def save_sensor_batch(data: SensorBatch):
    conn = get_db()
    cur = conn.cursor()

    rows = []

    for r in data.readings:
        rows.append((
            data.user_id,
            r.accel_x,
            r.accel_y,
            r.accel_z,
            r.gyro_x,
            r.gyro_y,
            r.gyro_z,
            r.pressure,
            r.measured_at or datetime.now(),
            datetime.now()
        ))

    cur.executemany("""
    INSERT INTO sensor_reading (
        user_id,
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
        pressure,
        measured_at,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()

    cur.execute("""
    SELECT
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
        pressure,
        measured_at
    FROM sensor_reading
    WHERE user_id = ?
    ORDER BY measured_at DESC
    LIMIT ?
    """, (data.user_id, WINDOW_SIZE))

    readings = list(reversed(cur.fetchall()))

    acc, gyro = build_model_input(readings)

    fall_detected = False
    confidence = 0.0

    if acc is not None and gyro is not None:
        with torch.no_grad():
            logits = model(acc, gyro)
            probs = torch.softmax(logits, dim=1)

            confidence = float(probs[0][1])
            pred = int(torch.argmax(probs, dim=1).item())
            fall_detected = pred == 1

    if fall_detected:
        cur.execute("""
        INSERT INTO alarm (
            user_id,
            type,
            message,
            status,
            slice_start_time,
            slice_end_time,
            detected_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data.user_id,
            "FALL",
            "낙상 감지",
            "PENDING",
            readings[0]["measured_at"],
            readings[-1]["measured_at"],
            datetime.now()
        ))

        conn.commit()

    conn.close()

    return {
        "message": "sensor batch saved",
        "count": len(data.readings),
        "used_sensor_count": len(readings),
        "fall_detected": fall_detected,
        "confidence": confidence
    }

def build_model_input(readings):
    """
    readings: DB에서 가져온 최근 센서 데이터
    return:
      acc_tensor:  (1, 400, 3)
      gyro_tensor: (1, 400, 3)
    """

    if len(readings) < WINDOW_SIZE:
        return None, None

    recent = readings[-WINDOW_SIZE:]

    acc = []
    gyro = []

    for r in recent:
        acc.append([
            r["accel_x"],
            r["accel_y"],
            r["accel_z"],
        ])

        gyro.append([
            r["gyro_x"],
            r["gyro_y"],
            r["gyro_z"],
        ])

    acc = torch.tensor([acc], dtype=torch.float32)
    gyro = torch.tensor([gyro], dtype=torch.float32)

    return acc, gyro

@app.get("/alarm/latest")
def get_latest_alarm(user_id: int):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM alarm
    WHERE user_id = ?
      AND type = 'FALL'
      AND status = 'PENDING'
    ORDER BY detected_at DESC
    LIMIT 1
    """, (user_id,))

    alarm = cur.fetchone()
    conn.close()

    if alarm is None:
        return {
            "has_alarm": False
        }

    return {
        "has_alarm": True,
        "alarm_id": alarm["id"],
        "type": alarm["type"],
        "message": alarm["message"],
        "detected_at": alarm["detected_at"]
    }

@app.post("/alarm/{alarm_id}/check")
def check_alarm(alarm_id: int):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    UPDATE alarm
    SET status = 'CHECKED'
    WHERE id = ?
    """, (alarm_id,))

    conn.commit()
    conn.close()

    return {
        "message": "alarm checked",
        "alarm_id": alarm_id
    }

@app.post("/status/update")
def update_status(payload: dict):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    UPDATE current_status
    SET walking_speed = ?,
        steps = ?,
        heart_rate = ?,
        updated_at = ?
    WHERE user_id = ?
    """, (
        payload["walking_speed"],
        payload["steps"],
        payload["heart_rate"],
        datetime.now(),
        payload["user_id"]
    ))

    conn.commit()
    conn.close()

    return {
        "message": "status updated",
        "user_id": payload["user_id"],
        "walking_speed": payload["walking_speed"],
        "steps": payload["steps"],
        "heart_rate": payload["heart_rate"]
    }