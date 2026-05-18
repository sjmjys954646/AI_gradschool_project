from fastapi import FastAPI
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from database import get_db, init_db
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")

@app.on_event("startup")
def startup():
    init_db()


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
    conn.close()

    if row is None:
        return {"message": "user not found"}

    return {
        "profile": {
            "name": row["name"],
            "age": row["age"],
            "status": row["status"],
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
    ORDER BY created_at DESC
    LIMIT 100
    """, (user_id,))

    readings = cur.fetchall()

    # TODO: 여기에 slicing + model inference 연결
    fall_detected = False
    confidence = 0.12

    if fall_detected and readings:
        slice_start_time = readings[-1]["created_at"]
        slice_end_time = readings[0]["created_at"]

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
            user_id,
            "FALL",
            "낙상 감지",
            "PENDING",
            slice_start_time,
            slice_end_time,
            datetime.now()
        ))

        conn.commit()

    conn.close()

    return {
        "user_id": user_id,
        "fall_detected": fall_detected,
        "confidence": confidence,
        "message": "Fall detected" if fall_detected else "No fall detected",
    }