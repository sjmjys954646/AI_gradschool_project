from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI()


# 센서 데이터 요청 형식
class SensorData(BaseModel):
    user_id: int

    accel_x: float
    accel_y: float
    accel_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    pressure: float

    measured_at: Optional[datetime] = None


# 현재 상태 응답 형식
class CurrentStatus(BaseModel):
    user_id: int
    name: str
    age: int

    walking_speed: float
    walking_speed_status: str

    steps: int
    step_goal: int

    heart_rate: int
    heart_rate_status: str

    battery: int
    battery_status: str

    fall_detected: bool
    last_active: str


@app.post("/save")
def save_current_information(data: SensorData):
    """
    센서 데이터를 DB에 저장
    """
    return {
        "message": "Sensor data saved",
        "data": data
    }


@app.get("/fall")
def detect_fall(user_id: int):
    """
    저장된 센서 데이터를 기반으로 낙상 감지 모델 실행
    """
    return {
        "user_id": user_id,
        "fall_detected": False,
        "confidence": 0.12,
        "message": "No fall detected"
    }


@app.get("/status")
def get_current_status(user_id: int):
    """
    현재 사용자 상태 및 프론트 표시 정보 반환
    """
    return {
        "user_id": user_id,
        "name": "김영희",
        "age": 78,

        "walking_speed": 0.8,
        "walking_speed_status": "느림",

        "steps": 3847,
        "step_goal": 5000,

        "heart_rate": 72,
        "heart_rate_status": "정상",

        "battery": 78,
        "battery_status": "충전 중",

        "fall_detected": False,
        "last_active": "2분 전 활동"
    }