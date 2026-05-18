from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SensorData(BaseModel):
    user_id: int

    accel_x: float
    accel_y: float
    accel_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    pressure: float


class CurrentStatusUpdate(BaseModel):
    user_id: int

    walking_speed: float
    walking_speed_status: str

    steps: int
    steps_goal: int

    heart_rate: int
    battery: int

    last_active_at: Optional[datetime] = None


class AlarmCreate(BaseModel):
    user_id: int
    type: str
    message: str
    status: str

    slice_start_time: datetime
    slice_end_time: datetime

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None