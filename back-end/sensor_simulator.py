import time
import random
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000/save"

SAMPLE_RATE = 1 / 50
BATCH_SIZE = 50    

while True:
    readings = []

    for _ in range(BATCH_SIZE):
        reading = {
            "accel_x": random.uniform(-0.2, 0.2),
            "accel_y": random.uniform(-0.2, 0.2),
            "accel_z": random.uniform(9.4, 10.1),

            "gyro_x": random.uniform(-0.05, 0.05),
            "gyro_y": random.uniform(-0.05, 0.05),
            "gyro_z": random.uniform(-0.05, 0.05),

            "pressure": random.uniform(60, 80),

            "measured_at": datetime.now().isoformat()
        }

        readings.append(reading)

        time.sleep(SAMPLE_RATE)

    payload = {
        "user_id": 1,
        "readings": readings
    }

    response = requests.post(API_URL, json=payload)

    print("sent batch:", len(readings))
    print(response.json())