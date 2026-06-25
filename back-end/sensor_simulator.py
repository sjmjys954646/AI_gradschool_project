import time
import random
import requests
import numpy as np
from datetime import datetime

API_URL = "http://127.0.0.1:8000/save"
STATUS_API_URL = "http://127.0.0.1:8000/status/update"

walking_speed = 0.8
heart_rate = 72
steps = 3847
NPZ_PATH = r"C:\Users\ECL1\PycharmProjects\Capstone\aihub_cane_adult2_window_val.npz"

USER_ID = 1

TARGET_FS = 60
SAMPLE_INTERVAL = 1 / TARGET_FS

# 몇 개의 비낙상 window 후 낙상 발생
NORMAL_WINDOWS_BEFORE_FALL = 10
NORMAL_WINDOWS_AFTER_FALL = 3

# 한 번에 보내는 센서 개수
# 120이면 2초 단위 전송
BATCH_SIZE = 120


def load_npz():
    data = np.load(NPZ_PATH, allow_pickle=True)

    acc = data["acc"]
    gyro = data["gyro"]
    labels = data["labels"]

    fall_indices = np.where(labels == 1)[0]
    nonfall_indices = np.where(labels == 0)[0]

    print("acc:", acc.shape)
    print("gyro:", gyro.shape)
    print("labels:", labels.shape)
    print("fall samples:", len(fall_indices))
    print("nonfall samples:", len(nonfall_indices))

    return acc, gyro, labels, fall_indices, nonfall_indices


def make_reading(acc_row, gyro_row):
    return {
        "accel_x": float(acc_row[0]),
        "accel_y": float(acc_row[1]),
        "accel_z": float(acc_row[2]),

        "gyro_x": float(gyro_row[0]),
        "gyro_y": float(gyro_row[1]),
        "gyro_z": float(gyro_row[2]),

        "pressure": random.uniform(60, 80),
        "measured_at": datetime.now().isoformat()
    }


def send_window(acc_window, gyro_window, label, idx):
    readings = []

    for i in range(len(acc_window)):
        readings.append(make_reading(acc_window[i], gyro_window[i]))
        time.sleep(SAMPLE_INTERVAL)

    send_status_update(is_fall=(label == 1))

    payload = {
        "user_id": USER_ID,
        "readings": readings
    }

    response = requests.post(API_URL, json=payload, timeout=10)

    print("=" * 60)
    print(f"sent idx={idx}, true_label={label}, rows={len(readings)}")
    print("response:", response.json())


def play_random_window(acc, gyro, labels, indices, label_name):
    idx = int(random.choice(indices))

    acc_window = acc[idx]
    gyro_window = gyro[idx]
    label = int(labels[idx])

    print(f"[{label_name}] playing sample idx={idx}")
    send_window(acc_window, gyro_window, label, idx)


def main():
    acc, gyro, labels, fall_indices, nonfall_indices = load_npz()

    while True:
        print("\nNORMAL WALKING PHASE")

        for _ in range(NORMAL_WINDOWS_BEFORE_FALL):
            play_random_window(
                acc,
                gyro,
                labels,
                nonfall_indices,
                "NON-FALL"
            )
            time.sleep(0.5)

        print("\nFALL EVENT PHASE")

        idx = 137
        acc_window = acc[idx]
        gyro_window = gyro[idx]
        label = int(labels[idx])

        print(f"[FALL] playing sample idx={idx}")
        send_window(acc_window, gyro_window, label, idx)

        # play_random_window(
        #     acc,
        #     gyro,
        #     labels,
        #     fall_indices,
        #     "FALL"
        # )

        # print("\nRECOVERY / NORMAL PHASE")

        # for _ in range(NORMAL_WINDOWS_AFTER_FALL):
        #     play_random_window(
        #         acc,
        #         gyro,
        #         labels,
        #         nonfall_indices,
        #         "NON-FALL"
        #     )
        #     time.sleep(0.5)

        print("\nSCENARIO FINISHED. RESTARTING...")
        time.sleep(3)
        break

def send_status_update(is_fall=False):
    global walking_speed, heart_rate, steps

    if is_fall:
        walking_speed = 0.0
        heart_rate = random.randint(95, 115)
    else:
        walking_speed += random.uniform(-0.05, 0.05)
        walking_speed = max(0.5, min(1.1, walking_speed))

        heart_rate += random.randint(-2, 2)
        heart_rate = max(68, min(82, heart_rate))

        steps += random.randint(1, 4)

    payload = {
        "user_id": USER_ID,
        "walking_speed": round(walking_speed, 2),
        "heart_rate": heart_rate,
        "steps": steps
    }

    try:
        response = requests.post(STATUS_API_URL, json=payload, timeout=3)
        print("[STATUS UPDATE]", response.status_code, payload, response.text)
    except Exception as e:
        print("[STATUS API ERROR]", e)


if __name__ == "__main__":
    main()