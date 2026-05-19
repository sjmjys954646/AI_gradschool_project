import sqlite3
import numpy as np
from datetime import datetime, timedelta

DB_PATH = "fall_detection.db"
NPZ_PATH = r"C:\Users\ECL1\PycharmProjects\Capstone\sisfall_data_8sec.npz"

USER_ID = 1
TARGET_LABEL = 0  # 1 = fall, 0 = ADL
SAMPLE_INDEX = None

TARGET_FS = 50
WINDOW_SECONDS = 8
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS  # 400


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)

    acc_data = data["acc"]
    gyro_data = data["gyro"]
    labels = data["labels"]

    print("acc shape:", acc_data.shape)
    print("gyro shape:", gyro_data.shape)
    print("labels shape:", labels.shape)

    if SAMPLE_INDEX is None:
        candidates = np.where(labels == TARGET_LABEL)[0]

        if len(candidates) == 0:
            raise ValueError(f"label={TARGET_LABEL} 샘플이 없습니다.")

        idx = int(candidates[0])
    else:
        idx = SAMPLE_INDEX

    acc = acc_data[idx]
    gyro = gyro_data[idx]
    label = int(labels[idx])

    if len(acc) != WINDOW_SIZE:
        raise ValueError(
            f"현재 샘플 길이={len(acc)}입니다. 모델 입력 길이 {WINDOW_SIZE}와 다릅니다."
        )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    DELETE FROM sensor_reading
    WHERE user_id = ?
    """, (USER_ID,))

    start_time = datetime.now() - timedelta(seconds=WINDOW_SECONDS)

    rows = []

    for i in range(WINDOW_SIZE):
        measured_at = start_time + timedelta(seconds=i / TARGET_FS)

        rows.append((
            USER_ID,
            float(acc[i][0]),
            float(acc[i][1]),
            float(acc[i][2]),
            float(gyro[i][0]),
            float(gyro[i][1]),
            float(gyro[i][2]),
            70.0,
            measured_at.isoformat(),
            datetime.now().isoformat()
        ))

    cur.executemany("""
    INSERT INTO sensor_reading (
        user_id,
        accel_x,
        accel_y,
        accel_z,
        gyro_x,
        gyro_y,
        gyro_z,
        pressure,
        measured_at,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()

    print(f"Inserted validation sample idx={idx}, label={label}")
    print(f"Inserted rows: {len(rows)}")


if __name__ == "__main__":
    main()