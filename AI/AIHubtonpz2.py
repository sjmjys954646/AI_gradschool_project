from pathlib import Path
import json
import numpy as np
import pandas as pd

SENSOR_ROOT = Path(r"D:\fall_sensor_right_forearm\Training\01.원천데이터\TS\센서")
LABEL_ROOT  = Path(r"D:\fall_sensor_right_forearm\Training\02.라벨링데이터\TL\센서")

SAVE_PATH = "aihub_cane_adult2_window.npz"

ACC_COLS = [
    "Segment Acceleration_Right Forearm x",
    "Segment Acceleration_Right Forearm y",
    "Segment Acceleration_Right Forearm z",
]

GYRO_COLS = [
    "Segment Angular Velocity_Right Forearm x",
    "Segment Angular Velocity_Right Forearm y",
    "Segment Angular Velocity_Right Forearm z",
]

WINDOW_SIZE = 120   # 60Hz 기준 2초
STRIDE = 30         # 60Hz 기준 0.5초
FALL_OVERLAP_RATIO = 0.3


def overlap_ratio(start, end, fall_start, fall_end):
    overlap_start = max(start, fall_start)
    overlap_end = min(end, fall_end)
    overlap = max(0, overlap_end - overlap_start)
    return overlap / (end - start)


acc_list = []
gyro_list = []
labels = []
subject_ids = []
scene_ids = []
window_starts = []
window_ends = []

json_files = list(LABEL_ROOT.rglob("*.json"))

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    age = meta.get("actor_info", {}).get("actor_age", "")
    method = meta.get("scene_info", {}).get("scene_method", "")
    is_fall = meta.get("scene_info", {}).get("scene_IsFall", "")

    if "adult2" not in age:
        continue
    if method != "cane":
        continue

    scene_id = meta["metadata"]["scene_id"]
    subject_id = scene_id.split("_")[0]

    rel_parent = json_path.parent.relative_to(LABEL_ROOT)
    csv_path = SENSOR_ROOT / rel_parent / f"{scene_id}.csv"

    if not csv_path.exists():
        print("[CSV 없음]", csv_path)
        continue

    df = pd.read_csv(csv_path)

    missing = [c for c in ACC_COLS + GYRO_COLS if c not in df.columns]
    if missing:
        print("[컬럼 없음]", csv_path, missing)
        continue

    acc_all = df[ACC_COLS].to_numpy(dtype=np.float32)
    gyro_all = df[GYRO_COLS].to_numpy(dtype=np.float32)

    n = len(df)

    if is_fall == "낙상":
        fall_start = int(meta["sensordata"]["fall_start_frame"])
        fall_end = int(meta["sensordata"]["fall_end_frame"])
    else:
        fall_start = None
        fall_end = None

    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE

        if is_fall == "낙상":
            ratio = overlap_ratio(start, end, fall_start, fall_end)

            if ratio >= FALL_OVERLAP_RATIO:
                label = 1
            else:
                continue
        else:
            label = 0

        acc_win = acc_all[start:end]
        gyro_win = gyro_all[start:end]

        acc_list.append(acc_win)
        gyro_list.append(gyro_win)
        labels.append(label)
        subject_ids.append(subject_id)
        scene_ids.append(scene_id)
        window_starts.append(start)
        window_ends.append(end)

acc_arr = np.stack(acc_list)
gyro_arr = np.stack(gyro_list)
labels_arr = np.array(labels, dtype=np.int64)
subject_ids_arr = np.array(subject_ids)
scene_ids_arr = np.array(scene_ids)
window_starts_arr = np.array(window_starts, dtype=np.int64)
window_ends_arr = np.array(window_ends, dtype=np.int64)

np.savez(
    SAVE_PATH,
    acc=acc_arr,
    gyro=gyro_arr,
    labels=labels_arr,
    subject_ids=subject_ids_arr,
    scene_ids=scene_ids_arr,
    window_starts=window_starts_arr,
    window_ends=window_ends_arr,
)

print("Saved:", SAVE_PATH)
print("acc:", acc_arr.shape)
print("gyro:", gyro_arr.shape)
print("labels:", labels_arr.shape)
print("Fall windows:", int((labels_arr == 1).sum()))
print("Non-fall windows:", int((labels_arr == 0).sum()))
print("Subjects:", len(np.unique(subject_ids_arr)))