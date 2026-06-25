from pathlib import Path
import json
import numpy as np
import pandas as pd

SENSOR_ROOT = Path(r"D:\fall_sensor_right_forearm\Training\01.원천데이터\TS\센서")
LABEL_ROOT  = Path(r"D:\fall_sensor_right_forearm\Training\02.라벨링데이터\TL\센서")

SAVE_PATH = "aihub_cane_adult2_clip600.npz"

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

TARGET_LEN = 600


def pad_or_cut(arr, target_len=600):
    if len(arr) >= target_len:
        return arr[:target_len]
    pad = np.zeros((target_len - len(arr), arr.shape[1]), dtype=arr.dtype)
    return np.vstack([arr, pad])


acc_list = []
gyro_list = []
labels = []
subject_ids = []
scene_ids = []

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

    acc = df[ACC_COLS].to_numpy(dtype=np.float32)
    gyro = df[GYRO_COLS].to_numpy(dtype=np.float32)

    acc = pad_or_cut(acc, TARGET_LEN)
    gyro = pad_or_cut(gyro, TARGET_LEN)

    label = 1 if is_fall == "낙상" else 0
    subject_id = scene_id.split("_")[0]

    acc_list.append(acc)
    gyro_list.append(gyro)
    labels.append(label)
    subject_ids.append(subject_id)
    scene_ids.append(scene_id)

acc_arr = np.stack(acc_list)
gyro_arr = np.stack(gyro_list)
labels_arr = np.array(labels, dtype=np.int64)
subject_ids_arr = np.array(subject_ids)
scene_ids_arr = np.array(scene_ids)

np.savez(
    SAVE_PATH,
    acc=acc_arr,
    gyro=gyro_arr,
    labels=labels_arr,
    subject_ids=subject_ids_arr,
    scene_ids=scene_ids_arr,
)

print("Saved:", SAVE_PATH)
print("acc:", acc_arr.shape)
print("gyro:", gyro_arr.shape)
print("labels:", labels_arr.shape)
print("Fall:", int((labels_arr == 1).sum()))
print("Non-fall:", int((labels_arr == 0).sum()))
print("Subjects:", len(np.unique(subject_ids_arr)))