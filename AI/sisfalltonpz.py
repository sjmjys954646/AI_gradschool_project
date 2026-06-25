import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, filtfilt, resample
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# SisFall folder -> data.npz converter
# ------------------------------------------------------------
# Expected SisFall filename format:
#   D01_SA01_R01.txt
# where:
#   D01 = activity id
#   SA01 = subject id
#   R01 = repetition id
#
# Each row in a SisFall text file usually contains 9 values:
#   acc1_x, acc1_y, acc1_z,
#   acc2_x, acc2_y, acc2_z,
#   gyro_x, gyro_y, gyro_z
#
# This script can:
#   1) read all SisFall txt files in a folder
#   2) parse filename -> activity / subject / repetition
#   3) extract channels
#   4) optionally filter + resample
#   5) create fixed-length windows
#   6) assign binary labels (fall=1, ADL=0)
#   7) save everything into a .npz file
#
# Output NPZ keys:
#   acc         (N, T, 3)
#   gyro        (N, T, 3)
#   labels      (N,)
#   subject_ids (N,)
#   activity_ids (N,)
#   file_names   (N,)
#
# Example:
#   python sisfall_to_npz.py \
#       --input-dir /path/to/SisFall \
#       --output ./sisfall_windows.npz \
#       --window-seconds 12 \
#       --original-fs 200 \
#       --target-fs 50 \
#       --apply-filter
# ============================================================


# SisFall activity IDs based on the public dataset convention
FALL_IDS = {f"F{i:02d}" for i in range(1, 16)}
ADL_IDS = {f"D{i:02d}" for i in range(1, 20)}

def parse_sisfall_filename(filename: str):
    m = re.match(r"^([DF]\d{2})_(S[AE]\d{2})_(R\d{2})\.txt$", filename)
    if not m:
        raise ValueError(f"Unexpected SisFall filename format: {filename}")
    activity_id, subject_id, repetition_id = m.groups()
    return activity_id, subject_id, repetition_id

def get_binary_label(activity_id: str) -> int:
    if activity_id in FALL_IDS:
        return 1
    if activity_id in ADL_IDS:
        return 0
    raise ValueError(f"Unknown activity_id: {activity_id}")


@dataclass
class PrepConfig:
    input_dir: str
    output_path: str
    original_fs: float = 200.0
    target_fs: Optional[float] = 50.0
    apply_filter: bool = False
    cutoff_hz: float = 20.0
    filter_order: int = 4
    window_seconds: float = 12.0
    step_seconds: Optional[float] = None
    use_acc_sensor: str = "acc1"   # acc1 | acc2 | mean
    drop_last_incomplete: bool = True


def butter_lowpass_filter(data: np.ndarray, fs: float, cutoff: float = 20.0, order: int = 4) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for filtering but is not installed.")
    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(f"cutoff ({cutoff}) must be smaller than Nyquist ({nyq}).")
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


def resample_signal(data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for resampling but is not installed.")
    if original_fs == target_fs:
        return data
    target_len = int(round(len(data) * target_fs / original_fs))
    return resample(data, target_len, axis=0)


def subject_str_to_int(subject_id: str) -> int:
    prefix = subject_id[:2]  # SA or SE
    num = int(subject_id[2:])

    if prefix == "SA":
        return num  # 1 ~ 23
    elif prefix == "SE":
        return 100 + num  # 101 ~ 115
    else:
        raise ValueError(f"Unknown subject prefix: {subject_id}")


def read_sisfall_txt(file_path: str) -> np.ndarray:
    rows: List[List[float]] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.rstrip(";")
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) != 9:
                # skip malformed rows quietly
                continue
            rows.append([float(x) for x in parts])

    if not rows:
        raise ValueError(f"No valid rows found in file: {file_path}")

    return np.asarray(rows, dtype=np.float32)  # (T, 9)


def split_channels(data_9ch: np.ndarray, use_acc_sensor: str = "acc1") -> Tuple[np.ndarray, np.ndarray]:
    acc1 = data_9ch[:, 0:3]
    acc2 = data_9ch[:, 3:6]
    gyro = data_9ch[:, 6:9]

    if use_acc_sensor == "acc1":
        acc = acc1
    elif use_acc_sensor == "acc2":
        acc = acc2
    elif use_acc_sensor == "mean":
        acc = (acc1 + acc2) / 2.0
    else:
        raise ValueError("use_acc_sensor must be one of: acc1, acc2, mean")

    return acc.astype(np.float32), gyro.astype(np.float32)


def preprocess_stream(
    stream: np.ndarray,
    original_fs: float,
    target_fs: Optional[float],
    apply_filter: bool,
    cutoff_hz: float,
    filter_order: int,
) -> np.ndarray:
    x = stream.astype(np.float32)
    if apply_filter:
        x = butter_lowpass_filter(x, fs=original_fs, cutoff=cutoff_hz, order=filter_order)
    if target_fs is not None and target_fs != original_fs:
        x = resample_signal(x, original_fs=original_fs, target_fs=target_fs)
    return x.astype(np.float32)


def make_windows(
    acc: np.ndarray,
    gyro: np.ndarray,
    window_len: int,
    step_len: int,
    drop_last_incomplete: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if len(acc) != len(gyro):
        raise ValueError("acc and gyro must have the same length")

    acc_windows: List[np.ndarray] = []
    gyro_windows: List[np.ndarray] = []
    total_len = len(acc)

    if total_len < window_len:
        if drop_last_incomplete:
            return acc_windows, gyro_windows
        pad_len = window_len - total_len
        acc_pad = np.pad(acc, ((0, pad_len), (0, 0)), mode="edge")
        gyro_pad = np.pad(gyro, ((0, pad_len), (0, 0)), mode="edge")
        return [acc_pad], [gyro_pad]

    start = 0
    while start + window_len <= total_len:
        end = start + window_len
        acc_windows.append(acc[start:end])
        gyro_windows.append(gyro[start:end])
        start += step_len

    if not drop_last_incomplete and start < total_len:
        acc_tail = acc[start:]
        gyro_tail = gyro[start:]
        pad_len = window_len - len(acc_tail)
        if pad_len > 0:
            acc_tail = np.pad(acc_tail, ((0, pad_len), (0, 0)), mode="edge")
            gyro_tail = np.pad(gyro_tail, ((0, pad_len), (0, 0)), mode="edge")
        acc_windows.append(acc_tail)
        gyro_windows.append(gyro_tail)

    return acc_windows, gyro_windows


def process_single_file(file_path: str, cfg: PrepConfig):
    filename = os.path.basename(file_path)
    activity_id, subject_id, repetition_id = parse_sisfall_filename(filename)
    label = get_binary_label(activity_id)
    subject_int = subject_str_to_int(subject_id)

    raw = read_sisfall_txt(file_path)  # (T, 9)
    acc, gyro = split_channels(raw, use_acc_sensor=cfg.use_acc_sensor)

    acc = preprocess_stream(
        acc,
        original_fs=cfg.original_fs,
        target_fs=cfg.target_fs,
        apply_filter=cfg.apply_filter,
        cutoff_hz=cfg.cutoff_hz,
        filter_order=cfg.filter_order,
    )
    gyro = preprocess_stream(
        gyro,
        original_fs=cfg.original_fs,
        target_fs=cfg.target_fs,
        apply_filter=cfg.apply_filter,
        cutoff_hz=cfg.cutoff_hz,
        filter_order=cfg.filter_order,
    )

    effective_fs = cfg.target_fs if cfg.target_fs is not None else cfg.original_fs
    window_len = int(round(cfg.window_seconds * effective_fs))
    step_seconds = cfg.step_seconds if cfg.step_seconds is not None else cfg.window_seconds
    step_len = int(round(step_seconds * effective_fs))

    acc_windows, gyro_windows = make_windows(
        acc,
        gyro,
        window_len=window_len,
        step_len=step_len,
        drop_last_incomplete=cfg.drop_last_incomplete,
    )

    labels = [label] * len(acc_windows)
    subject_ids = [subject_int] * len(acc_windows)
    activity_ids = [activity_id] * len(acc_windows)
    file_names = [filename] * len(acc_windows)

    return acc_windows, gyro_windows, labels, subject_ids, activity_ids, file_names


def convert_sisfall_folder(cfg: PrepConfig) -> None:
    if not os.path.isdir(cfg.input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {cfg.input_dir}")

    txt_files = []

    for root, dirs, files in os.walk(cfg.input_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {cfg.input_dir}")

    all_acc: List[np.ndarray] = []
    all_gyro: List[np.ndarray] = []
    all_labels: List[int] = []
    all_subject_ids: List[int] = []
    all_activity_ids: List[str] = []
    all_file_names: List[str] = []

    skipped_files = []

    for filename in txt_files:
        file_path = filename  # 이미 full path임
        filename = os.path.basename(file_path)
        try:
            acc_w, gyro_w, labels, subject_ids, activity_ids, file_names = process_single_file(file_path, cfg)
            all_acc.extend(acc_w)
            all_gyro.extend(gyro_w)
            all_labels.extend(labels)
            all_subject_ids.extend(subject_ids)
            all_activity_ids.extend(activity_ids)
            all_file_names.extend(file_names)
        except Exception as e:
            skipped_files.append((filename, str(e)))

    if not all_acc:
        raise RuntimeError("No windows were created. Check your folder path and preprocessing settings.")

    acc_arr = np.stack(all_acc, axis=0).astype(np.float32)
    gyro_arr = np.stack(all_gyro, axis=0).astype(np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    subject_ids_arr = np.asarray(all_subject_ids, dtype=np.int64)
    activity_ids_arr = np.asarray(all_activity_ids)
    file_names_arr = np.asarray(all_file_names)

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    np.savez(
        cfg.output_path,
        acc=acc_arr,
        gyro=gyro_arr,
        labels=labels_arr,
        subject_ids=subject_ids_arr,
        activity_ids=activity_ids_arr,
        file_names=file_names_arr,
    )

    n_fall = int((labels_arr == 1).sum())
    n_adl = int((labels_arr == 0).sum())

    print("=" * 80)
    print("SisFall conversion complete")
    print(f"Output file: {cfg.output_path}")
    print(f"acc shape:   {acc_arr.shape}")
    print(f"gyro shape:  {gyro_arr.shape}")
    print(f"labels:      {labels_arr.shape}")
    print(f"Fall windows: {n_fall}")
    print(f"ADL windows:  {n_adl}")
    print(f"Subjects:     {len(np.unique(subject_ids_arr))}")
    if skipped_files:
        print(f"Skipped files: {len(skipped_files)}")
        for fname, msg in skipped_files[:10]:
            print(f"  - {fname}: {msg}")
    print("=" * 80)


def build_argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert SisFall folder to data.npz")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing SisFall .txt files")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    parser.add_argument("--original-fs", type=float, default=200.0)
    parser.add_argument("--target-fs", type=float, default=50.0)
    parser.add_argument("--apply-filter", action="store_true")
    parser.add_argument("--cutoff-hz", type=float, default=20.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--step-seconds", type=float, default=None, help="Default: same as window_seconds (non-overlap)")
    parser.add_argument("--use-acc-sensor", type=str, default="acc1", choices=["acc1", "acc2", "mean"])
    parser.add_argument("--keep-last-incomplete", action="store_true", help="Pad and keep the last incomplete window")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    cfg = PrepConfig(
        input_dir=args.input_dir,
        output_path=args.output,
        original_fs=args.original_fs,
        target_fs=args.target_fs,
        apply_filter=args.apply_filter,
        cutoff_hz=args.cutoff_hz,
        filter_order=args.filter_order,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        use_acc_sensor=args.use_acc_sensor,
        drop_last_incomplete=not args.keep_last_incomplete,
    )

    convert_sisfall_folder(cfg)


if __name__ == "__main__":
    main()


"""
python sisfalltonpz.py --input-dir ./Sisfall_dataset --output ./sisfall_data.npz

python sisfalltonpz.py --input-dir ./Sisfall_dataset --output ./sisfall_data2.npz --original-fs 200 --target-fs 200 --window-seconds 12
"""