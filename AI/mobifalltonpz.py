import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# MobiFall folder -> data.npz converter
# ------------------------------------------------------------
# Expected MobiFall filename format:
#   <ACTIVITY>_<SENSOR>_<SUBJECT_ID>_<TRIAL>.txt
#
# Examples:
#   WAL_acc_5_1.txt
#   FKL_gyro_3_2.txt
#   BSC_ori_19_3.txt
#
# Folder structure example:
#   root/
#     sub1/
#       ADL/
#         WAL/
#           WAL_acc_1_1.txt
#           WAL_gyro_1_1.txt
#           WAL_ori_1_1.txt
#       FALLS/
#         FOL/
#           FOL_acc_1_1.txt
#           FOL_gyro_1_1.txt
#
# Output NPZ keys:
#   acc          (N, T, 3)
#   gyro         (N, T, 3)
#   labels       (N,)
#   subject_ids  (N,)
#   activity_ids (N,)
#   file_names   (N,)
#   trial_ids    (N,)
#
# Default behavior:
#   - pair acc + gyro per trial
#   - align using timestamp overlap
#   - resample to 50 Hz
#   - optional low-pass filtering
#   - fixed 8-second windows (default)
# ============================================================


FALL_CODES = {"FOL", "FKL", "BSC", "SDL"}
ADL_CODES = {"STD", "WAL", "JOG", "JUM", "STU", "STN", "SCH", "CSI", "CSO"}
VALID_SENSOR_CODES = {"acc", "gyro", "ori"}


def get_binary_label(activity_code: str) -> int:
    if activity_code in FALL_CODES:
        return 1
    if activity_code in ADL_CODES:
        return 0
    raise ValueError(f"Unknown activity code: {activity_code}")


def parse_mobifall_filename(filename: str) -> Tuple[str, str, int, int]:
    """
    Example:
        WAL_acc_5_1.txt -> ("WAL", "acc", 5, 1)
    """
    m = re.match(r"^([A-Z]{3})_(acc|gyro|ori)_(\d+)_(\d+)\.txt$", filename)
    if not m:
        raise ValueError(f"Unexpected MobiFall filename format: {filename}")
    activity_code, sensor_code, subject_id, trial_id = m.groups()
    return activity_code, sensor_code, int(subject_id), int(trial_id)


@dataclass
class PrepConfig:
    input_dir: str
    output_path: str

    target_fs: float = 50.0

    # filtering after alignment/resampling
    apply_filter: bool = False
    cutoff_hz: float = 20.0
    filter_order: int = 4

    # windowing
    window_seconds: float = 8.0
    step_seconds: Optional[float] = None
    drop_last_incomplete: bool = True

    # timestamp handling
    timestamp_unit: str = "ns"   # ns | ms | s


def butter_lowpass_filter(data: np.ndarray, fs: float, cutoff: float = 20.0, order: int = 4) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for filtering but is not installed.")
    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(f"cutoff ({cutoff}) must be smaller than Nyquist ({nyq}).")
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


def convert_timestamps_to_seconds(ts: np.ndarray, unit: str) -> np.ndarray:
    ts = ts.astype(np.float64)
    if unit == "ns":
        return ts / 1e9
    if unit == "ms":
        return ts / 1e3
    if unit == "s":
        return ts
    raise ValueError(f"Unsupported timestamp unit: {unit}")


def clean_and_sort_time_series(ts_xyz: np.ndarray, timestamp_unit: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        ts_xyz: (N, 4) -> [timestamp, x, y, z]
    Returns:
        t_sec:  (N,)
        xyz:    (N, 3)
    """
    if ts_xyz.ndim != 2 or ts_xyz.shape[1] < 4:
        raise ValueError(f"Expected shape (N,4+) but got {ts_xyz.shape}")

    t = convert_timestamps_to_seconds(ts_xyz[:, 0], timestamp_unit)
    xyz = ts_xyz[:, 1:4].astype(np.float32)

    # remove non-finite
    mask = np.isfinite(t) & np.all(np.isfinite(xyz), axis=1)
    t = t[mask]
    xyz = xyz[mask]

    if len(t) < 2:
        raise ValueError("Too few valid rows after cleaning.")

    # sort by time
    order = np.argsort(t)
    t = t[order]
    xyz = xyz[order]

    # remove duplicate timestamps (keep first)
    unique_mask = np.ones(len(t), dtype=bool)
    unique_mask[1:] = np.diff(t) > 0
    t = t[unique_mask]
    xyz = xyz[unique_mask]

    if len(t) < 2:
        raise ValueError("Too few unique timestamps after deduplication.")

    return t, xyz


def interpolate_to_common_timeline(
    acc_tsxyz: np.ndarray,
    gyro_tsxyz: np.ndarray,
    target_fs: float,
    timestamp_unit: str = "ns",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses the overlapping timestamp range between acc and gyro,
    then interpolates both streams onto a common uniform time grid.
    """
    t_acc, acc = clean_and_sort_time_series(acc_tsxyz, timestamp_unit)
    t_gyro, gyro = clean_and_sort_time_series(gyro_tsxyz, timestamp_unit)

    start_t = max(t_acc[0], t_gyro[0])
    end_t = min(t_acc[-1], t_gyro[-1])

    if end_t <= start_t:
        raise ValueError("No overlapping timestamp region between acc and gyro.")

    duration = end_t - start_t
    if duration <= 0:
        raise ValueError("Non-positive overlap duration.")

    n_samples = int(np.floor(duration * target_fs)) + 1
    if n_samples < 2:
        raise ValueError("Too few samples after alignment.")

    common_t = start_t + np.arange(n_samples, dtype=np.float64) / target_fs
    common_t = common_t[common_t <= end_t]

    if len(common_t) < 2:
        raise ValueError("Too few common timeline points.")

    acc_interp = np.stack(
        [np.interp(common_t, t_acc, acc[:, i]) for i in range(3)],
        axis=1
    ).astype(np.float32)

    gyro_interp = np.stack(
        [np.interp(common_t, t_gyro, gyro[:, i]) for i in range(3)],
        axis=1
    ).astype(np.float32)

    return acc_interp, gyro_interp


def preprocess_stream(
    stream: np.ndarray,
    fs: float,
    apply_filter: bool,
    cutoff_hz: float,
    filter_order: int,
) -> np.ndarray:
    x = stream.astype(np.float32)
    if apply_filter:
        x = butter_lowpass_filter(x, fs=fs, cutoff=cutoff_hz, order=filter_order)
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


def read_sensor_txt(file_path: str) -> np.ndarray:
    """
    Reads a MobiFall sensor txt file with rows like:
        timestamp,x,y,z
    Robust to whitespace / tabs / commas / semicolons.
    Skips malformed rows.
    """
    rows: List[List[float]] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # normalize separators
            line = line.replace(";", " ").replace(",", " ").replace("\t", " ")
            parts = [p for p in line.split() if p]

            if len(parts) < 4:
                continue

            try:
                row = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
                rows.append(row)
            except ValueError:
                # skip headers or malformed rows
                continue

    if not rows:
        raise ValueError(f"No valid rows found in file: {file_path}")

    return np.asarray(rows, dtype=np.float64)


def discover_trial_pairs(root_dir: str) -> Dict[Tuple[int, str, int], Dict[str, str]]:
    """
    Returns:
        {
            (subject_id, activity_code, trial_id): {
                "acc": path_to_acc,
                "gyro": path_to_gyro,
                "ori": path_to_ori (optional)
            },
            ...
        }
    """
    pairs: Dict[Tuple[int, str, int], Dict[str, str]] = {}

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(".txt"):
                continue

            try:
                activity_code, sensor_code, subject_id, trial_id = parse_mobifall_filename(file)
            except Exception:
                continue

            key = (subject_id, activity_code, trial_id)
            if key not in pairs:
                pairs[key] = {}

            full_path = os.path.join(root, file)
            pairs[key][sensor_code] = full_path

    return pairs


def process_single_trial(
    subject_id: int,
    activity_code: str,
    trial_id: int,
    sensor_files: Dict[str, str],
    cfg: PrepConfig,
):
    if "acc" not in sensor_files or "gyro" not in sensor_files:
        raise ValueError("Both acc and gyro files are required.")

    label = get_binary_label(activity_code)

    acc_raw = read_sensor_txt(sensor_files["acc"])    # (T1, 4)
    gyro_raw = read_sensor_txt(sensor_files["gyro"])  # (T2, 4)

    acc, gyro = interpolate_to_common_timeline(
        acc_tsxyz=acc_raw,
        gyro_tsxyz=gyro_raw,
        target_fs=cfg.target_fs,
        timestamp_unit=cfg.timestamp_unit,
    )

    acc = preprocess_stream(
        acc,
        fs=cfg.target_fs,
        apply_filter=cfg.apply_filter,
        cutoff_hz=cfg.cutoff_hz,
        filter_order=cfg.filter_order,
    )
    gyro = preprocess_stream(
        gyro,
        fs=cfg.target_fs,
        apply_filter=cfg.apply_filter,
        cutoff_hz=cfg.cutoff_hz,
        filter_order=cfg.filter_order,
    )

    window_len = int(round(cfg.window_seconds * cfg.target_fs))
    step_seconds = cfg.step_seconds if cfg.step_seconds is not None else cfg.window_seconds
    step_len = int(round(step_seconds * cfg.target_fs))

    acc_windows, gyro_windows = make_windows(
        acc,
        gyro,
        window_len=window_len,
        step_len=step_len,
        drop_last_incomplete=cfg.drop_last_incomplete,
    )

    base_name = f"{activity_code}_{subject_id}_{trial_id}"
    file_name_value = f"{base_name}[acc+gyro]"

    labels = [label] * len(acc_windows)
    subject_ids = [subject_id] * len(acc_windows)
    activity_ids = [activity_code] * len(acc_windows)
    file_names = [file_name_value] * len(acc_windows)
    trial_ids = [trial_id] * len(acc_windows)

    return acc_windows, gyro_windows, labels, subject_ids, activity_ids, file_names, trial_ids


def convert_mobifall_folder(cfg: PrepConfig) -> None:
    if not os.path.isdir(cfg.input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {cfg.input_dir}")

    trial_pairs = discover_trial_pairs(cfg.input_dir)
    if not trial_pairs:
        raise FileNotFoundError(f"No valid MobiFall sensor files found in: {cfg.input_dir}")

    all_acc: List[np.ndarray] = []
    all_gyro: List[np.ndarray] = []
    all_labels: List[int] = []
    all_subject_ids: List[int] = []
    all_activity_ids: List[str] = []
    all_file_names: List[str] = []
    all_trial_ids: List[int] = []

    skipped_trials: List[Tuple[str, str]] = []

    for (subject_id, activity_code, trial_id), sensor_files in sorted(trial_pairs.items()):
        trial_name = f"{activity_code}_{subject_id}_{trial_id}"
        try:
            acc_w, gyro_w, labels, subject_ids, activity_ids, file_names, trial_ids = process_single_trial(
                subject_id=subject_id,
                activity_code=activity_code,
                trial_id=trial_id,
                sensor_files=sensor_files,
                cfg=cfg,
            )

            all_acc.extend(acc_w)
            all_gyro.extend(gyro_w)
            all_labels.extend(labels)
            all_subject_ids.extend(subject_ids)
            all_activity_ids.extend(activity_ids)
            all_file_names.extend(file_names)
            all_trial_ids.extend(trial_ids)

        except Exception as e:
            skipped_trials.append((trial_name, str(e)))

    if not all_acc:
        raise RuntimeError("No windows were created. Check your folder path and preprocessing settings.")

    acc_arr = np.stack(all_acc, axis=0).astype(np.float32)
    gyro_arr = np.stack(all_gyro, axis=0).astype(np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    subject_ids_arr = np.asarray(all_subject_ids, dtype=np.int64)
    activity_ids_arr = np.asarray(all_activity_ids)
    file_names_arr = np.asarray(all_file_names)
    trial_ids_arr = np.asarray(all_trial_ids, dtype=np.int64)

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    np.savez(
        cfg.output_path,
        acc=acc_arr,
        gyro=gyro_arr,
        labels=labels_arr,
        subject_ids=subject_ids_arr,
        activity_ids=activity_ids_arr,
        file_names=file_names_arr,
        trial_ids=trial_ids_arr,   # remove this line if you want exact same keys as SisFall NPZ
    )

    n_fall = int((labels_arr == 1).sum())
    n_adl = int((labels_arr == 0).sum())

    print("=" * 80)
    print("MobiFall conversion complete")
    print(f"Output file:   {cfg.output_path}")
    print(f"acc shape:     {acc_arr.shape}")
    print(f"gyro shape:    {gyro_arr.shape}")
    print(f"labels shape:  {labels_arr.shape}")
    print(f"Fall windows:  {n_fall}")
    print(f"ADL windows:   {n_adl}")
    print(f"Subjects:      {len(np.unique(subject_ids_arr))}")
    print(f"Activities:    {sorted(set(activity_ids_arr.tolist()))}")
    if skipped_trials:
        print(f"Skipped trials: {len(skipped_trials)}")
        for name, msg in skipped_trials[:10]:
            print(f"  - {name}: {msg}")
    print("=" * 80)


def build_argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert MobiFall folder to data.npz")
    parser.add_argument("--input-dir", type=str, required=True, help="Root folder containing MobiFall subject folders")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")

    parser.add_argument("--target-fs", type=float, default=50.0)
    parser.add_argument("--timestamp-unit", type=str, default="ns", choices=["ns", "ms", "s"])

    parser.add_argument("--apply-filter", action="store_true")
    parser.add_argument("--cutoff-hz", type=float, default=20.0)
    parser.add_argument("--filter-order", type=int, default=4)

    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--step-seconds", type=float, default=None,
                        help="Default: same as window_seconds (non-overlap)")
    parser.add_argument("--keep-last-incomplete", action="store_true",
                        help="Pad and keep the last incomplete window")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    cfg = PrepConfig(
        input_dir=args.input_dir,
        output_path=args.output,
        target_fs=args.target_fs,
        apply_filter=args.apply_filter,
        cutoff_hz=args.cutoff_hz,
        filter_order=args.filter_order,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        drop_last_incomplete=not args.keep_last_incomplete,
        timestamp_unit=args.timestamp_unit,
    )

    convert_mobifall_folder(cfg)


if __name__ == "__main__":
    main()


"""
Examples:

1) Paper-like setting:
python mobifall_to_npz.py ^
  --input-dir C:\\MobiFall_Dataset ^
  --output .\\mobifall_data.npz ^
  --target-fs 50 ^
  --window-seconds 8 ^
  --apply-filter ^
  --cutoff-hz 20

2) Keep original duration chunks only if enough for 8 sec:
python mobifall_to_npz.py ^
  --input-dir C:\\MobiFall_Dataset ^
  --output .\\mobifall_data.npz

3) Keep last incomplete window with padding:
python mobifall_to_npz.py ^
  --input-dir C:\\MobiFall_Dataset ^
  --output .\\mobifall_data_pad.npz ^
  --keep-last-incomplete
"""