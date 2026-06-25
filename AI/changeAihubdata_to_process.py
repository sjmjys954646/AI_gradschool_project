from pathlib import Path
import pandas as pd

SRC_ROOT = Path(r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\01.원천데이터\VS\센서")
DST_ROOT = Path(r"D:\fall_sensor_right_forearm\Validation\01.원천데이터\VS\센서")

TARGET_COLS = [
    "Frame",
    "Segment Angular Velocity_Right Forearm x",
    "Segment Angular Velocity_Right Forearm y",
    "Segment Angular Velocity_Right Forearm z",
    "Segment Acceleration_Right Forearm x",
    "Segment Acceleration_Right Forearm y",
    "Segment Acceleration_Right Forearm z",
]

csv_files = list(SRC_ROOT.rglob("*.csv"))

print(f"총 CSV 수: {len(csv_files)}")

for src_csv in csv_files:
    rel_path = src_csv.relative_to(SRC_ROOT)
    dst_csv = DST_ROOT / rel_path

    try:
        df = pd.read_csv(src_csv)

        missing = [c for c in TARGET_COLS if c not in df.columns]
        if missing:
            print(f"[스킵] 컬럼 없음: {src_csv}")
            print("  missing:", missing)
            continue

        out = df[TARGET_COLS]

        dst_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst_csv, index=False, encoding="utf-8-sig")

        print(f"[완료] {dst_csv}")

    except Exception as e:
        print(f"[에러] {src_csv}")
        print(e)

print("끝")