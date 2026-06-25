
from pathlib import Path
import json
from collections import Counter

ROOTS = [
    Path(r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL\센서")
]

total = 0
fall_count = 0
nonfall_count = 0

fall_type_counter = Counter()

matched_files = []

for root in ROOTS:
    for json_path in root.rglob("*.json"):

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        age = data.get("actor_info", {}).get("actor_age", "")
        method = data.get("scene_info", {}).get("scene_method", "")
        is_fall = data.get("scene_info", {}).get("scene_IsFall", "")
        fall_type = data.get("scene_info", {}).get("scene_cat_name", "")



        # 40대 이상 + 지팡이
        if "adult2" in age and method == "cane":

            total += 1
            matched_files.append(json_path)

            if is_fall == "낙상":
                fall_count += 1
                fall_type_counter[fall_type] += 1
            else:
                nonfall_count += 1

print("=" * 50)
print("40대 이상 + 지팡이(cane)")
print("=" * 50)

print(f"전체: {total}")
print(f"낙상: {fall_count}")
print(f"비낙상: {nonfall_count}")

print("\n낙상 세부 유형")
for k, v in fall_type_counter.items():
    print(f"{k}: {v}")

print("\n예시 파일")
for p in matched_files[-10:]:
    print(p)