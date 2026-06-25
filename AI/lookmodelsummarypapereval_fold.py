import re
import pandas as pd

log_path = ".\paper_eval_outputs\메모장.txt"

rows = []
current_mode = None  # "5fold" or "10fold"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:

        # 모드 감지
        if "5-FOLD CROSS-VALIDATION" in line:
            current_mode = "5fold"
        elif "10-FOLD CROSS-VALIDATION" in line:
            current_mode = "10fold"

        if "Best test metrics for fold" in line:
            fold_match = re.search(r"fold (\d+)", line)
            fold = int(fold_match.group(1))

            dict_str = line.split(":", 1)[1].strip()
            metrics = eval(dict_str)

            row = {
                "mode": current_mode,   # 🔥 핵심
                "fold": fold,
                **metrics
            }

            rows.append(row)

df = pd.DataFrame(rows)

# 정렬
df = df.sort_values(by=["mode", "fold"])

df.to_csv("fixed_results.csv", index=False)

print(df)