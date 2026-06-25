import os
import torch
import pandas as pd
import numpy as np

folder = "loso_outputs20"  # 네 폴더명

rows = []

for file in os.listdir(folder):
    if file.endswith(".pt"):
        path = os.path.join(folder, file)
        data = torch.load(path, map_location="cpu")

        metrics = data["metrics"]

        row = {
            "subject": data["test_subject"],
            "accuracy": metrics["accuracy"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "loss": metrics["loss"],
        }

        rows.append(row)

# DataFrame 생성
df = pd.DataFrame(rows)

# subject 기준 정렬 (중요 ⭐)
df = df.sort_values(by="subject")

# 저장
df.to_csv("loso_results.csv", index=False)

print("✅ CSV 저장 완료: loso_results.csv")
print(df.head())

all_metrics = []

for file in os.listdir(folder):
    if file.endswith(".pt"):
        path = os.path.join(folder, file)
        data = torch.load(path, map_location="cpu")

        metrics = data["metrics"]
        all_metrics.append(metrics)

print("총 fold 수:", len(all_metrics))

# 평균 계산
metric_names = ["accuracy", "recall", "precision", "specificity", "f1"]

print("=" * 50)
print("LOSO RE-CALCULATED SUMMARY")

for name in metric_names:
    values = np.array([m[name] for m in all_metrics])
    print(f"{name:12s}: {values.mean():.4f} ± {values.std():.4f}")