import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NPZ_PATH = "./aihub_cane_adult2_window_val.npz"

MODEL_PATH = (
    "./aihub_cane_adult2_window_outputs2/"
    "aihub-holdout-80-20_best_model.pth"
)

SAVE_DIR = "./aihub_validation_test_outputs"

BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)


class FallDataset(Dataset):
    def __init__(self, acc_data, gyro_data, labels):
        self.acc_data = torch.tensor(acc_data, dtype=torch.float32)
        self.gyro_data = torch.tensor(gyro_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.acc_data[idx], self.gyro_data[idx], self.labels[idx]


class FeatureSelfAttention(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) / math.sqrt(self.dim)
        attn = F.softmax(scores, dim=-1)
        weighted = torch.bmm(attn, v.unsqueeze(-1)).squeeze(-1)
        return weighted


class SensorStreamCNN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.global_pool(x).squeeze(-1)
        return x


class DSCS(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, base_channels=64, dropout=0.5):
        super().__init__()
        self.acc_stream = SensorStreamCNN(in_channels, base_channels)
        self.gyro_stream = SensorStreamCNN(in_channels, base_channels)

        self.acc_attn = FeatureSelfAttention(dim=base_channels)
        self.gyro_attn = FeatureSelfAttention(dim=base_channels)

        self.bn = nn.BatchNorm1d(base_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(base_channels * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, acc, gyro):
        acc_feat = self.acc_stream(acc)
        gyro_feat = self.gyro_stream(gyro)

        acc_feat = self.acc_attn(acc_feat)
        gyro_feat = self.gyro_attn(gyro_feat)

        x = torch.cat([acc_feat, gyro_feat], dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def compute_binary_metrics(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    preds_all = []
    labels_all = []

    for acc, gyro, labels in loader:
        acc = acc.to(device)
        gyro = gyro.to(device)
        labels = labels.to(device)

        logits = model(acc, gyro)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)

        preds_all.append(preds.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    metrics = compute_binary_metrics(preds_all, labels_all)
    metrics["loss"] = total_loss / len(labels_all)

    return metrics, preds_all, labels_all


if __name__ == "__main__":
    print("=" * 80)
    print("AIHUB VALIDATION TEST")
    print("=" * 80)

    print("NPZ path:", NPZ_PATH)
    print("Model path:", MODEL_PATH)

    data = np.load(NPZ_PATH, allow_pickle=True)

    acc = data["acc"]
    gyro = data["gyro"]
    labels = data["labels"]

    print("acc shape:", acc.shape)
    print("gyro shape:", gyro.shape)
    print("labels shape:", labels.shape)
    print("Fall:", int((labels == 1).sum()))
    print("Non-fall:", int((labels == 0).sum()))

    dataset = FallDataset(acc, gyro, labels)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = DSCS(
        num_classes=2,
        in_channels=3,
        base_channels=64,
        dropout=0.5,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    metrics, preds, gt = evaluate(model, loader, criterion, DEVICE)

    print("=" * 80)
    print("VALIDATION RESULT")
    print("=" * 80)

    for k, v in metrics.items():
        print(f"{k:12s}: {v}")

    metrics_csv = os.path.join(SAVE_DIR, "validation_metrics.csv")

    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for k, v in metrics.items():
            writer.writerow([k, v])

    pred_csv = os.path.join(SAVE_DIR, "validation_predictions.csv")

    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt", "pred"])

        for g, p in zip(gt, preds):
            writer.writerow([int(g), int(p)])

    print("=" * 80)
    print("Saved metrics:", metrics_csv)
    print("Saved predictions:", pred_csv)
    print("=" * 80)