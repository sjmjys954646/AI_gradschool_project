import os
import math
import csv
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utils
# ============================================================
def compute_binary_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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


# ============================================================
# Dataset
# ============================================================
class FallDataset(Dataset):
    def __init__(self, acc_data: np.ndarray, gyro_data: np.ndarray, labels: np.ndarray):
        self.acc_data = torch.tensor(acc_data, dtype=torch.float32)
        self.gyro_data = torch.tensor(gyro_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.acc_data[idx], self.gyro_data[idx], self.labels[idx]


# ============================================================
# DSCS Model
# ============================================================
class FeatureSelfAttention(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) / math.sqrt(self.dim)
        attn = F.softmax(scores, dim=-1)
        weighted = torch.bmm(attn, v.unsqueeze(-1)).squeeze(-1)
        return weighted


class SensorStreamCNN(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3) -> (B, 3, T)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.global_pool(x).squeeze(-1)
        return x


class DSCS(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.acc_stream = SensorStreamCNN(in_channels, base_channels)
        self.gyro_stream = SensorStreamCNN(in_channels, base_channels)

        self.acc_attn = FeatureSelfAttention(dim=base_channels)
        self.gyro_attn = FeatureSelfAttention(dim=base_channels)

        self.bn = nn.BatchNorm1d(base_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(base_channels * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
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


# ============================================================
# Config
# ============================================================
@dataclass
class ExternalTestConfig:
    npz_path: str = "./aihub_cane_adult2_window.npz"
    model_path: str = "./combined_eval_outputs/combined-holdout-80-20_best_model.pth"
    save_dir: str = "./aihub_external_test_outputs"

    batch_size: int = 128
    num_workers: int = 0
    dropout: float = 0.5
    base_channels: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load / Eval
# ============================================================
def make_loader(acc, gyro, labels, batch_size, num_workers):
    dataset = FallDataset(acc, gyro, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_state_dict_safely(model: nn.Module, model_path: str, device: torch.device):
    state = torch.load(model_path, map_location=device)

    # pth가 state_dict 자체인 경우와 dict 안에 state_dict가 들어있는 경우 모두 처리
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # DataParallel로 저장된 경우 module. prefix 제거
    cleaned_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned_state[new_k] = v

    model.load_state_dict(cleaned_state)
    return model


@torch.no_grad()
def evaluate_external(model, loader, device):
    model.eval()

    preds_all = []
    labels_all = []
    probs_all = []

    for acc, gyro, labels in loader:
        acc = acc.to(device)
        gyro = gyro.to(device)

        logits = model(acc, gyro)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        preds_all.append(preds.cpu().numpy())
        labels_all.append(labels.numpy())
        probs_all.append(probs[:, 1].cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    probs_all = np.concatenate(probs_all)

    metrics = compute_binary_metrics(preds_all, labels_all)
    return metrics, preds_all, labels_all, probs_all


def save_metrics_csv(metrics: Dict[str, float], save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_predictions_csv(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    npz_data,
    save_path: str,
):
    scene_ids: Optional[np.ndarray] = npz_data["scene_ids"] if "scene_ids" in npz_data.files else None
    subject_ids: Optional[np.ndarray] = npz_data["subject_ids"] if "subject_ids" in npz_data.files else None
    window_starts: Optional[np.ndarray] = npz_data["window_starts"] if "window_starts" in npz_data.files else None
    window_ends: Optional[np.ndarray] = npz_data["window_ends"] if "window_ends" in npz_data.files else None

    fieldnames = ["index", "label", "pred", "prob_fall"]
    if subject_ids is not None:
        fieldnames.append("subject_id")
    if scene_ids is not None:
        fieldnames.append("scene_id")
    if window_starts is not None:
        fieldnames.append("window_start")
    if window_ends is not None:
        fieldnames.append("window_end")

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(labels)):
            row = {
                "index": i,
                "label": int(labels[i]),
                "pred": int(preds[i]),
                "prob_fall": float(probs[i]),
            }
            if subject_ids is not None:
                row["subject_id"] = subject_ids[i]
            if scene_ids is not None:
                row["scene_id"] = scene_ids[i]
            if window_starts is not None:
                row["window_start"] = int(window_starts[i])
            if window_ends is not None:
                row["window_end"] = int(window_ends[i])

            writer.writerow(row)


def run_external_test(cfg: ExternalTestConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device)

    data = np.load(cfg.npz_path, allow_pickle=True)
    acc = data["acc"]
    gyro = data["gyro"]
    labels = data["labels"]

    print("=" * 80)
    print("AIHUB EXTERNAL TEST DATA SUMMARY")
    print(f"NPZ path: {cfg.npz_path}")
    print(f"Model path: {cfg.model_path}")
    print(f"acc shape: {acc.shape}")
    print(f"gyro shape: {gyro.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"Fall windows/samples: {int((labels == 1).sum())}")
    print(f"Non-fall windows/samples: {int((labels == 0).sum())}")
    if "subject_ids" in data.files:
        print(f"Subjects: {len(np.unique(data['subject_ids']))}")
    if "scene_ids" in data.files:
        print(f"Scenes: {len(np.unique(data['scene_ids']))}")
    print("=" * 80)

    loader = make_loader(
        acc,
        gyro,
        labels,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = DSCS(
        num_classes=2,
        in_channels=3,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
    ).to(device)

    model = load_state_dict_safely(model, cfg.model_path, device)

    metrics, preds, true_labels, probs = evaluate_external(model, loader, device)

    print("=" * 80)
    print("AIHUB EXTERNAL TEST RESULT")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k:12s}: {v}")

    metrics_path = os.path.join(cfg.save_dir, "aihub_external_test_metrics.csv")
    preds_path = os.path.join(cfg.save_dir, "aihub_external_test_predictions.csv")

    save_metrics_csv(metrics, metrics_path)
    save_predictions_csv(preds, true_labels, probs, data, preds_path)

    print("=" * 80)
    print(f"Saved metrics:     {metrics_path}")
    print(f"Saved predictions: {preds_path}")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    cfg = ExternalTestConfig(
        npz_path="./aihub_cane_adult2_window.npz",
        model_path="./combined_eval_outputs/combined-holdout-80-20_best_model.pth",
        save_dir="./aihub_external_test_outputs",
        batch_size=128,
        num_workers=0,
        dropout=0.5,
        base_channels=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    run_external_test(cfg)
