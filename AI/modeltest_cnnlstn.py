import os
import math
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        x = self.global_pool(x).squeeze(-1)
        return x


class DSCS(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3, base_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.acc_stream = SensorStreamCNN(in_channels, base_channels)
        self.gyro_stream = SensorStreamCNN(in_channels, base_channels)

        self.acc_attn = FeatureSelfAttention(base_channels)
        self.gyro_attn = FeatureSelfAttention(base_channels)

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
# Baseline Models: input = concat(acc, gyro) -> (B, T, 6)
# ============================================================
class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        x = torch.cat([acc, gyro], dim=2)   # (B, T, 6)
        x = x.transpose(1, 2)               # (B, 6, T)
        x = self.features(x)
        return self.classifier(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        x = torch.cat([acc, gyro], dim=2)   # (B, T, 6)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)


class CNNLSTM(nn.Module):
    def __init__(self, in_channels: int = 6, lstm_hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        x = torch.cat([acc, gyro], dim=2)   # (B, T, 6)
        x = x.transpose(1, 2)               # (B, 6, T)
        x = self.cnn(x)                     # (B, 64, T')
        x = x.transpose(1, 2)               # (B, T', 64)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)


def build_model(model_name: str, dropout: float = 0.5, base_channels: int = 64):
    model_name = model_name.lower()
    if model_name == "dscs":
        return DSCS(num_classes=2, in_channels=3, base_channels=base_channels, dropout=dropout)
    if model_name == "cnn":
        return CNN1D(in_channels=6, num_classes=2)
    if model_name == "lstm":
        return LSTMModel(input_dim=6, hidden_dim=64, num_layers=2, num_classes=2)
    if model_name == "cnn_lstm":
        return CNNLSTM(in_channels=6, lstm_hidden=64, num_classes=2)
    raise ValueError(f"Unknown model_name: {model_name}")


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds_all = []
    labels_all = []

    for acc, gyro, labels in loader:
        acc = acc.to(device)
        gyro = gyro.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(acc, gyro)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        preds_all.append(preds.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    metrics = compute_binary_metrics(preds_all, labels_all)
    metrics["loss"] = running_loss / len(labels_all)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all = []
    labels_all = []

    for acc, gyro, labels in loader:
        acc = acc.to(device)
        gyro = gyro.to(device)
        labels = labels.to(device)

        logits = model(acc, gyro)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        preds_all.append(preds.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    metrics = compute_binary_metrics(preds_all, labels_all)
    metrics["loss"] = running_loss / len(labels_all)
    return metrics


# ============================================================
# LOSO
# ============================================================
@dataclass
class LOSOConfig:
    data_path: str = "./sisfall_data.npz"
    model_name: str = "dscs"   # dscs / cnn / lstm / cnn_lstm
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 20
    dropout: float = 0.5
    base_channels: int = 64
    seed: int = 42
    num_workers: int = 0
    save_dir: str = "./loso_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_loso(cfg: LOSOConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)

    data = np.load(cfg.data_path, allow_pickle=True)
    acc = data["acc"]
    gyro = data["gyro"]
    labels = data["labels"]
    subject_ids = data["subject_ids"]

    unique_subjects = np.unique(subject_ids)
    print("Unique subjects:", unique_subjects)
    print("Total subjects:", len(unique_subjects))

    all_fold_results = []

    for fold_idx, test_subject in enumerate(unique_subjects, start=1):
        print("=" * 80)
        print(f"[Fold {fold_idx}/{len(unique_subjects)}] TEST SUBJECT = {test_subject}")

        train_mask = subject_ids != test_subject
        test_mask = subject_ids == test_subject

        x_acc_train = acc[train_mask]
        x_gyro_train = gyro[train_mask]
        y_train = labels[train_mask]

        x_acc_test = acc[test_mask]
        x_gyro_test = gyro[test_mask]
        y_test = labels[test_mask]

        if (y_test == 1).sum() == 0:
            print(f"Skip subject {test_subject} (no fall in test set)")
            continue

        train_dataset = FallDataset(x_acc_train, x_gyro_train, y_train)
        test_dataset = FallDataset(x_acc_test, x_gyro_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = build_model(
            model_name=cfg.model_name,
            dropout=cfg.dropout,
            base_channels=cfg.base_channels
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        best_f1 = -1.0
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_metrics = evaluate(model, test_loader, criterion, device)

            print(
                f"[{cfg.model_name.upper()} | Fold {fold_idx:02d} | Epoch {epoch:03d}] "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"train_f1={train_metrics['f1']:.4f} | "
                f"test_acc={test_metrics['accuracy']:.4f} "
                f"test_recall={test_metrics['recall']:.4f} "
                f"test_precision={test_metrics['precision']:.4f} "
                f"test_specificity={test_metrics['specificity']:.4f} "
                f"test_f1={test_metrics['f1']:.4f}"
            )

            if test_metrics["f1"] > best_f1:
                best_f1 = test_metrics["f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        final_metrics = evaluate(model, test_loader, criterion, device)
        final_metrics["test_subject"] = int(test_subject)
        all_fold_results.append(final_metrics)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "test_subject": int(test_subject),
                "metrics": final_metrics,
                "model_name": cfg.model_name,
            },
            os.path.join(cfg.save_dir, f"{cfg.model_name}_subject_{int(test_subject)}.pt"),
        )

        print(f"Best test metrics for subject {test_subject}: {final_metrics}")

    print("=" * 80)
    print(f"{cfg.model_name.upper()} LOSO FINAL SUMMARY")

    metric_names = ["accuracy", "recall", "precision", "specificity", "f1"]
    summary = {}
    for name in metric_names:
        values = np.array([fold[name] for fold in all_fold_results], dtype=np.float32)
        summary[name] = (values.mean(), values.std())
        print(f"{name:12s}: {values.mean():.4f} ± {values.std():.4f}")



    # ============================================================
    # Save CSV
    # ============================================================
    fold_csv_path = os.path.join(cfg.save_dir, "fold_results.csv")
    summary_csv_path = os.path.join(cfg.save_dir, "summary.csv")

    # ---- 1. Fold별 결과 저장 ----
    with open(fold_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        header = ["model", "subject", "accuracy", "recall", "precision", "specificity", "f1", "tp", "fp", "tn", "fn"]
        writer.writerow(header)

        for fold in all_fold_results:
            writer.writerow([
                cfg.model_name,
                fold["test_subject"],
                fold["accuracy"],
                fold["recall"],
                fold["precision"],
                fold["specificity"],
                fold["f1"],
                fold["tp"],
                fold["fp"],
                fold["tn"],
                fold["fn"],
            ])

    print(f"Saved fold results to: {fold_csv_path}")

    # ---- 2. Summary 저장 ----
    with open(summary_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["model", "metric", "mean", "std"])

        for name in metric_names:
            mean_val, std_val = summary[name]
            writer.writerow([
                cfg.model_name,
                name,
                mean_val,
                std_val,
            ])

    print(f"Saved summary to: {summary_csv_path}")

    return all_fold_results, summary


if __name__ == "__main__":
    for model_name in ["cnn", "lstm", "cnn_lstm"]:
        cfg = LOSOConfig(
            data_path="./sisfall_data.npz",
            model_name=model_name,
            batch_size=128,
            lr=1e-3,
            epochs=10,
            dropout=0.5,
            base_channels=64,
            seed=42,
            num_workers=0,
            save_dir=f"./loso_outputs_{model_name}",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        run_loso(cfg)