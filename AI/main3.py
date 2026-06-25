import os
import math
import random
import csv
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedKFold


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
        # x: (B, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) / math.sqrt(self.dim)  # (B, D, D)
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
        x = self.global_pool(x).squeeze(-1)  # (B, C)
        return x


class DSCS(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3, base_channels: int = 64, dropout: float = 0.5):
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
class AIHubConfig:
    data_path: str = "./aihub_cane_adult2_window.npz"
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 30
    dropout: float = 0.5
    base_channels: int = 64
    seed: int = 42
    num_workers: int = 0
    save_dir: str = "./aihub_cane_adult2_window_outputs2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    test_size: float = 0.2
    use_group_holdout: bool = True
    use_group_kfold: bool = True
    use_class_weight: bool = True


# ============================================================
# Data Load
# ============================================================
def load_aihub_dataset(cfg: AIHubConfig):
    data = np.load(cfg.data_path, allow_pickle=True)

    acc = data["acc"]
    gyro = data["gyro"]
    labels = data["labels"]

    if "subject_ids" in data:
        subject_ids = data["subject_ids"]
    else:
        subject_ids = np.arange(len(labels))

    scene_ids = data["scene_ids"] if "scene_ids" in data else None

    print("=" * 80)
    print("AIHUB DATASET SUMMARY")
    print(f"NPZ path: {cfg.data_path}")
    print(f"Total samples/windows: {len(labels)}")
    print(f"acc shape: {acc.shape}")
    print(f"gyro shape: {gyro.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"Fall: {(labels == 1).sum()} | Non-fall: {(labels == 0).sum()}")
    print(f"Subjects: {len(np.unique(subject_ids))}")
    if scene_ids is not None:
        print(f"Scenes: {len(np.unique(scene_ids))}")
    print("=" * 80)

    return acc, gyro, labels, subject_ids, scene_ids


# ============================================================
# Helpers
# ============================================================
def make_loader(x_acc, x_gyro, y, batch_size, num_workers, shuffle, drop_last=False):
    ds = FallDataset(x_acc, x_gyro, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def make_criterion(y_train: np.ndarray, device: torch.device, use_class_weight: bool):
    if not use_class_weight:
        return nn.CrossEntropyLoss()

    n_neg = max(int((y_train == 0).sum()), 1)
    n_pos = max(int((y_train == 1).sum()), 1)
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    print(f"Class weights: non-fall={class_weights[0].item():.4f}, fall={class_weights[1].item():.4f}")
    return nn.CrossEntropyLoss(weight=class_weights)


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


def save_csv(rows: List[Dict], save_path: str):
    if not rows:
        return
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Train / Eval One Split
# ============================================================
def train_and_eval_one_split(
    x_acc_train, x_gyro_train, y_train,
    x_acc_test, x_gyro_test, y_test,
    cfg: AIHubConfig,
    split_name: str = "split",
):
    device = torch.device(cfg.device)

    train_loader = make_loader(
        x_acc_train, x_gyro_train, y_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )
    test_loader = make_loader(
        x_acc_test, x_gyro_test, y_test,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
    )

    model = DSCS(
        num_classes=2,
        in_channels=3,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
    ).to(device)

    criterion = make_criterion(y_train, device, cfg.use_class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    best_metrics = None
    epoch_log_rows: List[Dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        row = {
            "split": split_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_recall": train_metrics["recall"],
            "train_precision": train_metrics["precision"],
            "train_specificity": train_metrics["specificity"],
            "train_f1": train_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_specificity": test_metrics["specificity"],
            "test_f1": test_metrics["f1"],
            "tp": test_metrics["tp"],
            "fp": test_metrics["fp"],
            "tn": test_metrics["tn"],
            "fn": test_metrics["fn"],
        }
        epoch_log_rows.append(row)

        print(
            f"[{split_name} | Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"test_recall={test_metrics['recall']:.4f} "
            f"test_precision={test_metrics['precision']:.4f} "
            f"test_specificity={test_metrics['specificity']:.4f} "
            f"test_f1={test_metrics['f1']:.4f} "
            f"| TP={test_metrics['tp']} FP={test_metrics['fp']} "
            f"TN={test_metrics['tn']} FN={test_metrics['fn']}"
        )

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_epoch = epoch
            best_metrics = dict(test_metrics)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(cfg.save_dir, f"{split_name}_best_model.pth"))

    epoch_csv_path = os.path.join(cfg.save_dir, f"{split_name}_epoch_log.csv")
    save_csv(epoch_log_rows, epoch_csv_path)

    if best_state is None:
        raise RuntimeError(f"No best model was saved for split: {split_name}")

    model.load_state_dict(best_state)

    print("=" * 80)
    print(f"[{split_name}] BEST EPOCH = {best_epoch}")
    print(f"[{split_name}] BEST METRICS = {best_metrics}")
    print(f"[{split_name}] Saved best model to: {os.path.join(cfg.save_dir, f'{split_name}_best_model.pth')}")
    print(f"[{split_name}] Saved epoch log to:   {epoch_csv_path}")
    print("=" * 80)

    return model, best_epoch, best_metrics


# ============================================================
# Hold-out
# ============================================================
def run_aihub_holdout(cfg: AIHubConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    acc, gyro, labels, subject_ids, _scene_ids = load_aihub_dataset(cfg)

    if cfg.use_group_holdout:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.test_size,
            random_state=cfg.seed,
        )
        train_idx, test_idx = next(splitter.split(acc, labels, groups=subject_ids))
    else:
        indices = np.arange(len(labels))
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(indices)
        cut = int(len(indices) * (1.0 - cfg.test_size))
        train_idx, test_idx = indices[:cut], indices[cut:]

    x_acc_train = acc[train_idx]
    x_gyro_train = gyro[train_idx]
    y_train = labels[train_idx]

    x_acc_test = acc[test_idx]
    x_gyro_test = gyro[test_idx]
    y_test = labels[test_idx]

    train_subjects = np.unique(subject_ids[train_idx])
    test_subjects = np.unique(subject_ids[test_idx])

    print("=" * 80)
    print("AIHUB HOLD-OUT SPLIT")
    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    print(f"Train label counts -> Fall: {(y_train == 1).sum()}, Non-fall: {(y_train == 0).sum()}")
    print(f"Test  label counts -> Fall: {(y_test == 1).sum()}, Non-fall: {(y_test == 0).sum()}")
    print("=" * 80)

    _, best_epoch, metrics = train_and_eval_one_split(
        x_acc_train, x_gyro_train, y_train,
        x_acc_test, x_gyro_test, y_test,
        cfg,
        split_name="aihub-holdout-80-20",
    )

    final_rows = [{"split": "aihub-holdout-80-20", "best_epoch": best_epoch, **metrics}]
    save_csv(final_rows, os.path.join(cfg.save_dir, "aihub_holdout_metrics.csv"))

    return metrics


# ============================================================
# K-fold CV
# ============================================================
def run_aihub_kfold_cv(cfg: AIHubConfig, n_splits: int = 5):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    acc, gyro, labels, subject_ids, _scene_ids = load_aihub_dataset(cfg)

    if cfg.use_group_kfold:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(acc, labels, groups=subject_ids)
        cv_name = f"aihub-group-{n_splits}fold"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        split_iter = splitter.split(acc, labels)
        cv_name = f"aihub-stratified-{n_splits}fold"

    final_rows: List[Dict] = []

    print("=" * 80)
    print(f"AIHUB {cv_name.upper()}")

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        print("=" * 80)
        print(f"[Fold {fold_idx}/{n_splits}]")

        x_acc_train = acc[train_idx]
        x_gyro_train = gyro[train_idx]
        y_train = labels[train_idx]

        x_acc_test = acc[test_idx]
        x_gyro_test = gyro[test_idx]
        y_test = labels[test_idx]

        train_subjects = np.unique(subject_ids[train_idx])
        test_subjects = np.unique(subject_ids[test_idx])

        print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        print(f"Train label counts -> Fall: {(y_train == 1).sum()}, Non-fall: {(y_train == 0).sum()}")
        print(f"Test  label counts -> Fall: {(y_test == 1).sum()}, Non-fall: {(y_test == 0).sum()}")

        _, best_epoch, metrics = train_and_eval_one_split(
            x_acc_train, x_gyro_train, y_train,
            x_acc_test, x_gyro_test, y_test,
            cfg,
            split_name=f"{cv_name}-{fold_idx}",
        )

        final_rows.append({
            "split": f"{cv_name}-{fold_idx}",
            "fold": fold_idx,
            "best_epoch": best_epoch,
            **metrics,
        })

    metrics_csv_path = os.path.join(cfg.save_dir, f"{cv_name}_metrics.csv")
    save_csv(final_rows, metrics_csv_path)

    print("=" * 80)
    print(f"Saved final fold metrics to: {metrics_csv_path}")
    print("AIHUB K-FOLD FINAL SUMMARY")
    for metric_name in ["accuracy", "recall", "precision", "specificity", "f1"]:
        values = np.array([r[metric_name] for r in final_rows], dtype=np.float32)
        print(f"{metric_name:12s}: {values.mean():.4f} ± {values.std():.4f}")

    return final_rows


if __name__ == "__main__":
    cfg = AIHubConfig(
        data_path="./aihub_cane_adult2_window.npz",
        batch_size=128,
        lr=1e-3,
        epochs=30,
        dropout=0.5,
        base_channels=64,
        seed=42,
        num_workers=0,
        save_dir="./aihub_cane_adult2_window_outputs2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        test_size=0.2,
        use_group_holdout=True,
        use_group_kfold=True,
        use_class_weight=True,
    )

    # 1) AIHub 내부 80/20 subject-level hold-out
    run_aihub_holdout(cfg)

    # 2) AIHub 내부 5-fold CV
    run_aihub_kfold_cv(cfg, n_splits=5)

    # 3) AIHub 내부 10-fold CV
    # 데이터 subject 수가 너무 적거나 fold별 클래스가 깨지면 주석 처리해도 됨.
    run_aihub_kfold_cv(cfg, n_splits=10)
