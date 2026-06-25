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
        x = self.global_pool(x).squeeze(-1)
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
# Config
# ============================================================
@dataclass
class MobiFallEvalConfig:
    data_path: str = "./mobifall_data.npz"
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 20
    dropout: float = 0.5
    base_channels: int = 64
    seed: int = 42
    num_workers: int = 0
    save_dir: str = "./mobifall_eval_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # hold-out split
    test_size: float = 0.2
    use_group_holdout: bool = True

    # CV
    use_group_kfold: bool = True

    # imbalance 대응
    use_class_weight: bool = True


# ============================================================
# Helpers
# ============================================================
def make_loader(x_acc, x_gyro, y, batch_size, num_workers, shuffle):
    ds = FallDataset(x_acc, x_gyro, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_criterion(y_train: np.ndarray, device: torch.device, use_class_weight: bool):
    if not use_class_weight:
        return nn.CrossEntropyLoss()

    n_neg = max(int((y_train == 0).sum()), 1)
    n_pos = max(int((y_train == 1).sum()), 1)

    # label 0 = ADL, label 1 = Fall
    weights = torch.tensor(
        [1.0, n_neg / n_pos],
        dtype=torch.float32,
        device=device
    )
    return nn.CrossEntropyLoss(weight=weights)


def train_and_eval_one_split(
    x_acc_train, x_gyro_train, y_train,
    x_acc_test, x_gyro_test, y_test,
    cfg: MobiFallEvalConfig,
    split_name: str = "split",
):
    device = torch.device(cfg.device)

    train_loader = make_loader(
        x_acc_train, x_gyro_train, y_train,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True
    )
    test_loader = make_loader(
        x_acc_test, x_gyro_test, y_test,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
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

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        print(
            f"[{split_name} | Epoch {epoch:03d}] "
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
            torch.save(
                best_state,
                os.path.join(cfg.save_dir, f"{split_name}_best_model.pth")
            )

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, criterion, device)
    return model, final_metrics


# ============================================================
# Hold-out
# ============================================================
def run_mobifall_holdout(cfg: MobiFallEvalConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    data = np.load(cfg.data_path, allow_pickle=True)
    acc = data["acc"]
    gyro = data["gyro"]
    labels = data["labels"]
    subject_ids = data["subject_ids"]

    print("Total samples:", len(labels))
    print("Total subjects:", len(np.unique(subject_ids)))

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
    print("MOBIFALL HOLD-OUT SPLIT")
    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    print(f"Train label counts -> Fall: {(y_train == 1).sum()}, ADL: {(y_train == 0).sum()}")
    print(f"Test  label counts -> Fall: {(y_test == 1).sum()}, ADL: {(y_test == 0).sum()}")

    _, metrics = train_and_eval_one_split(
        x_acc_train, x_gyro_train, y_train,
        x_acc_test, x_gyro_test, y_test,
        cfg,
        split_name="mobifall-holdout-80-20",
    )

    print("=" * 80)
    print("FINAL HOLD-OUT METRICS")
    for k in ["accuracy", "recall", "precision", "specificity", "f1", "tp", "fp", "tn", "fn"]:
        print(f"{k:12s}: {metrics[k]}")

    with open(os.path.join(cfg.save_dir, "holdout_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    return metrics


# ============================================================
# CV
# ============================================================
def run_mobifall_cv(cfg: MobiFallEvalConfig, n_splits: int = 5):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    data = np.load(cfg.data_path, allow_pickle=True)
    acc = data["acc"]
    gyro = data["gyro"]
    labels = data["labels"]
    subject_ids = data["subject_ids"]

    if cfg.use_group_kfold:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(acc, labels, groups=subject_ids)
        cv_name = f"group-{n_splits}fold"
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=cfg.seed,
        )
        split_iter = splitter.split(acc, labels)
        cv_name = f"stratified-{n_splits}fold"

    all_fold_results: List[Dict[str, float]] = []

    print("=" * 80)
    print(f"MOBIFALL {cv_name.upper()}")

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
        print(f"Train label counts -> Fall: {(y_train == 1).sum()}, ADL: {(y_train == 0).sum()}")
        print(f"Test  label counts -> Fall: {(y_test == 1).sum()}, ADL: {(y_test == 0).sum()}")

        _, fold_metrics = train_and_eval_one_split(
            x_acc_train, x_gyro_train, y_train,
            x_acc_test, x_gyro_test, y_test,
            cfg,
            split_name=f"{cv_name}-{fold_idx}",
        )

        fold_metrics["fold"] = fold_idx
        all_fold_results.append(fold_metrics)

        print(f"Best test metrics for fold {fold_idx}: {fold_metrics}")

    print("=" * 80)
    print(f"{cv_name.upper()} FINAL SUMMARY")
    metric_names = ["accuracy", "recall", "precision", "specificity", "f1"]
    for name in metric_names:
        values = np.array([fold[name] for fold in all_fold_results], dtype=np.float32)
        print(f"{name:12s}: {values.mean():.4f} ± {values.std():.4f}")

    csv_path = os.path.join(cfg.save_dir, f"{cv_name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fold_results[0].keys())
        writer.writeheader()
        writer.writerows(all_fold_results)

    return all_fold_results


if __name__ == "__main__":
    cfg = MobiFallEvalConfig(
        data_path="./mobifall_data.npz",
        batch_size=128,
        lr=1e-3,
        epochs=20,
        dropout=0.5,
        base_channels=64,
        seed=42,
        num_workers=0,
        save_dir="./mobifall_eval_outputs",
        device="cuda" if torch.cuda.is_available() else "cpu",
        test_size=0.2,
        use_group_holdout=True,
        use_group_kfold=True,
        use_class_weight=True,
    )

    # 1) subject-level 80/20 hold-out
    run_mobifall_holdout(cfg)

    # 2) subject-level 5-fold CV
    run_mobifall_cv(cfg, n_splits=5)

    # 3) subject-level 10-fold CV
    run_mobifall_cv(cfg, n_splits=10)