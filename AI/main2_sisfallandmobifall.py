import os
import math
import random
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    def __init__(
        self,
        acc_data: np.ndarray,
        gyro_data: np.ndarray,
        labels: np.ndarray,
        dataset_ids: np.ndarray,
    ):
        self.acc_data = torch.tensor(acc_data, dtype=torch.float32)
        self.gyro_data = torch.tensor(gyro_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.dataset_ids = torch.tensor(dataset_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.acc_data[idx],
            self.gyro_data[idx],
            self.labels[idx],
            self.dataset_ids[idx],
        )


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
class CombinedEvalConfig:
    sisfall_path: str = "./sisfall_data_8sec.npz"
    mobifall_path: str = "./mobifall_data.npz"

    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 30
    dropout: float = 0.5
    base_channels: int = 64
    seed: int = 42
    num_workers: int = 0
    save_dir: str = "./combined_eval_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    test_size: float = 0.2
    use_group_holdout: bool = True
    use_group_kfold: bool = True
    use_class_weight: bool = True

    # MobiFall subject id offset to avoid collision
    mobifall_subject_offset: int = 1000


# ============================================================
# Data Merge
# ============================================================
def load_and_merge_datasets(cfg: CombinedEvalConfig):
    sis = np.load(cfg.sisfall_path, allow_pickle=True)
    mobi = np.load(cfg.mobifall_path, allow_pickle=True)

    sis_acc = sis["acc"]
    sis_gyro = sis["gyro"]
    sis_labels = sis["labels"]
    sis_subject_ids = sis["subject_ids"]

    mobi_acc = mobi["acc"]
    mobi_gyro = mobi["gyro"]
    mobi_labels = mobi["labels"]
    mobi_subject_ids = mobi["subject_ids"] + cfg.mobifall_subject_offset

    if sis_acc.shape[1:] != mobi_acc.shape[1:]:
        raise ValueError(
            f"Sequence shape mismatch: SisFall {sis_acc.shape[1:]} vs MobiFall {mobi_acc.shape[1:]}"
        )

    acc = np.concatenate([sis_acc, mobi_acc], axis=0)
    gyro = np.concatenate([sis_gyro, mobi_gyro], axis=0)
    labels = np.concatenate([sis_labels, mobi_labels], axis=0)
    subject_ids = np.concatenate([sis_subject_ids, mobi_subject_ids], axis=0)

    dataset_ids = np.concatenate([
        np.zeros(len(sis_labels), dtype=np.int64),   # 0 = SisFall
        np.ones(len(mobi_labels), dtype=np.int64),   # 1 = MobiFall
    ], axis=0)

    print("=" * 80)
    print("COMBINED DATASET SUMMARY")
    print(f"Total samples: {len(labels)}")
    print(f"Total subjects: {len(np.unique(subject_ids))}")
    print(f"SisFall samples: {len(sis_labels)} | subjects: {len(np.unique(sis_subject_ids))}")
    print(f"MobiFall samples: {len(mobi_labels)} | subjects: {len(np.unique(mobi_subject_ids))}")
    print(f"Combined acc shape: {acc.shape}")
    print(f"Combined gyro shape: {gyro.shape}")
    print(f"Combined label counts -> Fall: {(labels == 1).sum()}, ADL: {(labels == 0).sum()}")
    print("=" * 80)

    return acc, gyro, labels, subject_ids, dataset_ids


# ============================================================
# Helpers
# ============================================================
def make_loader(x_acc, x_gyro, y, dset_ids, batch_size, num_workers, shuffle, drop_last=False):
    ds = FallDataset(x_acc, x_gyro, y, dset_ids)
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
    return nn.CrossEntropyLoss(weight=class_weights)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds_all = []
    labels_all = []

    for acc, gyro, labels, _dataset_ids in loader:
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
    dataset_ids_all = []

    for acc, gyro, labels, dataset_ids in loader:
        acc = acc.to(device)
        gyro = gyro.to(device)
        labels = labels.to(device)

        logits = model(acc, gyro)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        preds_all.append(preds.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        dataset_ids_all.append(dataset_ids.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    dataset_ids_all = np.concatenate(dataset_ids_all)

    total_metrics = compute_binary_metrics(preds_all, labels_all)
    total_metrics["loss"] = running_loss / len(labels_all)

    sis_mask = dataset_ids_all == 0
    mobi_mask = dataset_ids_all == 1

    sis_metrics = compute_binary_metrics(preds_all[sis_mask], labels_all[sis_mask]) if sis_mask.any() else None
    mobi_metrics = compute_binary_metrics(preds_all[mobi_mask], labels_all[mobi_mask]) if mobi_mask.any() else None

    return total_metrics, sis_metrics, mobi_metrics


def save_epoch_log_csv(rows: List[Dict], save_path: str):
    if not rows:
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_final_metrics_csv(rows: List[Dict], save_path: str):
    if not rows:
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Train / Eval One Split
# ============================================================
def train_and_eval_one_split(
    x_acc_train, x_gyro_train, y_train, d_train,
    x_acc_test, x_gyro_test, y_test, d_test,
    cfg: CombinedEvalConfig,
    split_name: str = "split",
):
    device = torch.device(cfg.device)

    train_loader = make_loader(
        x_acc_train, x_gyro_train, y_train, d_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,  # 추가
    )
    test_loader = make_loader(
        x_acc_test, x_gyro_test, y_test, d_test,
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
    best_total_metrics = None
    best_sis_metrics = None
    best_mobi_metrics = None

    epoch_log_rows: List[Dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        total_metrics, sis_metrics, mobi_metrics = evaluate(model, test_loader, criterion, device)

        row = {
            "split": split_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_recall": train_metrics["recall"],
            "train_precision": train_metrics["precision"],
            "train_specificity": train_metrics["specificity"],
            "train_f1": train_metrics["f1"],
            "test_loss_total": total_metrics["loss"],
            "test_accuracy_total": total_metrics["accuracy"],
            "test_recall_total": total_metrics["recall"],
            "test_precision_total": total_metrics["precision"],
            "test_specificity_total": total_metrics["specificity"],
            "test_f1_total": total_metrics["f1"],
            "tp_total": total_metrics["tp"],
            "fp_total": total_metrics["fp"],
            "tn_total": total_metrics["tn"],
            "fn_total": total_metrics["fn"],
        }

        if sis_metrics is not None:
            row.update({
                "test_accuracy_sisfall": sis_metrics["accuracy"],
                "test_recall_sisfall": sis_metrics["recall"],
                "test_precision_sisfall": sis_metrics["precision"],
                "test_specificity_sisfall": sis_metrics["specificity"],
                "test_f1_sisfall": sis_metrics["f1"],
                "tp_sisfall": sis_metrics["tp"],
                "fp_sisfall": sis_metrics["fp"],
                "tn_sisfall": sis_metrics["tn"],
                "fn_sisfall": sis_metrics["fn"],
            })
        else:
            row.update({
                "test_accuracy_sisfall": None,
                "test_recall_sisfall": None,
                "test_precision_sisfall": None,
                "test_specificity_sisfall": None,
                "test_f1_sisfall": None,
                "tp_sisfall": None,
                "fp_sisfall": None,
                "tn_sisfall": None,
                "fn_sisfall": None,
            })

        if mobi_metrics is not None:
            row.update({
                "test_accuracy_mobifall": mobi_metrics["accuracy"],
                "test_recall_mobifall": mobi_metrics["recall"],
                "test_precision_mobifall": mobi_metrics["precision"],
                "test_specificity_mobifall": mobi_metrics["specificity"],
                "test_f1_mobifall": mobi_metrics["f1"],
                "tp_mobifall": mobi_metrics["tp"],
                "fp_mobifall": mobi_metrics["fp"],
                "tn_mobifall": mobi_metrics["tn"],
                "fn_mobifall": mobi_metrics["fn"],
            })
        else:
            row.update({
                "test_accuracy_mobifall": None,
                "test_recall_mobifall": None,
                "test_precision_mobifall": None,
                "test_specificity_mobifall": None,
                "test_f1_mobifall": None,
                "tp_mobifall": None,
                "fp_mobifall": None,
                "tn_mobifall": None,
                "fn_mobifall": None,
            })

        epoch_log_rows.append(row)

        print(
            f"[{split_name} | Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} | "
            f"test_loss={total_metrics['loss']:.4f} "
            f"test_acc={total_metrics['accuracy']:.4f} "
            f"test_recall={total_metrics['recall']:.4f} "
            f"test_precision={total_metrics['precision']:.4f} "
            f"test_specificity={total_metrics['specificity']:.4f} "
            f"test_f1={total_metrics['f1']:.4f} "
            f"| TP={total_metrics['tp']} FP={total_metrics['fp']} "
            f"TN={total_metrics['tn']} FN={total_metrics['fn']}"
        )

        if sis_metrics is not None:
            print(
                f"    SisFall  -> acc={sis_metrics['accuracy']:.4f}, "
                f"recall={sis_metrics['recall']:.4f}, "
                f"precision={sis_metrics['precision']:.4f}, "
                f"f1={sis_metrics['f1']:.4f}"
            )

        if mobi_metrics is not None:
            print(
                f"    MobiFall -> acc={mobi_metrics['accuracy']:.4f}, "
                f"recall={mobi_metrics['recall']:.4f}, "
                f"precision={mobi_metrics['precision']:.4f}, "
                f"f1={mobi_metrics['f1']:.4f}"
            )

        if total_metrics["f1"] > best_f1:
            best_f1 = total_metrics["f1"]
            best_epoch = epoch
            best_total_metrics = dict(total_metrics)
            best_sis_metrics = dict(sis_metrics) if sis_metrics is not None else None
            best_mobi_metrics = dict(mobi_metrics) if mobi_metrics is not None else None
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            torch.save(
                best_state,
                os.path.join(cfg.save_dir, f"{split_name}_best_model.pth")
            )

    epoch_csv_path = os.path.join(cfg.save_dir, f"{split_name}_epoch_log.csv")
    save_epoch_log_csv(epoch_log_rows, epoch_csv_path)

    if best_state is None:
        raise RuntimeError(f"No best model was saved for split: {split_name}")

    model.load_state_dict(best_state)

    print("=" * 80)
    print(f"[{split_name}] BEST EPOCH = {best_epoch}")
    print(f"[{split_name}] BEST TOTAL METRICS = {best_total_metrics}")
    print(f"[{split_name}] BEST SISFALL METRICS = {best_sis_metrics}")
    print(f"[{split_name}] BEST MOBIFALL METRICS = {best_mobi_metrics}")
    print(f"[{split_name}] Saved best model to: {os.path.join(cfg.save_dir, f'{split_name}_best_model.pth')}")
    print(f"[{split_name}] Saved epoch log to:   {epoch_csv_path}")
    print("=" * 80)

    return model, best_epoch, best_total_metrics, best_sis_metrics, best_mobi_metrics


# ============================================================
# Hold-out
# ============================================================
def run_combined_holdout(cfg: CombinedEvalConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    acc, gyro, labels, subject_ids, dataset_ids = load_and_merge_datasets(cfg)

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
    d_train = dataset_ids[train_idx]

    x_acc_test = acc[test_idx]
    x_gyro_test = gyro[test_idx]
    y_test = labels[test_idx]
    d_test = dataset_ids[test_idx]

    train_subjects = np.unique(subject_ids[train_idx])
    test_subjects = np.unique(subject_ids[test_idx])

    print("=" * 80)
    print("COMBINED HOLD-OUT SPLIT")
    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    print(
        f"Train label counts -> Fall: {(y_train == 1).sum()}, ADL: {(y_train == 0).sum()} "
        f"(fall ratio={(y_train == 1).mean():.4f})"
    )
    print(
        f"Test  label counts -> Fall: {(y_test == 1).sum()}, ADL: {(y_test == 0).sum()} "
        f"(fall ratio={(y_test == 1).mean():.4f})"
    )
    print(
        f"Train dataset counts -> SisFall: {(d_train == 0).sum()}, MobiFall: {(d_train == 1).sum()}"
    )
    print(
        f"Test  dataset counts -> SisFall: {(d_test == 0).sum()}, MobiFall: {(d_test == 1).sum()}"
    )

    _, best_epoch, total_metrics, sis_metrics, mobi_metrics = train_and_eval_one_split(
        x_acc_train, x_gyro_train, y_train, d_train,
        x_acc_test, x_gyro_test, y_test, d_test,
        cfg,
        split_name="combined-holdout-80-20",
    )

    final_rows = [
        {
            "split": "combined-holdout-80-20",
            "best_epoch": best_epoch,
            "scope": "total",
            **total_metrics,
        },
        {
            "split": "combined-holdout-80-20",
            "best_epoch": best_epoch,
            "scope": "sisfall",
            **(sis_metrics if sis_metrics is not None else {}),
        },
        {
            "split": "combined-holdout-80-20",
            "best_epoch": best_epoch,
            "scope": "mobifall",
            **(mobi_metrics if mobi_metrics is not None else {}),
        },
    ]

    save_final_metrics_csv(
        final_rows,
        os.path.join(cfg.save_dir, "combined_holdout_metrics.csv")
    )

    return total_metrics, sis_metrics, mobi_metrics


# ============================================================
# K-fold CV
# ============================================================
def run_combined_kfold_cv(cfg: CombinedEvalConfig, n_splits: int = 5):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    acc, gyro, labels, subject_ids, dataset_ids = load_and_merge_datasets(cfg)

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

    final_rows: List[Dict] = []

    print("=" * 80)
    print(f"COMBINED {cv_name.upper()}")

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        print("=" * 80)
        print(f"[Fold {fold_idx}/{n_splits}]")

        x_acc_train = acc[train_idx]
        x_gyro_train = gyro[train_idx]
        y_train = labels[train_idx]
        d_train = dataset_ids[train_idx]

        x_acc_test = acc[test_idx]
        x_gyro_test = gyro[test_idx]
        y_test = labels[test_idx]
        d_test = dataset_ids[test_idx]

        train_subjects = np.unique(subject_ids[train_idx])
        test_subjects = np.unique(subject_ids[test_idx])

        print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        print(
            f"Train label counts -> Fall: {(y_train == 1).sum()}, ADL: {(y_train == 0).sum()} "
            f"(fall ratio={(y_train == 1).mean():.4f})"
        )
        print(
            f"Test  label counts -> Fall: {(y_test == 1).sum()}, ADL: {(y_test == 0).sum()} "
            f"(fall ratio={(y_test == 1).mean():.4f})"
        )
        print(
            f"Train dataset counts -> SisFall: {(d_train == 0).sum()}, MobiFall: {(d_train == 1).sum()}"
        )
        print(
            f"Test  dataset counts -> SisFall: {(d_test == 0).sum()}, MobiFall: {(d_test == 1).sum()}"
        )

        _, best_epoch, total_metrics, sis_metrics, mobi_metrics = train_and_eval_one_split(
            x_acc_train, x_gyro_train, y_train, d_train,
            x_acc_test, x_gyro_test, y_test, d_test,
            cfg,
            split_name=f"{cv_name}-{fold_idx}",
        )

        final_rows.append({
            "split": f"{cv_name}-{fold_idx}",
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "scope": "total",
            **total_metrics,
        })

        if sis_metrics is not None:
            final_rows.append({
                "split": f"{cv_name}-{fold_idx}",
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "scope": "sisfall",
                **sis_metrics,
            })

        if mobi_metrics is not None:
            final_rows.append({
                "split": f"{cv_name}-{fold_idx}",
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "scope": "mobifall",
                **mobi_metrics,
            })

    metrics_csv_path = os.path.join(cfg.save_dir, f"{cv_name}_metrics.csv")
    save_final_metrics_csv(final_rows, metrics_csv_path)

    print("=" * 80)
    print(f"Saved final fold metrics to: {metrics_csv_path}")

    for scope in ["total", "sisfall", "mobifall"]:
        scope_rows = [r for r in final_rows if r["scope"] == scope]
        if not scope_rows:
            continue
        print(f"\n[{scope.upper()}] SUMMARY")
        for metric_name in ["accuracy", "recall", "precision", "specificity", "f1"]:
            values = np.array([r[metric_name] for r in scope_rows], dtype=np.float32)
            print(f"{metric_name:12s}: {values.mean():.4f} ± {values.std():.4f}")

    return final_rows


if __name__ == "__main__":
    cfg = CombinedEvalConfig(
        sisfall_path="./sisfall_data_8sec.npz",
        mobifall_path="./mobifall_data.npz",
        batch_size=128,
        lr=1e-3,
        epochs=30,
        dropout=0.5,
        base_channels=64,
        seed=42,
        num_workers=0,
        save_dir="./combined_eval_outputs",
        device="cuda" if torch.cuda.is_available() else "cpu",
        test_size=0.2,
        use_group_holdout=True,
        use_group_kfold=True,
        use_class_weight=True,
        mobifall_subject_offset=1000,
    )

    # 1) hold-out
    run_combined_holdout(cfg)

    # 2) 5-fold CV
    run_combined_kfold_cv(cfg, n_splits=5)

    # 3) 10-fold CV
    run_combined_kfold_cv(cfg, n_splits=10)