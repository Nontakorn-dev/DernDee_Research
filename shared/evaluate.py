"""Shared classification metrics for gait phase models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gait_labels import PHASE_NAMES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(4))
    target_names = [PHASE_NAMES[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    phase_accuracy = np.divide(
        np.diag(cm),
        support,
        out=np.zeros(len(labels), dtype=np.float64),
        where=support != 0,
    )
    phase_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "phase_f1": {PHASE_NAMES[i]: float(phase_f1[i]) for i in labels},
        "phase_accuracy": {PHASE_NAMES[i]: float(phase_accuracy[i]) for i in labels},
        "phase_support": {PHASE_NAMES[i]: int(support[i]) for i in labels},
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        ),
    }


def evaluate_single(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    desc: str = "eval",
) -> dict:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, unit="batch", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(yb.cpu().tolist())

    y_true = np.array(labels)
    y_pred = np.array(preds)
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(y_true)
    return metrics


def evaluate_dual(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    weight_lt: torch.Tensor,
    weight_rt: torch.Tensor,
    desc: str = "eval",
) -> dict:
    model.eval()
    preds_lt: list[int] = []
    preds_rt: list[int] = []
    labels_lt: list[int] = []
    labels_rt: list[int] = []
    total_loss = 0.0
    criterion_lt = nn.CrossEntropyLoss(weight=weight_lt)
    criterion_rt = nn.CrossEntropyLoss(weight=weight_rt)

    with torch.no_grad():
        for xb, y_lt, y_rt in tqdm(loader, desc=desc, unit="batch", leave=False):
            xb = xb.to(device)
            y_lt = y_lt.to(device)
            y_rt = y_rt.to(device)
            logits_lt, logits_rt = model(xb)
            loss = (criterion_lt(logits_lt, y_lt) + criterion_rt(logits_rt, y_rt)) / 2
            total_loss += loss.item() * len(y_lt)
            preds_lt.extend(logits_lt.argmax(dim=1).cpu().tolist())
            preds_rt.extend(logits_rt.argmax(dim=1).cpu().tolist())
            labels_lt.extend(y_lt.cpu().tolist())
            labels_rt.extend(y_rt.cpu().tolist())

    y_lt = np.array(labels_lt)
    y_rt = np.array(labels_rt)
    m_lt = classification_metrics(y_lt, np.array(preds_lt))
    m_rt = classification_metrics(y_rt, np.array(preds_rt))
    return {
        "loss": total_loss / len(y_lt),
        "accuracy": (m_lt["accuracy"] + m_rt["accuracy"]) / 2,
        "macro_f1": (m_lt["macro_f1"] + m_rt["macro_f1"]) / 2,
        "left": m_lt,
        "right": m_rt,
    }
