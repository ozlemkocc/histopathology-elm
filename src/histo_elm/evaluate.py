"""Metrics, reports and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def text_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    return classification_report(y_true, y_pred, labels=range(len(class_names)),
                                 target_names=class_names, digits=4, zero_division=0)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
                          title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    size = max(5, 1.3 * len(class_names))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    threshold = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_summary_plot(summary: pd.DataFrame, title: str, path: Path) -> None:
    metrics = ["accuracy", "f1_macro", "cohen_kappa"]
    data = summary.set_index("model")[metrics]
    ax = data.plot.bar(figsize=(10, 5), width=0.8, rot=30)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(["Accuracy", "F1 (macro)", "Cohen's kappa"], loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
