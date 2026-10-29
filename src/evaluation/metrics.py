"""
TRUST-X Evaluation Metrics Module
Computes:
- Accuracy
- Precision (Macro & per-class)
- Recall (Macro & per-class)
- F1-Score (Macro & per-class)
- Confusion Matrix (raw counts and normalized, with plotting)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def compute_all_metrics(y_true, y_pred, labels=[0, 1], target_names=["Non-Harmful", "Harmful"]):
    """
    Computes standard performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class scores
    prec_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics_dict = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_class_0": float(f1_per_class[0]),
        "f1_class_1": float(f1_per_class[1]),
        "precision_class_0": float(prec_per_class[0]),
        "precision_class_1": float(prec_per_class[1]),
        "recall_class_0": float(rec_per_class[0]),
        "recall_class_1": float(rec_per_class[1]),
        "confusion_matrix": cm.tolist()
    }

    return metrics_dict, cm

def plot_confusion_matrix(cm, language, model_name, output_path, target_names=["Non-Harmful", "Harmful"]):
    """
    Generates high-resolution publication-quality confusion matrix figure.
    Displays both absolute counts and normalized percentages.
    """
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(6, 5))
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_norm[i, j] * 100
            annot[i, j] = f"{c}\n({p:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=target_names,
        yticklabels=target_names,
        annot_kws={"size": 11, "weight": "bold"}
    )

    plt.title(f"Confusion Matrix: {model_name} ({language.capitalize()})", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("True Class", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved confusion matrix plot to: {output_path}")
