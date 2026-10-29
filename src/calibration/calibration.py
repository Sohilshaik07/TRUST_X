"""
TRUST-X Calibration Module
Calculates:
1. Expected Calibration Error (ECE) with equal-width binning
2. Brier Score
3. Reliability Diagrams (Calibration Curves)
4. Temperature Scaling (fitted on validation set, applied to test set)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for binary classification.
    probs: array-like of predicted probabilities for the positive class (class 1)
           OR 2D array of class probabilities (N, 2).
    labels: array-like of ground truth binary labels (0 or 1)
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.ndim == 2:
        # Confidence is max predicted probability, prediction is argmax
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
    else:
        # Binary case with prob of class 1
        predictions = (probs >= 0.5).astype(int)
        confidences = np.where(predictions == 1, probs, 1.0 - probs)
        accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if in_bin.sum() > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            bin_data.append({
                "bin": i,
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "count": int(in_bin.sum()),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "gap": float(abs_diff)
            })
        else:
            bin_data.append({
                "bin": i,
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
                "gap": 0.0
            })

    return float(ece), bin_data

def compute_brier_score(probs, labels):
    """
    Computes Brier score for binary classification.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim == 2:
        prob_pos = probs[:, 1]
    else:
        prob_pos = probs
    return float(brier_score_loss(labels, prob_pos))

def plot_reliability_diagram(bin_data, ece, brier, language, model_name, output_path):
    """
    Plots high-resolution Reliability Diagram with confidence bars, accuracy bars, and gap.
    """
    df_bins = pd.DataFrame(bin_data)
    valid_bins = df_bins[df_bins['count'] > 0].copy()

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    # Bar plot of accuracy vs confidence
    centers = (df_bins['bin_lower'] + df_bins['bin_upper']) / 2.0
    width = 1.0 / len(df_bins)

    plt.bar(
        centers,
        df_bins['accuracy'],
        width=width * 0.9,
        alpha=0.6,
        color="#3a86ff",
        edgecolor="#1d3557",
        label="Observed Accuracy"
    )

    # Plot error gap
    for _, row in valid_bins.iterrows():
        c = (row['bin_lower'] + row['bin_upper']) / 2.0
        plt.plot([c, c], [row['accuracy'], row['confidence']], color="#e63946", linewidth=2.5)

    plt.plot([], [], color="#e63946", linewidth=2.5, label="Calibration Gap")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean Predicted Confidence", fontsize=11, fontweight='bold')
    plt.ylabel("Observed Empirical Accuracy", fontsize=11, fontweight='bold')
    plt.title(f"Reliability Diagram: {model_name} ({language.capitalize()})\nECE = {ece:.4f} | Brier = {brier:.4f}", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved reliability diagram to: {output_path}")

class ModelWithTemperature(nn.Module):
    """
    A decorator which wraps a model with temperature scaling.
    Temperature T is learned on the validation set only (Rule 4 & 5).
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs).logits
        return logits / self.temperature

    def set_temperature_from_logits(self, val_logits, val_labels, lr=0.01, max_iter=50):
        """
        Optimizes temperature T using NLL on validation logits ONLY.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits = torch.tensor(val_logits, dtype=torch.float32)
        labels = torch.tensor(val_labels, dtype=torch.long)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"[*] Optimal temperature T learned on validation set: {self.temperature.item():.4f}")
        return float(self.temperature.item())
