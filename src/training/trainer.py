"""
TRUST-X Training Engine
Unified training, validation, and multi-dimensional trustworthiness evaluation:
- Automatic hardware acceleration (CUDA GPU detection with VRAM & name, or CPU fallback)
- Mixed Precision (FP16 via torch.cuda.amp) for GPU acceleration
- Gradient Accumulation for memory-constrained environments
- Class-Weighted Cross Entropy for imbalanced datasets (e.g. Malayalam)
- Validation after every epoch; best model selected ONLY on Validation Macro-F1
- End-to-end pipeline: Training -> Validation -> Test Evaluation -> Calibration -> Robustness -> SHAP -> TWI
- Saves checkpoints, predictions.csv, metrics.json, confusion matrix plots, and experiments.csv
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd

from src.models.model_factory import get_device, get_model_and_tokenizer
from src.data.loaders import get_dataloaders
from src.evaluation.metrics import compute_all_metrics, plot_confusion_matrix
from src.calibration.calibration import (
    compute_ece,
    compute_brier_score,
    plot_reliability_diagram,
    ModelWithTemperature
)
from src.robustness.evaluate import run_robustness_evaluation
from src.explainability.shap_analysis import select_representative_subset, run_shap_analysis
from src.twi.twi import (
    normalize_calibration_score,
    normalize_robustness_score,
    calculate_twi,
    run_twi_sensitivity_analysis
)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_weights(train_df, device):
    """
    Computes balanced class weights for Cross-Entropy loss:
    weight[c] = N / (n_classes * count[c])
    """
    labels = train_df['label'].values
    classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    weights = total_samples / (len(classes) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[*] Computed Class Weights for imbalanced loss: Class 0 = {weights[0]:.4f}, Class 1 = {weights[1]:.4f}")
    return class_weights

def train_and_evaluate(config):
    """
    Executes a complete TRUST-X experiment from configuration dictionary.
    """
    exp_id = config["experiment_id"]
    language = config["language"]
    model_key = config["model_key"]
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    seed = config.get("seed", 42)
    epochs = config.get("epochs", 3)
    batch_size = config.get("batch_size", 16)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    lr = float(config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.01))
    max_length = config.get("max_length", 128)
    use_class_weights = config.get("use_class_weights", True)
    run_shap = config.get("run_shap", True)
    run_calibration = config.get("run_calibration", True)
    run_robustness = config.get("run_robustness", True)
    run_twi = config.get("run_twi", True)
    require_cuda = config.get("require_cuda", False)

    if require_cuda and not torch.cuda.is_available():
        error_msg = "PILOT ABORTED — CUDA GPU NOT AVAILABLE"
        print(f"\n[!] CRITICAL ERROR: {error_msg}")
        print("    Actual training must be performed in Google Colab with GPU acceleration.")
        print("    Local environment is CPU-only. Aborting training per compute rules.\n")
        raise RuntimeError(error_msg)

    set_seed(seed)
    device = get_device(verbose=True)

    # 1. Print Experiment Configuration Banner
    print("\n" + "#" * 70)
    print(f"# EXPERIMENT: {exp_id}")
    print(f"# Language:      {language.upper()}")
    print(f"# Model:         {model_key.upper()}")
    print(f"# Device:        {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"# Batch Size:    {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"# Max Length:    {max_length}")
    print(f"# Learning Rate: {lr}")
    print(f"# Epochs:        {epochs}")
    print(f"# Seed:          {seed}")
    print("#" * 70 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "best_checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. Load Model & Tokenizer
    model, tokenizer, checkpoint_name = get_model_and_tokenizer(model_key, num_labels=2)
    model.to(device)

    # 3. Load Data
    dataloaders = get_dataloaders(data_dir, tokenizer, batch_size=batch_size, max_length=max_length)
    train_loader = dataloaders['train_loader']
    dev_loader = dataloaders['dev_loader']
    test_loader = dataloaders['test_loader']
    train_df = dataloaders['train_df']
    test_df = dataloaders['test_df']

    # 4. Loss Function & Optimizer
    if use_class_weights:
        weights = compute_class_weights(train_df, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5. Training Loop
    best_val_f1 = -1.0
    training_history = []
    start_train_time = time.time()

    print(f"[*] Starting fine-tuning for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * grad_accum_steps

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds, val_labels_list, val_probs_list = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                val_preds.extend(preds)
                val_probs_list.extend(probs)
                val_labels_list.extend(labels.cpu().numpy())

        val_metrics, _ = compute_all_metrics(np.array(val_labels_list), np.array(val_preds))
        val_f1 = val_metrics['f1_macro']
        val_acc = val_metrics['accuracy']

        print(f"    Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1
        })

        # Save Best Checkpoint based ONLY on Validation Macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"    [+] New best validation F1 ({best_val_f1:.4f}). Saving checkpoint to {ckpt_dir}...")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    train_duration = time.time() - start_train_time
    print(f"[*] Training finished in {train_duration:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}")

    # 6. Final Evaluation on Test Set (UNTOUCHED UNTIL NOW)
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (Using Best Validated Checkpoint)")
    print("=" * 60)
    # Reload from the HF-format checkpoint directory saved via save_pretrained()
    from transformers import AutoModelForSequenceClassification
    best_model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    best_model.to(device)
    best_model.eval()

    test_preds, test_labels_list, test_probs_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            test_preds.extend(preds)
            test_probs_list.extend(probs)
            test_labels_list.extend(labels.cpu().numpy())

    y_test_true = np.array(test_labels_list)
    y_test_pred = np.array(test_preds)
    test_probs = np.array(test_probs_list)

    # 7. Performance Metrics
    test_metrics, cm = compute_all_metrics(y_test_true, y_test_pred)
    cm_fig_path = os.path.join(output_dir, f"confusion_matrix_{exp_id}.png")
    plot_confusion_matrix(cm, language, model_key.upper(), cm_fig_path)

    # 8. Calibration Analysis
    if run_calibration:
        ece, bin_data = compute_ece(test_probs, y_test_true, n_bins=10)
        brier = compute_brier_score(test_probs, y_test_true)
        rel_diag_path = os.path.join(output_dir, f"reliability_diagram_{exp_id}.png")
        plot_reliability_diagram(bin_data, ece, brier, language, model_key.upper(), rel_diag_path)
        calib_score = normalize_calibration_score(ece, brier)
        print(f"[*] Calibration: ECE={ece:.4f}, Brier={brier:.4f}, Normalized Calib Score={calib_score:.4f}")
    else:
        ece, brier, bin_data, calib_score = None, None, [], None
        print("[*] Skipping Calibration analysis (Pilot mode)")

    # 9. Robustness Testing
    if run_robustness:
        print("\n[*] Running Linguistic Robustness Testing...")
        robustness_results = run_robustness_evaluation(
            best_model, tokenizer, test_df, device, max_length=max_length
        )
        rob_retention = robustness_results["overall_robustness_score"]
        rob_flip_rate = robustness_results["composite_flip_rate"]
        rob_score = normalize_robustness_score(rob_retention, rob_flip_rate)
        print(f"[*] Robustness: Retention={rob_retention:.4f}, FlipRate={rob_flip_rate:.4f}, Normalized Rob Score={rob_score:.4f}")
    else:
        robustness_results, rob_score = {}, None
        print("[*] Skipping Robustness testing (Pilot mode)")

    # 10. SHAP Explainability
    if run_shap:
        print("\n[*] Running SHAP Token Explainability...")
        rep_df = select_representative_subset(test_df, y_test_pred, test_probs, n_samples=40, seed=seed)
        shap_dir = os.path.join(output_dir, "shap_explanations")
        shap_res = run_shap_analysis(best_model, tokenizer, rep_df, device, shap_dir)
        expl_score = shap_res["mean_compactness_score"]
    else:
        expl_score = None
        print("[*] Skipping SHAP explainability (Pilot mode)")

    # 11. Trustworthiness Index (TWI)
    perf_score = test_metrics["f1_macro"]
    if run_twi and calib_score is not None and rob_score is not None:
        twi_default = calculate_twi(perf_score, expl_score if expl_score is not None else 0.5, rob_score, calib_score, profile="balanced")
        twi_sens = run_twi_sensitivity_analysis(perf_score, expl_score if expl_score is not None else 0.5, rob_score, calib_score)

        print("\n" + "=" * 60)
        print(f"TRUST-X FINAL TRUSTWORTHINESS EVALUATION: {exp_id}")
        print(f"  Pillar 1: Performance (Macro-F1) = {perf_score:.4f}")
        print(f"  Pillar 2: Explainability Score   = {expl_score:.4f}")
        print(f"  Pillar 3: Robustness Score       = {rob_score:.4f}")
        print(f"  Pillar 4: Calibration Score      = {calib_score:.4f}")
        print(f"  => FINAL TWI (Balanced Profile)  = {twi_default:.4f}")
        print(f"  => TWI Sensitivity Profiles:      {twi_sens}")
        print("=" * 60 + "\n")
    else:
        twi_default, twi_sens = None, {}
        print("[*] Skipping TWI calculation (Pilot mode)")

    # 12. Save Predictions & Probabilities
    pred_df = test_df.copy()
    pred_df['predicted_label'] = y_test_pred
    pred_df['probability_class_0'] = test_probs[:, 0]
    pred_df['probability_class_1'] = test_probs[:, 1]
    pred_df['model'] = model_key
    pred_df['language'] = language
    pred_file = os.path.join(output_dir, f"predictions_{exp_id}.csv")
    pred_df.to_csv(pred_file, index=False)
    print(f"[+] Saved test predictions to: {pred_file}")

    # 13. Save Comprehensive Experiment Results JSON
    exp_summary = {
        "experiment_id": exp_id,
        "language": language,
        "model": model_key,
        "checkpoint": checkpoint_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "train_duration_sec": train_duration,
        "best_val_f1": best_val_f1,
        "test_performance": test_metrics,
        "calibration": {
            "ece": ece,
            "brier": brier,
            "normalized_calibration_score": calib_score,
            "bins": bin_data
        },
        "robustness": robustness_results,
        "normalized_robustness_score": rob_score,
        "explainability": {
            "compactness_score": expl_score
        },
        "twi": {
            "balanced": twi_default,
            "sensitivity": twi_sens
        },
        "status": "COMPLETED"
    }

    summary_json_path = os.path.join(output_dir, f"results_{exp_id}.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=4)
    print(f"[+] Saved comprehensive results to: {summary_json_path}")

    # 14. Append to Global experiments.csv
    record_to_experiments_csv(exp_summary)

    return exp_summary

def record_to_experiments_csv(exp_summary):
    """
    Maintains centralized experiments tracking log.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exp_csv = os.path.join(base_dir, "experiments.csv")

    r = lambda x: round(x, 4) if x is not None else None
    row = {
        "experiment_id": exp_summary["experiment_id"],
        "language": exp_summary["language"],
        "model": exp_summary["model"],
        "checkpoint": exp_summary["checkpoint"],
        "learning_rate": exp_summary["learning_rate"],
        "batch_size": exp_summary["batch_size"],
        "epochs": exp_summary["epochs"],
        "max_length": exp_summary["max_length"],
        "seed": exp_summary["seed"],
        "best_val_f1": r(exp_summary.get("best_val_f1")),
        "test_accuracy": r(exp_summary.get("test_performance", {}).get("accuracy")),
        "test_macro_f1": r(exp_summary.get("test_performance", {}).get("f1_macro")),
        "ece": r(exp_summary.get("calibration", {}).get("ece")),
        "brier": r(exp_summary.get("calibration", {}).get("brier")),
        "robustness_score": r(exp_summary.get("normalized_robustness_score")),
        "explainability_score": r(exp_summary.get("explainability", {}).get("compactness_score")),
        "twi": r(exp_summary.get("twi", {}).get("balanced")),
        "status": exp_summary.get("status", "COMPLETED")
    }

    df_new = pd.DataFrame([row])
    if os.path.exists(exp_csv):
        df_old = pd.read_csv(exp_csv)
        df_updated = pd.concat([df_old[df_old["experiment_id"] != row["experiment_id"]], df_new], ignore_index=True)
    else:
        df_updated = df_new

    df_updated.to_csv(exp_csv, index=False)
    print(f"[+] Updated central experiments log: {exp_csv}")
