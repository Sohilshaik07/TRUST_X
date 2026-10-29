"""
TRUST-X Robustness Evaluation Module
Runs trained model against clean vs perturbed test sets and calculates:
1. Clean Accuracy & Clean Macro-F1
2. Perturbed Accuracy & Perturbed Macro-F1
3. Performance Drop (Delta F1 = Clean F1 - Perturbed F1)
4. Robustness Retention Rate (Perturbed F1 / Clean F1)
5. Prediction Flip Rate (percentage of samples where prediction changed)
6. Mean Confidence Drop
"""

import os
import numpy as np
import pandas as pd
import torch
from src.data.loaders import CodeMixedDataset
from src.evaluation.metrics import compute_all_metrics
from src.robustness.perturbations import (
    perturb_leet_substitution,
    perturb_char_deletion,
    perturb_char_repetition,
    perturb_punctuation_insertion,
    perturb_whitespace,
    apply_composite_perturbation
)

PERTURBATION_SUITE = {
    "leet_substitution": perturb_leet_substitution,
    "char_deletion": perturb_char_deletion,
    "char_repetition": perturb_char_repetition,
    "punctuation_insertion": perturb_punctuation_insertion,
    "whitespace": perturb_whitespace,
    "composite": apply_composite_perturbation
}

def evaluate_model_on_dataset(model, dataset, device, batch_size=32):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def run_robustness_evaluation(model, tokenizer, test_df, device, max_length=128, sample_size=None, seed=42):
    """
    Evaluates model across clean and perturbed test samples.
    """
    if sample_size and sample_size < len(test_df):
        eval_df = test_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        eval_df = test_df.reset_index(drop=True)

    # 1. Clean Evaluation
    clean_dataset = CodeMixedDataset(eval_df['text'], eval_df['label'], tokenizer, max_length)
    clean_labels, clean_preds, clean_probs = evaluate_model_on_dataset(model, clean_dataset, device)
    clean_metrics, _ = compute_all_metrics(clean_labels, clean_preds)

    clean_f1 = clean_metrics['f1_macro']
    clean_acc = clean_metrics['accuracy']
    clean_conf = np.max(clean_probs, axis=1)

    results = {
        "clean_accuracy": clean_acc,
        "clean_macro_f1": clean_f1,
        "perturbation_results": {}
    }

    print(f"[*] Robustness Baseline (Clean): Accuracy={clean_acc:.4f}, Macro-F1={clean_f1:.4f}")

    # 2. Evaluate across perturbation suite
    for p_name, p_func in PERTURBATION_SUITE.items():
        perturbed_texts = [p_func(str(t)) for t in eval_df['text']]
        perturbed_dataset = CodeMixedDataset(perturbed_texts, eval_df['label'], tokenizer, max_length)

        _, p_preds, p_probs = evaluate_model_on_dataset(model, perturbed_dataset, device)
        p_metrics, _ = compute_all_metrics(clean_labels, p_preds)

        p_f1 = p_metrics['f1_macro']
        p_acc = p_metrics['accuracy']
        p_conf = np.max(p_probs, axis=1)

        f1_drop = clean_f1 - p_f1
        retention = p_f1 / (clean_f1 + 1e-9)
        flip_rate = float(np.mean(clean_preds != p_preds))
        conf_drop = float(np.mean(clean_conf - p_conf))

        print(f"    -> [{p_name.upper()}]: F1={p_f1:.4f} (Drop={f1_drop:.4f}, Retention={retention:.2%}, FlipRate={flip_rate:.2%})")

        results["perturbation_results"][p_name] = {
            "perturbed_accuracy": float(p_acc),
            "perturbed_macro_f1": float(p_f1),
            "f1_drop": float(f1_drop),
            "f1_retention": float(retention),
            "flip_rate": float(flip_rate),
            "mean_confidence_drop": float(conf_drop)
        }

    # Aggregate composite score
    composite_res = results["perturbation_results"]["composite"]
    results["overall_robustness_score"] = float(composite_res["f1_retention"])
    results["composite_flip_rate"] = float(composite_res["flip_rate"])

    return results
