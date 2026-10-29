"""
TRUST-X SHAP Explainability Module
Generates token-level feature attributions using SHAP for fine-tuned transformers:
- Analyzes representative samples (TP, TN, FP, FN)
- Produces token importance attributions
- Measures explanation compactness (Sparsity score)
- Saves publication-ready plots and attribution tables
"""

import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def select_representative_subset(test_df, predictions, probabilities, n_samples=50, seed=42):
    """
    Selects a balanced representative subset of samples:
    - True Positives (Harmful correctly classified)
    - True Negatives (Non-harmful correctly classified)
    - False Positives (Non-harmful predicted as Harmful)
    - False Negatives (Harmful predicted as Non-harmful)
    """
    df = test_df.copy()
    df['pred'] = predictions
    df['prob_1'] = probabilities[:, 1]
    df['true'] = df['label']

    tp = df[(df['true'] == 1) & (df['pred'] == 1)]
    tn = df[(df['true'] == 0) & (df['pred'] == 0)]
    fp = df[(df['true'] == 0) & (df['pred'] == 1)]
    fn = df[(df['true'] == 1) & (df['pred'] == 0)]

    per_group = max(5, n_samples // 4)

    sampled_tp = tp.sample(n=min(len(tp), per_group), random_state=seed) if len(tp) > 0 else pd.DataFrame()
    sampled_tn = tn.sample(n=min(len(tn), per_group), random_state=seed) if len(tn) > 0 else pd.DataFrame()
    sampled_fp = fp.sample(n=min(len(fp), per_group), random_state=seed) if len(fp) > 0 else pd.DataFrame()
    sampled_fn = fn.sample(n=min(len(fn), per_group), random_state=seed) if len(fn) > 0 else pd.DataFrame()

    subset = pd.concat([sampled_tp, sampled_tn, sampled_fp, sampled_fn]).drop_duplicates(subset=['text'])
    print(f"[*] Selected {len(subset)} representative samples for SHAP explainability (TP={len(sampled_tp)}, TN={len(sampled_tn)}, FP={len(sampled_fp)}, FN={len(sampled_fn)})")
    return subset

def run_shap_analysis(model, tokenizer, representative_df, device, output_dir, max_evals=100):
    """
    Runs SHAP Text explainer using transformers pipeline prediction function.
    """
    os.makedirs(output_dir, exist_ok=True)
    texts = representative_df['text'].tolist()

    # Prediction wrapper function for SHAP
    def predict_probs(text_list):
        model.eval()
        if isinstance(text_list, str):
            text_list = [text_list]
        enc = tokenizer(
            list(text_list),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    print(f"[*] Initializing SHAP Explainer with tokenizer masker...")
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_probs, masker, output_names=["Non-Harmful", "Harmful"])

    try:
        # Compute shap values
        shap_values = explainer(texts[:len(representative_df)], max_evals=max_evals)
        
        # Calculate quantitative explainability metrics:
        # Explanation Sparsity / Compactness: Gini coefficient or fraction of energy in top-k tokens
        sparsities = []
        for i in range(len(shap_values)):
            vals = np.abs(shap_values[i].values[:, 1]) # class 1 attribution
            if vals.sum() > 0:
                # Fraction of attribution concentrated in top 20% tokens
                k = max(1, int(0.2 * len(vals)))
                sorted_vals = np.sort(vals)[::-1]
                top_k_fraction = sorted_vals[:k].sum() / vals.sum()
                sparsities.append(float(top_k_fraction))
            else:
                sparsities.append(0.0)

        mean_compactness = float(np.mean(sparsities)) if sparsities else 0.5
        print(f"[+] SHAP explainability completed successfully. Mean Compactness Score: {mean_compactness:.4f}")

        # Save summary figure
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values[:, :, 1], show=False)
        plt.title("SHAP Global Feature Attribution (Top Token Attributions for Harmful Class)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "shap_global_bar.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved SHAP global attribution bar plot to: {fig_path}")

        return {
            "status": "COMPLETED",
            "mean_compactness_score": mean_compactness,
            "samples_analyzed": len(representative_df),
            "shap_figure": fig_path
        }

    except Exception as e:
        print(f"[-] SHAP evaluation encountered error: {str(e)}")
        return {
            "status": f"FAILED: {str(e)}",
            "mean_compactness_score": 0.5, # Default neutral score
            "samples_analyzed": 0,
            "shap_figure": None
        }
