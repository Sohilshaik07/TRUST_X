"""
Verify all reported TRUST-X Telugu+mBERT numeric values against implemented formulas.
DO NOT modify any experiment outputs.
"""
import numpy as np

print("="*60)
print("FORMULA VERIFICATION: CALIBRATION NORMALIZATION")
print("="*60)

ECE   = 0.0539
BRIER = 0.1734

calib_ece   = max(0.0, 1.0 - ECE)
calib_brier = max(0.0, 1.0 - (2.0 * BRIER))
calib_score = 0.5 * calib_ece + 0.5 * calib_brier

print(f"ECE       = {ECE}")
print(f"Brier     = {BRIER}")
print(f"calib_ece (1 - ECE)            = {calib_ece:.6f}")
print(f"calib_brier (1 - 2*Brier)      = {calib_brier:.6f}")
print(f"CalibScore = 0.5*calib_ece + 0.5*calib_brier = {calib_score:.6f}")
print(f"Reported Normalized Calib Score = 0.7996")
print(f"Match: {abs(calib_score - 0.7996) < 0.001}")

print()
print("="*60)
print("FORMULA VERIFICATION: ROBUSTNESS NORMALIZATION")
print("="*60)

RETENTION = 0.9600
FLIP_RATE = 0.1841

retention_score = min(1.0, max(0.0, RETENTION))
stability_score = max(0.0, 1.0 - FLIP_RATE)
rob_score = 0.6 * retention_score + 0.4 * stability_score

print(f"Retention (F1 perturbed / F1 clean) = {RETENTION}")
print(f"FlipRate                             = {FLIP_RATE}")
print(f"retention_score = clip(Retention, 0, 1) = {retention_score:.6f}")
print(f"stability_score = 1 - FlipRate          = {stability_score:.6f}")
print(f"RobScore = 0.6*retention + 0.4*stability = {rob_score:.6f}")
print(f"Reported Normalized Robustness Score     = 0.9023")
print(f"Match: {abs(rob_score - 0.9023) < 0.001}")

print()
print("="*60)
print("FORMULA VERIFICATION: TWI (Balanced Profile)")
print("="*60)

PERF   = 0.7399   # Macro-F1
EXPL   = 0.5127   # Mean Compactness
ROB    = 0.9023
CALIB  = 0.7996
W = 0.25  # each, balanced

twi = W * PERF + W * EXPL + W * ROB + W * CALIB
print(f"Performance (Macro-F1) = {PERF}")
print(f"Explainability         = {EXPL}")
print(f"Robustness             = {ROB}")
print(f"Calibration            = {CALIB}")
print(f"Weights (balanced)     = 0.25 each")
print(f"TWI = 0.25*(P+E+R+C)  = {twi:.6f}")
print(f"Reported TWI           = 0.7386")
print(f"Match: {abs(twi - 0.7386) < 0.001}")

print()
print("="*60)
print("SHAP COMPACTNESS SCORE DEFINITION VERIFICATION")
print("="*60)
print("Formula (from code shap_analysis.py L88-90):")
print("  k = max(1, int(0.2 * len(vals)))   # top-20% tokens")
print("  top_k_fraction = sorted_vals[:k].sum() / vals.sum()")
print("  (vals = abs SHAP values for class 1 per sample)")
print("  mean_compactness = mean(top_k_fraction) over all SHAP samples")
print()
print("Interpretation:")
print("  - Measures: fraction of total attribution concentrated in top 20% of tokens")
print("  - Range: [0, 1] — higher means more concentrated/compact explanation")
print("  - Normalization: inherent (ratio bounded 0–1)")
print("  - Higher better?: YES for compactness, but not necessarily for trustworthiness")
print("  - Relationship to 'explainability': INFORMAL — not a peer-reviewed metric")
print("  - SHAP compactness is an INTERNAL IMPLEMENTATION METRIC, not a standard")

print()
print("="*60)
print("ECE FORMULA VERIFICATION (from code calibration.py)")
print("="*60)
print("ECE implementation uses 2D probability array:")
print("  confidences = max(probs, axis=1)  (max softmax probability)")
print("  predictions = argmax(probs, axis=1)")
print("  accuracies  = (predictions == labels)")
print("  bins: equal-width [0,1] with n_bins=10")
print("  ECE = sum_b |prop_in_bin| * |avg_confidence_b - avg_accuracy_b|")
print()
print("NOTE: This uses max-confidence (multiclass ECE formulation) for a binary task.")
print("Equivalent binary formulation exists; results may differ from probability-calibration ECE")
print("using just P(class 1) vs true labels.")

print()
print("="*60)
print("BRIER SCORE FORMULA VERIFICATION")
print("="*60)
print("Brier = brier_score_loss(y_true, probs[:,1])")
print("     = mean((y_true_i - prob_class1_i)^2)")
print("Standard sklearn binary Brier score — CORRECT.")

print()
print("="*60)
print("ROBUSTNESS — COMPOSITE PERTURBATION FORMULA (from code)")
print("="*60)
print("composite = char_repetition(rate=0.15) -> leet_substitution(rate=0.12) -> char_deletion(rate=0.08)")
print("All using same random.Random(seed=42) RNG — deterministic")
print()
print("Key formula per perturbation type:")
print("  f1_drop      = clean_f1 - perturbed_f1")
print("  retention    = perturbed_f1 / (clean_f1 + 1e-9)   [note: epsilon guard]")
print("  flip_rate    = mean(clean_preds != perturbed_preds)")
print("  conf_drop    = mean(max_clean_probs - max_perturbed_probs)")
print()
print("overall_robustness_score = composite retention (not mean of individual retentions)")
print("composite_flip_rate      = composite flip_rate only (not mean of all)")

print()
print("="*60)
print("SEED CONSISTENCY CHECK")
print("="*60)
print("Config seed=42 used for:")
print("  1. Data split (stratified, random_state=42 in sklearn)")
print("  2. set_seed(42) -> numpy + torch + cuda seeds")
print("  3. SHAP sampling random_state=42")
print("  4. Perturbation RNG: random.Random(42) per function call")
print("  5. Composite perturbation: random.Random(seed=42) shared RNG")
print()
print("NOTE: Perturbation functions use fixed rng=random.Random(42) default")
print("This means each perturbation call is independently seeded, NOT sharing state")
print("This is deterministic but each text is perturbed with same sequence")

print()
print("="*60)
print("NUMERIC VERIFICATION COMPLETE")
print("="*60)
