"""
TRUST-X Trustworthiness Index (TWI) Module
Formulates and computes the multi-dimensional Trustworthiness Index:
    TWI = w1 * PerformanceScore + w2 * ExplainabilityScore + w3 * RobustnessScore + w4 * CalibrationScore
where w1 + w2 + w3 + w4 = 1.0

Implements:
- Scientific normalization (higher = better for all components)
- Weight profile configurations (Default, Performance-Heavy, Robustness-Heavy, Calibration-Heavy)
- Sensitivity analysis across weight profiles
"""

import numpy as np
import pandas as pd

WEIGHT_PROFILES = {
    "balanced": {
        "description": "Equal balance across all 4 trustworthiness pillars",
        "weights": {"w_perf": 0.25, "w_expl": 0.25, "w_rob": 0.25, "w_calib": 0.25}
    },
    "performance_heavy": {
        "description": "Emphasizes predictive performance while auditing trustworthiness",
        "weights": {"w_perf": 0.50, "w_expl": 0.15, "w_rob": 0.20, "w_calib": 0.15}
    },
    "robustness_heavy": {
        "description": "Emphasizes resilience to code-mixed adversarial noise & typos",
        "weights": {"w_perf": 0.20, "w_expl": 0.15, "w_rob": 0.45, "w_calib": 0.20}
    },
    "calibration_heavy": {
        "description": "Emphasizes risk-sensitive, well-calibrated confidence in moderation",
        "weights": {"w_perf": 0.20, "w_expl": 0.15, "w_rob": 0.20, "w_calib": 0.45}
    }
}

def normalize_calibration_score(ece, brier):
    """
    Normalizes ECE and Brier score into a Calibration Score in [0, 1] where higher = better.
    ECE is in [0, 1], lower is better. Calib_ECE = 1.0 - ECE.
    Brier score for binary is in [0, 1] (0.25 is random guessing). Calib_Brier = 1.0 - (2.0 * Brier).
    Composite Calibration Score = 0.5 * Calib_ECE + 0.5 * max(0.0, Calib_Brier).
    """
    calib_ece = max(0.0, 1.0 - ece)
    calib_brier = max(0.0, 1.0 - (2.0 * brier))
    return float(0.5 * calib_ece + 0.5 * calib_brier)

def normalize_robustness_score(f1_retention, flip_rate):
    """
    Robustness Score in [0, 1] where higher = better:
    Combines F1 retention (how much F1 is preserved under perturbation)
    and (1 - flip_rate) (fraction of predictions that stayed consistent).
    """
    retention_score = min(1.0, max(0.0, f1_retention))
    stability_score = max(0.0, 1.0 - flip_rate)
    return float(0.6 * retention_score + 0.4 * stability_score)

def calculate_twi(performance_f1, explainability_score, robustness_score, calibration_score, profile="balanced"):
    """
    Calculates TWI for a given weight profile.
    All component scores must be in [0, 1].
    """
    weights = WEIGHT_PROFILES[profile]["weights"]
    twi = (
        weights["w_perf"] * performance_f1 +
        weights["w_expl"] * explainability_score +
        weights["w_rob"] * robustness_score +
        weights["w_calib"] * calibration_score
    )
    return float(twi)

def run_twi_sensitivity_analysis(performance_f1, explainability_score, robustness_score, calibration_score):
    """
    Computes TWI under all predefined weight profiles for scientific sensitivity reporting.
    """
    results = {}
    for profile, cfg in WEIGHT_PROFILES.items():
        twi_val = calculate_twi(
            performance_f1,
            explainability_score,
            robustness_score,
            calibration_score,
            profile=profile
        )
        results[profile] = round(twi_val, 4)
    return results
