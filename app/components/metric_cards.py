"""
TRUST-X Metric Cards Component
Renders clean, publication-ready metric cards for experimental evaluation.
"""

from typing import Dict, Any, Optional
import streamlit as st


def render_experiment_metric_cards(record: Dict[str, Any], title: Optional[str] = None):
    """
    Renders 2 rows of metric cards for a given frozen experiment record.
    Row 1: Primary Performance & Calibration (Accuracy, Macro-F1, ECE, Brier Score)
    Row 2: Trustworthiness Pillars & Composite TWI (Robustness, Explainability, Calibration Score, Balanced TWI)
    """
    if title:
        st.subheader(title)
        
    acc = float(record.get("test_accuracy", 0.0))
    f1 = float(record.get("test_macro_f1", 0.0))
    ece = float(record.get("ece", 0.0))
    brier = float(record.get("brier", 0.0))
    rob = float(record.get("robustness_score", 0.0))
    expl = float(record.get("explainability_score", 0.0))
    calib = float(record.get("calibration_score", 0.0))
    twi = float(record.get("twi", 0.0))

    # Row 1: Traditional Metrics
    st.markdown("##### 📌 Standard Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Test Accuracy",
            value=f"{acc:.4f}",
            help="Overall classification accuracy on the test set. Higher is better."
        )
    with c2:
        st.metric(
            label="Test Macro-F1",
            value=f"{f1:.4f}",
            help="Unweighted mean of class F1 scores. Crucial for imbalanced hate speech data. Higher is better."
        )
    with c3:
        st.metric(
            label="Expected Calib. Error (ECE)",
            value=f"{ece:.4f}",
            help="Difference between predicted probability confidence and actual accuracy. Lower is better."
        )
    with c4:
        st.metric(
            label="Brier Score",
            value=f"{brier:.4f}",
            help="Mean squared error of probabilistic predictions. Lower is better."
        )

    # Row 2: TRUST-X 4 Pillars & TWI
    st.markdown("##### 🛡️ TRUST-X Pillars & Trustworthiness Index (TWI)")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(
            label="Robustness Score",
            value=f"{rob:.4f}",
            help="Normalized score measuring stability against character/word noise perturbations. Higher is better."
        )
    with c6:
        st.metric(
            label="Explainability Score",
            value=f"{expl:.4f}",
            help="Normalized SHAP attribution compactness on salient keywords. Higher is better."
        )
    with c7:
        st.metric(
            label="Calibration Score",
            value=f"{calib:.4f}",
            help="Composite calibration score normalized to [0, 1] from ECE and Brier. Higher is better."
        )
    with c8:
        st.metric(
            label="Balanced TWI",
            value=f"{twi:.4f}",
            help="Trustworthiness Index with equal 0.25 weights across all 4 pillars. Higher is better."
        )
