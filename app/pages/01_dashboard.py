"""
TRUST-X Application: Page 01 - Dashboard
High-level project orientation, mathematical framework, 4-pillar overview, and frozen experiment leaderboard.
"""

import streamlit as st
import pandas as pd

from app.data_loader import load_model_averages_df, load_experiments_df
from app.config import PILLARS, LANGUAGES, MODELS, LANGUAGE_DISPLAY, MODEL_DISPLAY


def render():
    st.title("🛡️ TRUST-X: Trustworthy NLP for Code-Mixed Indian Languages")
    st.caption("Benchmarking Performance, Explainability, Robustness, and Calibration across 12 Empirical Experiments")

    st.markdown(
        """
        **TRUST-X** is a comprehensive evaluation framework designed to assess the **multi-dimensional trustworthiness** 
        of multilingual transformers (mBERT, IndicBERT, XLM-RoBERTa) on code-mixed Dravidian and Indo-Aryan social media text.
        """
    )

    # Key Scope Banner
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", "12", "Frozen & Canonical")
    c2.metric("Languages", "4", "Hindi, Telugu, Tamil, Malayalam")
    c3.metric("Architectures", "3", "mBERT, IndicBERT, XLM-R")
    c4.metric("Trust Pillars", "4", "25% Weight Each")

    st.divider()

    # Section 1: The 4 TRUST-X Pillars & TWI Formulation
    st.subheader("🏛️ The Four Pillars of Trustworthiness")
    st.markdown(
        """
        Standard NLP benchmarks evaluate models solely on predictive accuracy or Macro-F1. 
        In high-stakes moderation of code-mixed languages, a trustworthy model must additionally be:
        well-calibrated, robust to adversarial spelling perturbations, and faithful in its explanations.
        """
    )

    cols = st.columns(4)
    for col, (name, meta) in zip(cols, PILLARS.items()):
        with col:
            st.markdown(
                f"""
                <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 14px; background: #fafafa; min-height: 180px;">
                    <h5 style="margin-top: 0; color: #0d6efd;">{name}</h5>
                    <p style="font-size: 0.88em; color: #495057;"><strong>Metric:</strong> {meta['metric']}</p>
                    <p style="font-size: 0.82em; color: #6c757d;">{meta['description']}</p>
                    <div style="font-size: 0.85em; font-weight: bold; color: #198754;">Default Weight: 25%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(" ")
    with st.expander("📐 Trustworthiness Index (TWI) Mathematical Formulation", expanded=False):
        st.latex(
            r"\text{TWI} = w_{\text{perf}} \cdot \mathcal{S}_{\text{perf}} + "
            r"w_{\text{expl}} \cdot \mathcal{S}_{\text{expl}} + "
            r"w_{\text{rob}} \cdot \mathcal{S}_{\text{rob}} + "
            r"w_{\text{calib}} \cdot \mathcal{S}_{\text{calib}}"
        )
        st.markdown(
            r"""
            Where:
            - $\mathcal{S}_{\text{perf}} = \text{Macro-F1}$ (class-balanced predictive score)
            - $\mathcal{S}_{\text{expl}} = \text{Compactness}$ (SHAP attribution mass concentrated on top salient tokens)
            - $\mathcal{S}_{\text{rob}} = 0.6 \cdot \text{F1-Retention} + 0.4 \cdot (1 - \text{Flip-Rate})$ (noise robustness)
            - $\mathcal{S}_{\text{calib}} = 0.5 \cdot (1 - \text{ECE}) + 0.5 \cdot \max(0, 1 - 2 \cdot \text{Brier})$ (probabilistic calibration)
            - Default Balanced Profile: $w_{\text{perf}} = w_{\text{expl}} = w_{\text{rob}} = w_{\text{calib}} = 0.25$
            """
        )

    st.divider()

    # Section 2: Overall Model Summary from model_averages.csv
    st.subheader("📊 Cross-Language Model Averages")
    st.markdown("Aggregated frozen results across all 4 Indian languages (from `tables/model_averages.csv`):")

    try:
        model_avg_df = load_model_averages_df()
        # Format display
        display_df = model_avg_df.copy()
        for col in ["test_accuracy", "test_macro_f1", "ece", "brier", "robustness_score", "explainability_score", "twi"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.4f}")
        
        display_df.rename(columns={
            "model": "Model",
            "test_accuracy": "Accuracy",
            "test_macro_f1": "Macro-F1",
            "ece": "ECE (↓)",
            "brier": "Brier (↓)",
            "robustness_score": "Robustness (↑)",
            "explainability_score": "Explainability (↑)",
            "twi": "Balanced TWI (↑)",
        }, inplace=True)

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading model averages: {e}")

    st.divider()

    # Section 3: Frozen Experiments Leaderboard
    st.subheader("🏆 Canonical Experiments Leaderboard")
    st.markdown("All 12 experimental runs sorted by overall Trustworthiness Index (TWI):")

    try:
        exp_df = load_experiments_df().copy()
        # Sort by twi descending
        exp_df = exp_df.sort_values(by="twi", ascending=False)
        
        # Format table for presentation
        fmt_cols = ["test_accuracy", "test_macro_f1", "ece", "brier", "robustness_score", "explainability_score", "twi"]
        for c in fmt_cols:
            if c in exp_df.columns:
                exp_df[c] = exp_df[c].apply(lambda x: f"{float(x):.4f}")
                
        leaderboard_cols = ["experiment_id", "language", "model", "test_macro_f1", "twi", "test_accuracy", "ece", "robustness_score", "explainability_score"]
        lb_df = exp_df[[c for c in leaderboard_cols if c in exp_df.columns]].copy()
        lb_df.rename(columns={
            "experiment_id": "Experiment ID",
            "language": "Language",
            "model": "Model",
            "test_macro_f1": "Macro-F1",
            "twi": "Balanced TWI",
            "test_accuracy": "Accuracy",
            "ece": "ECE",
            "robustness_score": "Robustness",
            "explainability_score": "Explainability",
        }, inplace=True)
        
        st.dataframe(lb_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading experiments leaderboard: {e}")

    st.divider()

    # Section 4: Scientific Integrity Disclaimer
    st.info(
        """
        ℹ️ **Scientific Integrity Distinction:**
        - **Dataset-Level Evaluation (Pages C, D, E):** All reported metrics (Macro-F1, ECE, Brier, Robustness, TWI) 
          are fixed, canonical values derived from frozen test splits across 12 empirical experiments.
        - **Live Sample Prediction (Page B):** Single-sample inference provides only class probability 
          and token-level attribution. Dataset-level metrics are never fabricated or estimated from individual inputs.
        """
    )


if __name__ == "__main__":
    render()
