"""
TRUST-X Application: Page 04 - Model Comparison
Cross-model comparative analysis per language and embedded publication-ready figures.
"""

import streamlit as st
import pandas as pd

from app.config import (
    LANGUAGES,
    LANGUAGE_DISPLAY,
    LANGUAGE_REVERSE,
    FIG_MACRO_F1,
    FIG_TWI,
    FIG_PILLARS,
    FIG_ACC_VS_F1,
    FIG_F1_VS_TWI,
)
from app.data_loader import get_language_comparison_df, load_model_averages_df


def render():
    st.title("⚖️ Model Comparison")
    st.caption("Side-by-side comparative analysis of mBERT, IndicBERT, and XLM-RoBERTa across all metrics.")

    # Language Selector
    selected_lang_display = st.selectbox(
        "Select Language for Direct Comparison:",
        options=[LANGUAGE_DISPLAY[l] for l in LANGUAGES],
        index=0,
    )
    selected_lang = LANGUAGE_REVERSE[selected_lang_display]

    # Comparative Table
    st.subheader(f"📊 3-Model Comparison: {selected_lang_display}")
    
    comp_df = get_language_comparison_df(selected_lang)
    if not comp_df.empty:
        # Format table
        display_df = comp_df.copy()
        for col in ["test_accuracy", "test_macro_f1", "ece", "brier", "robustness_score", "explainability_score", "calibration_score", "twi"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.4f}")

        display_df = display_df[[
            "model", "test_accuracy", "test_macro_f1", "ece", "brier", 
            "robustness_score", "explainability_score", "calibration_score", "twi"
        ]].copy()
        
        display_df.rename(columns={
            "model": "Model",
            "test_accuracy": "Accuracy (↑)",
            "test_macro_f1": "Macro-F1 (↑)",
            "ece": "ECE (↓)",
            "brier": "Brier Score (↓)",
            "robustness_score": "Robustness (↑)",
            "explainability_score": "Explainability (↑)",
            "calibration_score": "Calibration (↑)",
            "twi": "Balanced TWI (↑)",
        }, inplace=True)

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"No comparative data available for {selected_lang_display}.")

    st.divider()

    # Publication Figures Section
    st.subheader("🖼️ Publication-Quality Comparative Figures")
    st.markdown("Inspect the canonical figures generated from the frozen experimental benchmark:")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Macro-F1 Comparison",
        "TWI Comparison",
        "Trust Pillars",
        "Accuracy vs Macro-F1",
        "Macro-F1 vs TWI",
    ])

    with tab1:
        if FIG_MACRO_F1.exists():
            st.image(str(FIG_MACRO_F1), caption="Figure 1: Test Macro-F1 comparison across 4 languages and 3 models.", use_container_width=True)
        else:
            st.error(f"Figure not found: {FIG_MACRO_F1}")

    with tab2:
        if FIG_TWI.exists():
            st.image(str(FIG_TWI), caption="Figure 2: Trustworthiness Index (Balanced TWI) across 4 languages and 3 models.", use_container_width=True)
        else:
            st.error(f"Figure not found: {FIG_TWI}")

    with tab3:
        if FIG_PILLARS.exists():
            st.image(str(FIG_PILLARS), caption="Figure 3: Multi-dimensional breakdown of all 4 Trust Pillars per language and model.", use_container_width=True)
        else:
            st.error(f"Figure not found: {FIG_PILLARS}")

    with tab4:
        if FIG_ACC_VS_F1.exists():
            st.image(str(FIG_ACC_VS_F1), caption="Figure 4: Test Accuracy vs Test Macro-F1 with clean, non-overlapping annotations.", use_container_width=True)
        else:
            st.error(f"Figure not found: {FIG_ACC_VS_F1}")

    with tab5:
        if FIG_F1_VS_TWI.exists():
            st.image(str(FIG_F1_VS_TWI), caption="Figure 5: Macro-F1 vs TWI scatter showing the divergence between raw performance and holistic trust.", use_container_width=True)
        else:
            st.error(f"Figure not found: {FIG_F1_VS_TWI}")

    st.divider()

    # Cross-Language Summary
    st.subheader("🌐 Global Model Averages Across All 4 Languages")
    try:
        avg_df = load_model_averages_df().copy()
        for col in ["test_accuracy", "test_macro_f1", "ece", "brier", "robustness_score", "explainability_score", "twi"]:
            if col in avg_df.columns:
                avg_df[col] = avg_df[col].apply(lambda x: f"{float(x):.4f}")
        
        avg_df.rename(columns={
            "model": "Model Architecture",
            "test_accuracy": "Mean Accuracy",
            "test_macro_f1": "Mean Macro-F1",
            "ece": "Mean ECE",
            "brier": "Mean Brier",
            "robustness_score": "Mean Robustness",
            "explainability_score": "Mean Explainability",
            "twi": "Mean Balanced TWI",
        }, inplace=True)
        
        st.dataframe(avg_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading model averages: {e}")


if __name__ == "__main__":
    render()
