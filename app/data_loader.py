"""
TRUST-X Data Loader Module
Provides cached, read-only access to the frozen experimental CSV files.
Strictly adheres to read-only constraints: no modifications to frozen data.
"""

from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from app.config import (
    EXPERIMENTS_CSV,
    FINAL_RESULTS_CSV,
    MODEL_AVERAGES_CSV,
    LANGUAGE_AVERAGES_CSV,
    TWI_SENSITIVITY_CSV,
    TWI_SENSITIVITY_MODEL_CSV,
    TWI_SENSITIVITY_LANG_CSV,
)
from src.twi.twi import normalize_calibration_score


@st.cache_data(show_spinner=False)
def load_experiments_df() -> pd.DataFrame:
    """Load canonical experiments.csv (12 frozen experiments)."""
    if not EXPERIMENTS_CSV.exists():
        raise FileNotFoundError(f"Canonical experiments.csv not found at: {EXPERIMENTS_CSV}")
    return pd.read_csv(EXPERIMENTS_CSV)


@st.cache_data(show_spinner=False)
def load_final_results_df() -> pd.DataFrame:
    """Load tables/final_results.csv."""
    if not FINAL_RESULTS_CSV.exists():
        raise FileNotFoundError(f"final_results.csv not found at: {FINAL_RESULTS_CSV}")
    return pd.read_csv(FINAL_RESULTS_CSV)


@st.cache_data(show_spinner=False)
def load_model_averages_df() -> pd.DataFrame:
    """Load tables/model_averages.csv."""
    if not MODEL_AVERAGES_CSV.exists():
        raise FileNotFoundError(f"model_averages.csv not found at: {MODEL_AVERAGES_CSV}")
    return pd.read_csv(MODEL_AVERAGES_CSV)


@st.cache_data(show_spinner=False)
def load_language_averages_df() -> pd.DataFrame:
    """Load tables/language_averages.csv."""
    if not LANGUAGE_AVERAGES_CSV.exists():
        raise FileNotFoundError(f"language_averages.csv not found at: {LANGUAGE_AVERAGES_CSV}")
    return pd.read_csv(LANGUAGE_AVERAGES_CSV)


@st.cache_data(show_spinner=False)
def load_twi_sensitivity_df() -> pd.DataFrame:
    """Load tables/twi_sensitivity.csv."""
    if not TWI_SENSITIVITY_CSV.exists():
        raise FileNotFoundError(f"twi_sensitivity.csv not found at: {TWI_SENSITIVITY_CSV}")
    return pd.read_csv(TWI_SENSITIVITY_CSV)


@st.cache_data(show_spinner=False)
def load_twi_sensitivity_model_df() -> pd.DataFrame:
    """Load tables/twi_sensitivity_model_averages.csv."""
    if not TWI_SENSITIVITY_MODEL_CSV.exists():
        raise FileNotFoundError(f"twi_sensitivity_model_averages.csv not found at: {TWI_SENSITIVITY_MODEL_CSV}")
    return pd.read_csv(TWI_SENSITIVITY_MODEL_CSV)


@st.cache_data(show_spinner=False)
def load_twi_sensitivity_lang_df() -> pd.DataFrame:
    """Load tables/twi_sensitivity_language_averages.csv."""
    if not TWI_SENSITIVITY_LANG_CSV.exists():
        raise FileNotFoundError(f"twi_sensitivity_language_averages.csv not found at: {TWI_SENSITIVITY_LANG_CSV}")
    return pd.read_csv(TWI_SENSITIVITY_LANG_CSV)


def get_experiment_record(language: str, model: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve frozen experiment metrics for a specific language and model.
    Computes the composite calibration score from frozen ECE and Brier score.
    """
    df = load_experiments_df()
    match = df[(df["language"].str.lower() == language.lower()) & 
               (df["model"].str.lower() == model.lower())]
    if match.empty:
        return None
    
    record = match.iloc[0].to_dict()
    
    # Calculate composite calibration score from frozen ece and brier
    ece = float(record.get("ece", 0.0))
    brier = float(record.get("brier", 0.0))
    record["calibration_score"] = normalize_calibration_score(ece, brier)
    
    return record


def get_language_comparison_df(language: str) -> pd.DataFrame:
    """
    Get 3-model comparison subset for a specific language.
    """
    df = load_experiments_df()
    lang_df = df[df["language"].str.lower() == language.lower()].copy()
    
    # Add composite calibration score column
    lang_df["calibration_score"] = lang_df.apply(
        lambda r: normalize_calibration_score(float(r["ece"]), float(r["brier"])), axis=1
    )
    
    return lang_df
