"""
TRUST-X Main Application Entry Point
Multi-page Streamlit application for benchmarking and auditing trustworthy NLP 
in Dravidian and Indo-Aryan code-mixed social media text.
"""

from pathlib import Path
import sys
import streamlit as st

# Ensure project root is in sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Streamlit Page Setup
st.set_page_config(
    page_title="TRUST-X: Trustworthy Code-Mixed NLP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define Navigation Structure using Streamlit 1.56 st.navigation
pages = {
    "Benchmark Overview": [
        st.Page("pages/01_dashboard.py", title="Dashboard", icon="📊", default=True),
    ],
    "Model Auditing": [
        st.Page("pages/03_model_evaluation.py", title="Model Evaluation", icon="📈"),
        st.Page("pages/04_model_comparison.py", title="Model Comparison", icon="⚖️"),
        st.Page("pages/05_twi_sensitivity.py", title="TWI Sensitivity", icon="🔬"),
    ],
    "Interactive Inference": [
        st.Page("pages/02_live_detection.py", title="Live Detection", icon="🔍"),
    ],
}

# Sidebar Branding & Metadata
with st.sidebar:
    st.markdown("### 🛡️ TRUST-X")
    st.markdown(
        "<small style='color: #6c757d;'>Benchmarking Multi-Dimensional Trust in Indian Code-Mixed NLP</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    st.markdown(
        """
        <div style='font-size: 0.85em; color: #495057;'>
            <strong>Pillars:</strong><br>
            • Performance (Macro-F1)<br>
            • Explainability (Compactness)<br>
            • Robustness (Noise Resilience)<br>
            • Calibration (ECE & Brier)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Version 1.0 (Phase 1 Frozen Benchmark)")

# Run Selected Page
pg = st.navigation(pages)
pg.run()
