"""
TRUST-X Application: Page 02 - Live Detection & Comparative Explainability
3-Model parallel inference interface: mBERT | IndicBERT | XLM-R
Displays predictions, probabilities, model agreement, and side-by-side SHAP attributions.

Scientific Boundaries:
- Only classification, softmax probabilities, and SHAP token attributions are produced.
- Dataset-level metrics (Macro-F1, ECE, TWI) are never calculated for live input.
- Frozen experiment results, checkpoints, and CSVs are never modified.
"""

import streamlit as st
import torch
import numpy as np
import shap
import html as html_lib

from app.config import (
    LANGUAGES,
    MODELS,
    LANGUAGE_DISPLAY,
    MODEL_SHORT_DISPLAY,
    LANGUAGE_REVERSE,
    SAMPLE_TEXTS,
    HF_FALLBACK_MODELS,
    get_label_semantics,
    TASK_LABEL_MAPPINGS,
)
from app.model_loader import (
    check_checkpoint_exists,
    load_model_and_tokenizer,
)
from src.data.preprocessing import clean_text
from app.components.shap_display import render_token_attributions


# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_KEYS = ["mbert", "indicbert", "xlmr"]
_MODEL_LABELS = {
    "mbert":     "mBERT",
    "indicbert": "IndicBERT",
    "xlmr":      "XLM-R",
}
_SHAP_MAX_EVALS = 30   # Matches existing project SHAP budget


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_inference(model, tokenizer, device, cleaned_text: str):
    """Run tokenization + forward pass; return (prob_0, prob_1, pred_class, confidence)."""
    enc = tokenizer(
        cleaned_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    prob_0     = float(probs[0])
    prob_1     = float(probs[1])
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return prob_0, prob_1, pred_class, confidence


def _run_shap(model, tokenizer, device, cleaned_text: str, target_class: int):
    """
    Compute SHAP token attributions for `target_class`.
    Returns (tokens: list[str], attributions: list[float]) or raises.
    """
    def _predict_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            return torch.softmax(out.logits, dim=-1).cpu().numpy()

    masker    = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(_predict_fn, masker)
    shap_res  = explainer([cleaned_text], max_evals=_SHAP_MAX_EVALS)
    tokens       = list(shap_res[0].data)
    attributions = shap_res[0].values[:, target_class].tolist()
    return tokens, attributions


def _checkpoint_badge(lang: str, model_key: str) -> str:
    """Return a compact HTML badge for checkpoint status."""
    has_ckpt, _, safetensors_path = check_checkpoint_exists(lang, model_key)
    if has_ckpt:
        return (
            '<span style="background:#d1e7dd;color:#0f5132;padding:2px 8px;'
            'border-radius:12px;font-size:0.78em;font-weight:bold;">&#x1F6E1;&#xFE0F; Fine-tuned</span>'
        )
    return (
        f'<span style="background:#fff3cd;color:#664d03;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:bold;" '
        f'title="Missing: {safetensors_path}">&#x26A0;&#xFE0F; Base fallback</span>'
    )


def _prediction_card(
    model_label: str,
    pred_class: int,
    prob_0: float,
    prob_1: float,
    confidence: float,
    sem_0: dict,
    sem_1: dict,
    pred_sem: dict,
    checkpoint_badge_html: str,
):
    """Render a single model prediction card inside the caller's st.column context."""
    is_harmful_pred = pred_sem.get("is_harmful", False)
    border_color    = "#dc3545" if is_harmful_pred else "#198754"
    badge_bg        = pred_sem.get("badge_bg", "#e2e3e5")
    badge_color     = pred_sem.get("badge_color", "#41464b")

    st.markdown(
        f"""
        <div style="border:2px solid {border_color};border-radius:10px;
                    padding:16px;margin-bottom:8px;background:#fafafa;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-size:1.05em;font-weight:700;color:#212529;">{model_label}</span>
                {checkpoint_badge_html}
            </div>
            <div style="background:{badge_bg};color:{badge_color};border-radius:6px;
                        padding:10px;text-align:center;margin-bottom:12px;">
                <div style="font-size:0.75em;text-transform:uppercase;letter-spacing:1px;
                            font-weight:600;opacity:0.8;">Prediction</div>
                <div style="font-size:1.45em;font-weight:800;">{pred_sem['name']}</div>
                <div style="font-size:0.9em;margin-top:2px;">Confidence: <strong>{confidence*100:.1f}%</strong></div>
            </div>
            <table style="width:100%;font-size:0.88em;border-collapse:collapse;">
                <tr>
                    <td style="padding:4px 6px;color:#495057;">
                        Class&nbsp;0&nbsp;&mdash;&nbsp;{html_lib.escape(sem_0['name'])}
                    </td>
                    <td style="padding:4px 6px;text-align:right;font-weight:600;
                               color:{'#842029' if sem_0.get('is_harmful') else '#0f5132'};">
                        {prob_0*100:.1f}%
                    </td>
                </tr>
                <tr style="background:#f8f9fa;">
                    <td style="padding:4px 6px;color:#495057;">
                        Class&nbsp;1&nbsp;&mdash;&nbsp;{html_lib.escape(sem_1['name'])}
                    </td>
                    <td style="padding:4px 6px;text-align:right;font-weight:600;
                               color:{'#842029' if sem_1.get('is_harmful') else '#0f5132'};">
                        {prob_1*100:.1f}%
                    </td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _agreement_section(results: dict, selected_lang: str):
    """
    Render the model agreement / disagreement section.
    `results` is keyed by model_key; each value has 'pred_class'.
    """
    preds         = {mk: results[mk]["pred_class"] for mk in _MODEL_KEYS if mk in results}
    harmful_class = 0 if selected_lang == "hindi" else 1

    harmful_count  = sum(1 for p in preds.values() if p == harmful_class)
    harmless_count = len(preds) - harmful_count

    sem_harmful  = get_label_semantics(selected_lang, harmful_class)
    sem_harmless = get_label_semantics(selected_lang, 1 - harmful_class)

    all_agree     = harmful_count in (0, len(preds))
    majority_harm = harmful_count > len(preds) / 2

    if all_agree:
        label = sem_harmful["name"] if harmful_count > 0 else sem_harmless["name"]
        color = "#0f5132" if harmful_count == 0 else "#842029"
        bg    = "#d1e7dd" if harmful_count == 0 else "#f8d7da"
        icon  = "&#x2705;"
        headline = f"{len(preds)} of {len(preds)} models agree: <strong>{html_lib.escape(label)}</strong>"
    else:
        dominant_label = sem_harmful["name"] if majority_harm else sem_harmless["name"]
        count_d  = harmful_count if majority_harm else harmless_count
        color    = "#664d03"
        bg       = "#fff3cd"
        icon     = "&#x26A0;&#xFE0F;"
        headline = f"{count_d} of {len(preds)} models predict <strong>{html_lib.escape(dominant_label)}</strong>"

    st.markdown(
        f"""
        <div style="background:{bg};border-left:4px solid {color};
                    border-radius:8px;padding:14px 18px;margin:12px 0;">
            <span style="font-size:1.1em;font-weight:700;color:{color};">{icon} Model Agreement</span><br>
            <span style="color:{color};font-size:0.95em;">{headline}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(preds))
    for i, mk in enumerate([k for k in _MODEL_KEYS if k in preds]):
        pred_c = preds[mk]
        sem    = get_label_semantics(selected_lang, pred_c)
        with cols[i]:
            st.markdown(
                f"""
                <div style="text-align:center;padding:8px;border-radius:6px;
                            background:{sem['badge_bg']};color:{sem['badge_color']};
                            font-size:0.9em;font-weight:600;">
                    {_MODEL_LABELS[mk]}<br>&rarr; {sem['name']}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.title("&#x1F50D; TRUST-X Live Detection")
    st.caption(
        "Enter one text and select one language — TRUST-X analyzes it through **all three** "
        "fine-tuned transformer models simultaneously: mBERT &middot; IndicBERT &middot; XLM-R"
    )

    # ── Language selector ─────────────────────────────────────────────────────
    col_lang, col_spacer = st.columns([1, 3])
    with col_lang:
        selected_lang_display = st.selectbox(
            "Target Language:",
            options=[LANGUAGE_DISPLAY[l] for l in LANGUAGES],
            index=0,
            key="live_lang_select",
        )
        selected_lang = LANGUAGE_REVERSE[selected_lang_display]

    # ── Checkpoint status row ─────────────────────────────────────────────────
    task_info = TASK_LABEL_MAPPINGS.get(selected_lang, {})
    task_name = task_info.get("task_name", "Classification")

    ckpt_cols = st.columns(3)
    for i, mk in enumerate(_MODEL_KEYS):
        has_ckpt, _, sp = check_checkpoint_exists(selected_lang, mk)
        status_icon = "&#x1F6E1;&#xFE0F; Fine-tuned" if has_ckpt else "&#x26A0;&#xFE0F; Base fallback"
        status_bg   = "#d1e7dd" if has_ckpt else "#fff3cd"
        status_col  = "#0f5132" if has_ckpt else "#664d03"
        with ckpt_cols[i]:
            st.markdown(
                f"""
                <div style="background:{status_bg};border-radius:6px;
                            padding:8px 12px;font-size:0.85em;color:{status_col};">
                    <strong>{_MODEL_LABELS[mk]}</strong><br>
                    {status_icon}<br>
                    <span style="font-size:0.78em;opacity:0.8;">{sp.name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Task badge ─────────────────────────────────────────────────────────────
    st.caption(
        f"Task: **{task_name}** | "
        f"Class 0 = `{task_info.get(0, {}).get('name')}` | "
        f"Class 1 = `{task_info.get(1, {}).get('name')}`"
    )

    # ── Sample text loader ────────────────────────────────────────────────────
    s_col1, s_col2 = st.columns([5, 1])
    with s_col2:
        if st.button("Load Sample", use_container_width=True):
            st.session_state["live_input_text"] = SAMPLE_TEXTS.get(selected_lang, "")

    default_text = st.session_state.get("live_input_text", SAMPLE_TEXTS.get(selected_lang, ""))
    user_input = st.text_area(
        "Enter social media comment or post (Latin script / Code-Mixed):",
        value=default_text,
        height=90,
        placeholder="Type or paste code-mixed sentence here...",
        key="live_text_area",
    )

    # SHAP toggle
    compute_shap = st.checkbox(
        "Compute SHAP Token Attributions (adds ~30-60 s per model on CPU)",
        value=False,
        help=(
            "Computes Shapley values using a partition text masker for all three models sequentially. "
            "Highlights tokens contributing toward or against the harmful class."
        ),
    )

    analyze_clicked = st.button(
        "&#x1F50D; Analyze with All Models", type="primary", use_container_width=True
    )

    # ── Live disclaimer ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#f8f9fa;border-left:4px solid #0d6efd;
                    padding:10px 14px;border-radius:4px;font-size:0.85em;color:#495057;margin:6px 0;">
            <strong>&#x2139;&#xFE0F; Scientific Boundaries:</strong>
            Live input is not part of the frozen evaluation dataset.
            Predictions shown here are <strong>qualitative model outputs</strong> and do not have
            dataset ground truth unless the user explicitly provides an annotated sample.
            Dataset-level metrics (Macro-F1, ECE, TWI) are <strong>never calculated</strong> for live input.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not analyze_clicked:
        return

    # ── Input validation ───────────────────────────────────────────────────────
    raw_text = user_input.strip()
    if not raw_text:
        st.warning("Please enter non-empty text to analyze.")
        return

    cleaned_input = clean_text(raw_text)
    if not cleaned_input:
        st.warning("Text contains only whitespace or invalid characters after preprocessing.")
        return

    # ── Step 1: Load all 3 models and run predictions ─────────────────────────
    st.markdown("### &#x1F4CA; 3-Model Comparative Predictions")
    st.caption("All three fine-tuned TRUST-X models analyzing the same input:")

    results = {}

    with st.spinner("Loading models and running inference for all three architectures..."):
        for mk in _MODEL_KEYS:
            try:
                loaded = load_model_and_tokenizer(selected_lang, mk)
                prob_0, prob_1, pred_class, confidence = _run_inference(
                    loaded["model"], loaded["tokenizer"], loaded["device"], cleaned_input
                )
                results[mk] = {
                    "prob_0":     prob_0,
                    "prob_1":     prob_1,
                    "pred_class": pred_class,
                    "confidence": confidence,
                    "loaded":     loaded,
                    "error":      None,
                }
            except Exception as e:
                results[mk] = {
                    "prob_0":     0.0,
                    "prob_1":     0.0,
                    "pred_class": -1,
                    "confidence": 0.0,
                    "loaded":     None,
                    "error":      str(e),
                }

    # ── Step 2: Render 3 prediction cards side-by-side ───────────────────────
    card_cols = st.columns(3)
    sem_0 = get_label_semantics(selected_lang, 0)
    sem_1 = get_label_semantics(selected_lang, 1)

    for i, mk in enumerate(_MODEL_KEYS):
        with card_cols[i]:
            r = results[mk]
            if r["error"]:
                st.markdown(
                    f"""
                    <div style="border:2px solid #dc3545;border-radius:10px;
                                padding:16px;background:#fff5f5;">
                        <strong style="color:#842029;">{_MODEL_LABELS[mk]}</strong><br>
                        <span style="font-size:0.9em;color:#842029;">
                            &#x274C; Failed to load:<br>{html_lib.escape(str(r['error'])[:200])}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                pred_sem  = get_label_semantics(selected_lang, r["pred_class"])
                ckpt_html = _checkpoint_badge(selected_lang, mk)
                _prediction_card(
                    model_label           = _MODEL_LABELS[mk],
                    pred_class            = r["pred_class"],
                    prob_0                = r["prob_0"],
                    prob_1                = r["prob_1"],
                    confidence            = r["confidence"],
                    sem_0                 = sem_0,
                    sem_1                 = sem_1,
                    pred_sem              = pred_sem,
                    checkpoint_badge_html = ckpt_html,
                )

    # ── Step 3: Model Agreement ───────────────────────────────────────────────
    successful = {mk: results[mk] for mk in _MODEL_KEYS if results[mk]["error"] is None}
    if len(successful) >= 2:
        st.markdown("---")
        _agreement_section(successful, selected_lang)

    # ── Step 4: Summary table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### &#x1F4CB; Comparative Summary")

    table_rows = ""
    for mk in _MODEL_KEYS:
        r = results[mk]
        if r["error"]:
            table_rows += (
                f"<tr><td><strong>{_MODEL_LABELS[mk]}</strong></td>"
                "<td colspan='4' style='color:#842029;'>&#x274C; Load error</td></tr>"
            )
        else:
            pred_sem    = get_label_semantics(selected_lang, r["pred_class"])
            badge_color = pred_sem.get("badge_color", "#41464b")
            table_rows += (
                f"<tr>"
                f"<td style='padding:7px 10px;'><strong>{_MODEL_LABELS[mk]}</strong></td>"
                f"<td style='padding:7px 10px;color:{badge_color};font-weight:700;'>"
                f"{html_lib.escape(pred_sem['name'])}</td>"
                f"<td style='padding:7px 10px;'>{r['prob_0']*100:.1f}%</td>"
                f"<td style='padding:7px 10px;'>{r['prob_1']*100:.1f}%</td>"
                f"<td style='padding:7px 10px;'>{r['confidence']*100:.1f}%</td>"
                f"</tr>"
            )

    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.9em;
                      border:1px solid #dee2e6;border-radius:6px;overflow:hidden;">
            <thead style="background:#f1f3f5;">
                <tr>
                    <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #dee2e6;">Model</th>
                    <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #dee2e6;">Prediction</th>
                    <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #dee2e6;">P({html_lib.escape(sem_0['name'])})</th>
                    <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #dee2e6;">P({html_lib.escape(sem_1['name'])})</th>
                    <th style="padding:8px 10px;text-align:left;border-bottom:2px solid #dee2e6;">Confidence</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # ── Step 5: SHAP (sequential, one model at a time) ────────────────────────
    if not compute_shap:
        st.markdown(
            """
            <div style="background:#e9ecef;padding:12px;border-radius:6px;
                        border-left:4px solid #6c757d;font-size:0.9em;color:#495057;margin-top:12px;">
                <strong>SHAP Token Attribution</strong> is disabled. Enable the checkbox above
                before clicking Analyze to see per-token explanations for all three models.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Harmful class index varies by language
    harmful_class_idx = 0 if selected_lang == "hindi" else 1
    sem_harmful = get_label_semantics(selected_lang, harmful_class_idx)

    # Task-specific terminology for SHAP explanation
    if selected_lang in ("tamil", "malayalam"):
        shap_pos_term = "Offensive"
        shap_neg_term = "Not Offensive"
    else:  # hindi, telugu
        shap_pos_term = "Hateful"
        shap_neg_term = "Non-Hateful"

    st.markdown("---")
    st.markdown("### &#x1F9E9; SHAP Token Attribution &mdash; Side-by-Side")
    st.caption(
        "Token contributions toward each model's predicted output, computed separately per architecture. "
        f"**Red** = pushes toward {shap_pos_term} &middot; **Blue** = pushes toward {shap_neg_term}"
    )
    st.info(
        f"Positive SHAP attribution indicates contribution toward the **{shap_pos_term}** class; "
        f"negative attribution indicates contribution toward the **{shap_neg_term}** class. "
        "SHAP values reflect learned statistical associations — "
        "they are not causal explanations of model reasoning."
    )

    shap_cols = st.columns(3)

    for i, mk in enumerate(_MODEL_KEYS):
        with shap_cols[i]:
            st.markdown(
                f"<div style='font-weight:700;font-size:1em;margin-bottom:6px;'>{_MODEL_LABELS[mk]}</div>",
                unsafe_allow_html=True,
            )

            r = results[mk]
            if r["error"]:
                st.warning(
                    f"Cannot compute SHAP: model failed to load.\n\n{r['error'][:150]}"
                )
                continue

            pred_sem_shap = get_label_semantics(selected_lang, r["pred_class"])
            st.markdown(
                f"<div style='font-size:0.85em;color:#495057;margin-bottom:8px;'>"
                f"Prediction: <strong>{html_lib.escape(pred_sem_shap['name'])}</strong> "
                f"({r['confidence']*100:.1f}%)</div>",
                unsafe_allow_html=True,
            )

            with st.spinner(f"Computing SHAP for {_MODEL_LABELS[mk]}..."):
                try:
                    loaded = r["loaded"]
                    tokens, attributions = _run_shap(
                        model        = loaded["model"],
                        tokenizer    = loaded["tokenizer"],
                        device       = loaded["device"],
                        cleaned_text = cleaned_input,
                        target_class = harmful_class_idx,
                    )
                    st.caption("SHAP Token Contributions to Model Output")
                    render_token_attributions(
                        tokens, attributions, pred_sem_shap["name"], r["confidence"]
                    )
                except Exception as ex:
                    st.warning(
                        f"SHAP attribution could not be computed for **{_MODEL_LABELS[mk]}**.\n\n"
                        f"Error: `{str(ex)[:300]}`\n\n"
                        "Predictions above remain valid — only the token-level explanation failed."
                    )


if __name__ == "__main__":
    render()
