"""
TRUST-X Model Loader Module
Manages model and tokenizer loading with rigorous checkpoint policy compliance:
1. Primary: Reconstruct base transformer architecture and load fine-tuned weights from 
   checkpoints/{language}/{model}/best_checkpoint/model.safetensors via strict load_state_dict.
2. Fallback: Base pretrained model is loaded ONLY when model.safetensors is genuinely unavailable.
3. If fine-tuned checkpoint exists but fails to load, raises an explicit error (NO silent fallback).
Uses @st.cache_resource to cache memory-heavy models and tokenizers.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import torch
import streamlit as st
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoConfig
)
from safetensors.torch import load_file

from app.config import (
    CHECKPOINT_TEMPLATE,
    HF_FALLBACK_MODELS,
)


def get_inference_device(verbose: bool = False) -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[*] Inference running on CUDA: {torch.cuda.get_device_name(0)}")
        return device
    return torch.device("cpu")


def check_checkpoint_exists(language: str, model_key: str) -> Tuple[bool, Path, Path]:
    """
    Check if a fine-tuned checkpoint exists locally.
    Specifically checks for the presence of model.safetensors inside best_checkpoint/.
    Does NOT require config.json.

    Returns:
        (exists: bool, ckpt_dir: Path, safetensors_path: Path)
    """
    ckpt_dir = Path(str(CHECKPOINT_TEMPLATE).format(language=language.lower(), model=model_key.lower()))
    safetensors_path = ckpt_dir / "model.safetensors"
    has_safetensors = safetensors_path.exists() and safetensors_path.is_file()
    return has_safetensors, ckpt_dir, safetensors_path


@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model_and_tokenizer(language: str, model_key: str) -> Dict[str, Any]:
    """
    Loads model and tokenizer following strict TRUST-X Checkpoint Policy:
    
    1. Reconstructs base architecture:
       - mbert: bert-base-multilingual-cased
       - indicbert: ai4bharat/indic-bert
       - xlmr: xlm-roberta-base
       with num_labels=2, classifier_dropout=0.2, finetuning_task="harmful-speech-detection".
    2. If model.safetensors exists in best_checkpoint/:
       - Loads weights via safetensors.torch.load_file(safetensors_path)
       - Injects weights via model.load_state_dict(state_dict, strict=True)
       - If strict loading fails, raises RuntimeError (NEVER silently falls back).
    3. If model.safetensors does NOT exist:
       - Uses base pretrained model with explicit is_fine_tuned=False flag.
    4. Tokenizer:
       - Loads from ckpt_dir if tokenizer.json exists, otherwise loads from base_id.
    """
    device = get_inference_device()
    base_id = HF_FALLBACK_MODELS.get(model_key.lower())
    if not base_id:
        raise ValueError(f"Unknown model_key '{model_key}'. Available: {list(HF_FALLBACK_MODELS.keys())}")

    has_safetensors, ckpt_dir, safetensors_path = check_checkpoint_exists(language, model_key)

    # Checkpoint Loading / Strict Verification
    if has_safetensors:
        try:
            if (ckpt_dir / "config.json").exists():
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(ckpt_dir), 
                    local_files_only=True
                )
            else:
                config = AutoConfig.from_pretrained(
                    base_id,
                    num_labels=2,
                    classifier_dropout=0.2,
                    finetuning_task="harmful-speech-detection"
                )
                model = AutoModelForSequenceClassification.from_pretrained(base_id, config=config)
                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict, strict=True)

            model.to(device)
            model.eval()
            is_fine_tuned = True
            source_type = "checkpoint"
            source_path = str(safetensors_path)
            warning_msg = None
        except Exception as e:
            # Explicit error: NEVER silently fall back when fine-tuned checkpoint exists!
            raise RuntimeError(
                f"FATAL: Fine-tuned checkpoint found at {safetensors_path} but failed loading: {e}"
            )
    else:
        # Base Model Fallback (only when checkpoint genuinely does not exist)
        try:
            config = AutoConfig.from_pretrained(
                base_id,
                num_labels=2,
                classifier_dropout=0.2,
                finetuning_task="harmful-speech-detection"
            )
            model = AutoModelForSequenceClassification.from_pretrained(base_id, config=config)
            model.to(device)
            model.eval()
            is_fine_tuned = False
            source_type = "base_pretrained_fallback"
            source_path = base_id
            warning_msg = f"Fine-tuned checkpoint unavailable at {safetensors_path} — using base pretrained fallback ({base_id})."
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate base fallback architecture for {base_id}: {e}")

    # Tokenizer Loading
    tokenizer = None
    if (ckpt_dir / "tokenizer.json").exists() or (ckpt_dir / "tokenizer_config.json").exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "is_fine_tuned": is_fine_tuned,
        "source_type": source_type,
        "source_path": source_path,
        "base_model_id": base_id,
        "ckpt_dir": str(ckpt_dir),
        "device": device,
        "warning": warning_msg,
    }
