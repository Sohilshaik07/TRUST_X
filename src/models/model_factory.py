"""
TRUST-X Model Factory Module
Supports:
1. mBERT: 'bert-base-multilingual-cased'
2. IndicBERT: 'ai4bharat/indic-bert'
3. XLM-R: 'xlm-roberta-base'

Handles device detection (CUDA GPU with memory/name logging, or CPU fallback).
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)

MODEL_REGISTRY = {
    "mbert": {
        "checkpoint": "bert-base-multilingual-cased",
        "name": "mBERT (Multilingual BERT)",
        "description": "Pretrained on 104 languages using Wikipedia"
    },
    "indicbert": {
        "checkpoint": "ai4bharat/indic-bert",
        "name": "IndicBERT",
        "description": "Pretrained by AI4Bharat on 12 major Indian languages + English"
    },
    "xlmr": {
        "checkpoint": "xlm-roberta-base",
        "name": "XLM-RoBERTa (Base)",
        "description": "Pretrained on 2.5TB of CommonCrawl data in 100 languages"
    }
}

def get_device(verbose=True):
    """
    Detects and logs hardware environment (CUDA GPU vs CPU).
    Prints GPU name, VRAM, and CUDA capability when available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if verbose:
            print("\n" + "=" * 60)
            print("HARDWARE ACCELERATION: CUDA GPU DETECTED")
            print(f"  Device Name:   {gpu_name}")
            print(f"  CUDA Version:  {torch.version.cuda}")
            print(f"  Total VRAM:    {total_mem:.2f} GB")
            print(f"  Device Index:  0 of {torch.cuda.device_count()}")
            print("=" * 60)
    else:
        device = torch.device("cpu")
        if verbose:
            print("\n" + "=" * 60)
            print("HARDWARE ENVIRONMENT: CPU-ONLY EXECUTION")
            print("  CUDA is not available. Running on system CPU.")
            print("=" * 60)
    return device

def get_model_and_tokenizer(model_key, num_labels=2, dropout=0.2):
    """
    Loads pretrained model and tokenizer given model key ('mbert', 'indicbert', 'xlmr').
    Sets up sequence classification head for binary classification.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")

    checkpoint = MODEL_REGISTRY[model_key]["checkpoint"]
    print(f"[*] Loading tokenizer from checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    print(f"[*] Loading model configuration & weights for: {MODEL_REGISTRY[model_key]['name']}")
    config = AutoConfig.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        classifier_dropout=dropout,
        finetuning_task="harmful-speech-detection"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=config
    )

    return model, tokenizer, checkpoint
