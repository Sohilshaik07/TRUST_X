# IndicBERT Access Audit Report
**Project**: TRUST-X Trustworthiness Evaluation Framework  
**Audit Date**: 2026-09-16  
**Experiment Affected**: `telugu_indicbert_seed42`  
**Auditor**: Antigravity IDE (read-only inspection — no training, no code changes)

---

## 1. Configured Model Identifier

| Setting | Value |
|---|---|
| Config file | `configs/telugu_indicbert.yaml` |
| `model_key` field | `indicbert` |
| Resolved checkpoint | `ai4bharat/indic-bert` |
| Source of resolution | `src/models/model_factory.py` → `MODEL_REGISTRY["indicbert"]["checkpoint"]` |

The single authoritative checkpoint string in the codebase is on **line 26** of
`src/models/model_factory.py`:

```python
"indicbert": {
    "checkpoint": "ai4bharat/indic-bert",
    "name": "IndicBERT",
    "description": "Pretrained by AI4Bharat on 12 major Indian languages + English"
},
```

The same identifier appears (documentation-only) in:
- `README.md` line 43
- `reports/pre_training_validation.md` lines 114, 117, 120, 140, 142, 171-172
- `notebooks/TRUST_X_Colab_Runner.ipynb` lines 88, 98, 142
- `scripts/create_colab_notebook.py` line 127

No other checkpoint strings exist for IndicBERT anywhere in the codebase.

---

## 2. Is `ai4bharat/indic-bert` the Intended Checkpoint?

**Yes.** This is the original, canonical IndicBERT checkpoint released by AI4Bharat.
It is an ALBERT-style model pretrained on 12 major Indian languages + English.

The pre-training validation report (`reports/pre_training_validation.md`, lines 140-144)
previously confirmed that **this exact repository is gated on Hugging Face Hub** and
returned a `401 Client Error` during a live access check — confirming the failure is
access-level (authentication missing), **not** a misconfigured checkpoint ID.

> **IMPORTANT**: No checkpoint switch is needed. `ai4bharat/indic-bert` is the correct
> and intended identifier for this TRUST-X experiment.

---

## 3. Transformers Version and HF Hub Compatibility

From `requirements.txt`:

| Package | Pinned Version |
|---|---|
| `transformers` | `5.15.0` |
| `torch` | `2.11.0+cpu` |
| `python` | `3.13.12` |

`transformers` 5.15.0 is a very recent release and bundles `huggingface_hub` as a
mandatory dependency. In this version range:

- `AutoTokenizer.from_pretrained()` and `AutoModelForSequenceClassification.from_pretrained()`
  **automatically read the `HF_TOKEN` environment variable** (or the cached token written
  by `huggingface_hub.login()`) when contacting the Hub.
- No explicit `token=` argument is required in calling code — the Hub client picks it
  up transparently from the environment.

**Compatibility assessment**: `transformers==5.15.0` is fully compatible with gated-repo
authentication via environment variable or `huggingface_hub.login()`. There is no version
mismatch contributing to the failure.

> **NOTE**: The sole cause of the 401 error is the **absence of a valid HF token at
> runtime**, not a library version problem.

---

## 4. Authentication Requirements for `ai4bharat/indic-bert`

The `ai4bharat/indic-bert` repository on Hugging Face Hub is **gated**. Access requires:

1. A free Hugging Face account at https://huggingface.co
2. Acceptance of the AI4Bharat model terms at https://huggingface.co/ai4bharat/indic-bert
3. A Hugging Face User Access Token (read scope is sufficient)

Without a valid authenticated token, any call to
`AutoTokenizer.from_pretrained("ai4bharat/indic-bert")` or
`AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert")` will
immediately raise:

```
401 Client Error: Authorization Required for url: ...
Cannot access gated repo: ai4bharat/indic-bert
```

This is a gating policy enforced by AI4Bharat — it is not a code bug.

---

## 5. Current Authentication Support in the Codebase

### 5a. `src/models/model_factory.py` (the loading code)

`model_factory.py` lines 73-86 call:

```python
# Line 73
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

# Lines 76-86
config = AutoConfig.from_pretrained(checkpoint, ...)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
```

**No `token=` argument is passed.** However, in `transformers >= 4.28`, the
`huggingface_hub` client automatically reads authentication from the following
sources (in priority order):

| Priority | Source | Status in this codebase |
|---|---|---|
| 1 | `token=` kwarg in `from_pretrained()` | **Not present** (correct — avoids hardcoding) |
| 2 | `HF_TOKEN` environment variable | **Not set** — no code sets it |
| 3 | `HUGGINGFACE_HUB_TOKEN` environment variable | **Not set** — no code sets it |
| 4 | Cached token from `huggingface_hub.login()` | **Commented out** in Colab notebook |
| 5 | `~/.cache/huggingface/token` file | May exist locally if user ever ran `login()` |

### 5b. `notebooks/TRUST_X_Colab_Runner.ipynb`

The notebook includes an authentication cell (lines 96-99), but it is **fully commented out**:

```python
# from huggingface_hub import login
# login() # paste HF token if using gated ai4bharat/indic-bert
```

The cell is under the heading _"(Optional) Hugging Face Hub Authentication"_ and is
marked as optional. It is **not executed** in the default Colab run.

**This is the root cause of the 401 failure**: When Colab runs
`!python scripts/run_all_experiments.py`, no token has been supplied to the process,
so the Hub returns 401 when it reaches the IndicBERT experiments.

### 5c. Hard-coded tokens

A full grep of the entire repository confirms **no Hugging Face token is hard-coded**
anywhere. This is correct and must remain so.

---

## 6. Whether Any Code Change Is Actually Required

**No code change is required** to fix this failure.

The model ID is correct. The loading mechanism is correct. The Colab notebook already
contains the authentication cell. The only missing action is the user uncommenting
and executing the authentication cell before running the experiment.

---

## 7. Recommended Safe Colab Authentication Procedure

> **WARNING**: Do NOT hard-code a token into any file in this repository.

### Step A — Accept Model Terms (one-time, browser)
1. Visit https://huggingface.co/ai4bharat/indic-bert
2. Log in to your Hugging Face account.
3. Click **"Agree and access repository"** to accept the AI4Bharat usage terms.

### Step B — Obtain a Read Token
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"** → select **"Read"** scope → copy the token.

### Step C — Authenticate in Colab (interactive, never stored in repo)

In the Colab notebook, navigate to **Step 3 → (Optional) Hugging Face Hub Authentication**
and **uncomment both lines**:

```python
from huggingface_hub import login
login()  # paste HF token if using gated ai4bharat/indic-bert
```

Running this cell opens an interactive prompt — paste your read token there.
The token is stored in `~/.cache/huggingface/token` inside the Colab session **only**.
It is **not** written to the project directory or any tracked file.

### Step D — Alternative: Use an Environment Variable (also safe)

In a Colab code cell, before the experiment run cell:

```python
import os
os.environ["HF_TOKEN"] = "hf_..."   # paste token — runs in memory only
```

`transformers` and `huggingface_hub` will pick this up automatically.  
Do **not** save or commit a cell containing a real token value.

### Step E — Proceed to Run Experiments

After authentication, run cells in order:
```
Step 5: Pilot (Telugu mBERT — no auth needed)
Step 6: Full suite  →  !python scripts/run_all_experiments.py
```

IndicBERT experiments will now download `ai4bharat/indic-bert` successfully.

---

## 8. Summary Table

| Audit Item | Finding |
|---|---|
| Configured model ID | `ai4bharat/indic-bert` |
| ID is correct for this experiment | **Yes** |
| ID defined in | `src/models/model_factory.py` line 26 |
| Tokenizer loading method | `AutoTokenizer.from_pretrained(checkpoint, use_fast=True)` |
| Model loading method | `AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)` |
| Config loading method | `AutoConfig.from_pretrained(checkpoint, ...)` |
| `token=` kwarg passed in code | **No** (correct — avoids hardcoding) |
| `HF_TOKEN` env var set in code | **No** (not present anywhere in codebase) |
| `HUGGINGFACE_HUB_TOKEN` env var | **No** (not present anywhere in codebase) |
| `huggingface_hub.login()` in code | **Commented out** in `notebooks/TRUST_X_Colab_Runner.ipynb` lines 97-98 |
| Hard-coded token anywhere | **No** |
| `transformers` version | `5.15.0` — fully compatible with gated repo auth |
| Root cause of 401 error | Authentication cell commented out; no token present at runtime |
| Code change required | **No** |
| Required user action | Uncomment and execute the `login()` cell in Colab **before** running IndicBERT experiments |

---

## 9. Telugu mBERT Status

The Telugu mBERT experiment (`telugu_mbert_seed42`) is **unaffected**.
`bert-base-multilingual-cased` is a fully open, non-gated checkpoint requiring no
authentication.

This audit has made **zero modifications** to any existing code, configuration, dataset,
evaluation pipeline, TWI module, or Telugu mBERT outputs.

---

*Audit complete. Stopping as instructed.*
