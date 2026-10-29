# HECM Hindi Dataset Audit Report

**Project:** TRUST-X: An Integrated Trustworthiness Evaluation Framework for
Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection

| Field | Value |
|---|---|
| Audit date | 2026-09-17 |
| Auditor | Antigravity IDE (read-only — no source files modified) |
| Audit scope | Pre-processing audit before designing `Data/processed/hindi/` |
| Dataset | HECM (Hindi-English Code-Mixed) |
| Files audited | `Data/raw/hindi/HECM.txt`, `Data/raw/hindi/HECM_Class.txt` |

---

## 1. Dataset Identity

| Field | Value |
|---|---|
| Dataset name | HECM (Hindi-English Code-Mixed) |
| Language | Hindi / Romanized Hindi / Hindi-English Code-Mixed |
| Source | Directly provided by dataset author |
| Treatment | ORIGINAL AUTHOR-PROVIDED RAW DATA — do not modify |
| Role in TRUST-X | 4th language (Telugu, Tamil, Malayalam, **Hindi**) |
| Expected new experiments | 3 models x 1 language = **3 new experiments** (total 12) |

---

## 2. File Verification — PASS

| Property | HECM.txt | HECM_Class.txt |
|---|---|---|
| Path | `Data/raw/hindi/HECM.txt` | `Data/raw/hindi/HECM_Class.txt` |
| File exists | YES | YES |
| File size | 1,453,936 bytes (~1.4 MB) | 18,714 bytes (~18 KB) |
| Raw line count | 9,357 | 9,357 |
| Detected encoding | UTF-8 with BOM (utf-8-sig) | UTF-8 with BOM (utf-8-sig) |
| Line count match | **YES — exact** | |

---

## 3. Alignment Verification — PASS

| Check | Result |
|---|---|
| One-to-one line alignment | PASS |
| Total text rows | 9,357 |
| Total label rows | 9,357 |
| Line count difference | 0 |
| Blank text rows | 0 |
| Blank label rows | 0 |
| Unexpected label values | None |
| Unique raw label values | `'0'` and `'1'` only |

Every line i in `HECM.txt` maps exactly to line i in `HECM_Class.txt`.
No misaligned, blank, or missing rows found.

---

## 4. Dataset Size

| Metric | Value |
|---|---|
| Total raw lines (both files) | 9,357 |
| Total valid pairs | 9,357 |
| Expected (from paper/external) | ~9,300 |
| Difference | +57 |
| Assessment | Consistent with reported ~9.3K |

**The EXACT count is 9,357 samples.** Do not rely on rounded figures.

---

## 5. Label Distribution

| Label | Count | Percentage | Semantics Verified? |
|---|---|---|---|
| 0 | 2,823 | 30.17% | **NO** |
| 1 | 6,534 | 69.83% | **NO** |
| **TOTAL** | **9,357** | **100.00%** | |

- Class imbalance ratio (label 0 / label 1): **0.4321**
- Minority class (label 0): ~30% of dataset
- Majority class (label 1): ~70% of dataset

> **CRITICAL:** Label semantics require verification from the dataset
> author's documentation/communication before preprocessing or training.
> Do NOT assume 0 = non-hate, 1 = hate without explicit author confirmation.

---

## 6. Text Quality Statistics

### 6.1 Length Statistics

| Metric | Character Length | Word Count (whitespace-split) |
|---|---|---|
| Minimum | 5 | 1 |
| Maximum | 744 | 89 |
| Mean | 147.58 | 24.46 |
| Median | 131.0 | 21.0 |
| 95th percentile | — | 54 |

### 6.2 Feature Presence (no modification applied)

| Feature | Count | % of 9,357 |
|---|---|---|
| Texts with URLs | 1,464 | 15.65% |
| Texts with @mentions | 6,743 | 72.07% |
| Texts with #hashtags | 3,128 | 33.43% |
| Texts with emojis | 1,824 | 19.50% |
| Texts with repeated chars (4+) | 1,018 | 10.88% |
| Texts with elongated spellings | 261 | 2.79% |

**Key observations:**

- **72% of texts contain @mentions** — strongly indicates Twitter/social media origin
- **33% contain hashtags** — consistent with Twitter data
- **15% contain URLs** — important for robustness analysis (URL-as-noise perturbation)
- **~20% contain emojis** — do not remove; relevant for robustness and code-mixing analysis
- Repeated/elongated chars (~11%) are present — relevant for character-level noise robustness

### 6.3 Duplicate Summary

| Metric | Count |
|---|---|
| Unique text values | 9,239 |
| Exact duplicate text groups | 118 |
| Excess rows (duplicates) | 118 |
| Exact (text+label) duplicate groups | 93 |
| Same text, conflicting labels | **25** |

---

## 7. Script / Code-Mixing Analysis

**Method:** Unicode block heuristic.
- Devanagari = U+0900-U+097F
- Latin = ASCII A-Z / a-z codepoints only
- **No token-level language identification (LID) model used.**

| Script Category | Count | % of 9,357 |
|---|---|---|
| Devanagari-only (0 Latin chars) | 22 | 0.24% |
| Latin-only (0 Devanagari chars) | 7,815 | 83.52% |
| Mixed-script (both present) | 1,520 | 16.25% |
| Devanagari-heavy | 206 | 2.20% |
| Latin-heavy | 9,148 | 97.77% |
| Neither script (digits/punct) | 0 | 0.00% |

**Key findings:**

- ~84% of samples are Latin-script-only, written in **Romanized Hindi (Hinglish)** or English
- Only 22 samples are purely Devanagari
- 1,520 samples (16%) contain both scripts

**Critical Limitation:** Latin characters cannot reliably distinguish between
Romanized Hindi and English without a token-level LID model (e.g., fasttext-lid-176,
IndicLID). This audit explicitly states it cannot make this determination from
character heuristics alone.

Despite the dataset name "Hindi-English Code-Mixed," the data is **predominantly
Romanized/Hinglish** — a critical finding for tokenizer behavior across mBERT,
IndicBERT, and XLM-R.

---

## 8. Duplicate / Near-Duplicate Audit

### A. Exact Duplicate Texts

| Metric | Count |
|---|---|
| Exact duplicate text groups | 118 |
| Excess rows from duplicates | 118 |

Each of the 118 duplicate groups appears exactly twice in the raw file.

### B. Exact Duplicate (Text + Label) Combinations

| Metric | Count |
|---|---|
| Exact (text+label) duplicate groups | 93 |
| Excess rows | 93 |

### C. Same Text, Different Labels (Conflicting)

| Metric | Count |
|---|---|
| Texts with conflicting labels | **25** |

25 texts appear with BOTH label 0 and label 1 in the dataset.
These represent annotation disagreements or data collection artifacts.

Example conflicting text (exact, as stored):
```
@myogiadityanath Yuva nahi hai sir kyoki 2014 se ham apko Chun rahe hai aur aap...
  -> Appears as label 0 AND label 1
```

### D. Near-Duplicate Analysis

Not performed in this read-only audit. Recommended before splitting:
run MinHash LSH (`datasketch` library, Jaccard threshold ~0.8).

---

## 9. Potential Data Leakage Concerns

| Risk | Detail | Severity |
|---|---|---|
| Exact duplicate texts | 118 excess rows | MODERATE |
| Same text, same label pairs | 93 groups | MODERATE |
| **Same text, conflicting labels** | **25 texts** | **HIGH** |
| Near-duplicate leakage | Not measured | UNKNOWN |
| Template-like texts | @mention chains may share structure | POSSIBLE |

A naive random 80/10/10 split will almost certainly place duplicate pairs
in both train and test sets, artificially inflating test performance.

**Required actions before splitting (gate-blocked — do not proceed yet):**

1. Verify label semantics with HECM author
2. Decide deduplication strategy
3. Resolve 25 conflicting-label texts
4. Optionally run MinHash LSH
5. Design stratified split preserving label distribution

---

## 10. Representative Samples

> Samples shown exactly as stored in HECM.txt. Long samples truncated at 120
> characters for display only. Labels shown as raw values. Semantics NOT inferred.

### 10 Examples — Label 0 (first 10 in file)

```
[01] @SanjayAzadSln Modi bhagat pagal ho gye hain.. bhagwan ki tulna usse se kar rahe hain...
[02] @RubikaLiyaquat @ABPNews @beingvkb Abe Salo BHADWO 16 SAAL SE JHOOTH PROS RAHE HAI...
[03] @Republic_Bharat @yadavakhilesh aur ye roj Modi se sawal poochte phirte hain,bas koi...
[04] @DevDLall @wdamidoinhere @Baliramyadav007 @_sabanaqvi @Cow__Momma Bhai mujhe maaf...
[05] @sachin_rt Are BHOSDE WALAY TENDULKAR TRIBUTE TOH KISAANO KO BHI DE SKTA THA BUT...
[06] @coolfunnytshirt @rahulroushan @SanghiTheBook Ha to isme badi bat kya hai Rahul Gandhi...
[07] @vikasbha Sahi kaha kisi ne dalle dalle hote hai aur aaj ke daur mein Modi k dalle...
[08] @Razanagnostic @motabhaiclub @yadavakhilesh @MamataOfficial Bhosdi k....Tum deshdrohi...
[09] @yedhu_pillai @butterfly0o0o0o @PMOIndia fir serum institute tere baap me thoda banwaya...
[10] 1 hi chij mil sakti h ya to aarakshan le lo ya ijjat hamr chaiye full ijjat isliye...
```

### 10 Examples — Label 1 (first 10 in file)

```
[01] @msisodia Sir aap loktantra ki taakat dikha rahe the kya jab aapke paarshad Delhi me...
[02] @Sanjay_Dixit Ab Rohingya ko Bbagane ka bhu socho Bangladesh ne pehle deport kia fir...
[03] @gauravbh Modi ji ka ek bhi sabd nikla kya jo promise kiya tha black money wapas...
[04] @ndtv Desh k liye kaala din h kabza kiya jaa raha h ab to jaag jaao 2024 tak kq wait...
[05] @brijmohn @Gabbbar01 @csaajad1947 @deba6616 @RK000120 @aajtak @JournoAshutosh Kuch...
[06] @dmgonda2 Mahoday Jo road babhnan se gora choki jata hai via khodare usko bhi dekhe...
[07] @ArifRetd Baqi sub theak hai . Laikin fojiun ko shunwai ki zarurat kahan parti hai...
[08] @64wdelgado @PalashSinghal03 @rohitco96759585 @Syed_UQ @dhruv_rathee Aur tum jihad ke...
[09] @deepak9744 Sahi baat he par jese mamta or police rohingya or gundo ko nahi rok pati...
[10] @Ashok_Kashmir Maine nahi padha. Main sirf sacche logoko padhati hoon. Jo south Africa...
```

### 5 Shortest Texts

| Rank | Char Len | Label | Text (exact) |
|---|---|---|---|
| 1 | 5 | 0 | `manhoos` (Devanagari) |
| 2 | 5 | 1 | `brain` (Devanagari) |
| 3 | 6 | 1 | `toti :)` (Devanagari + emoji) |
| 4 | 8 | 0 | `Tontychor` (Devanagari) |
| 5 | 11 | 1 | `Ye sahi hai` |

### 5 Longest Texts

| Rank | Char Len | Label | Preview |
|---|---|---|---|
| 1 | 498 | 1 | `@aarav_77777 @Rahulplayboy @Priyankabjym...` (many @mentions) |
| 2 | 523 | 0 | `@Ashwani60763085 @RatishShivam @meevkt...` |
| 3 | 529 | 0 | `@07Himanshuverma @M964037561 @Jhon_The_Don...` |
| 4 | 677 | 1 | `@07Himanshuverma @ProudKisaan @VivekBharti_...` |
| 5 | 744 | 1 | `@furqanbakhas @AnkitJa92014578 @Xino95149568...` |

Longest texts are primarily @mention chains — Twitter thread reply format.
These may exceed the 512-token limit of mBERT/XLM-R/IndicBERT.

---

## 11. Information Requiring Author / Documentation Verification

The following CANNOT be established from `HECM.txt` and `HECM_Class.txt` alone:

| Item | Status |
|---|---|
| Meaning of label 0 | **UNVERIFIED — do not assume** |
| Meaning of label 1 | **UNVERIFIED — do not assume** |
| Official train/dev/test split | Not present in raw files |
| Annotation protocol | Not available locally |
| Inter-annotator agreement | Not available locally |
| Exact dataset version | Not available locally |
| Whether authors applied any preprocessing | Not available locally |
| Official recommended split strategy | Not available locally |
| Whether duplicates are intentional | Not available locally |
| How conflicting-label texts should be handled | Not available locally |

Do not fabricate any of the above. Contact the HECM dataset author.

---

## 12. Existing TRUST-X Processed Dataset Schema

Verified from `Data/processed/telugu/`, `tamil/`, `malayalam/` train.csv files.

**Columns:** `id, text, label, original_label, language, task, split`

| Column | Type | Description | Hindi value (proposed) |
|---|---|---|---|
| `id` | string | `<lang>_<split>_<NNNNN>` | `hi_train_00000`, etc. |
| `text` | string | Raw text, no normalization | as-authored |
| `label` | int | 0 or 1 | 0 or 1 (after semantics confirmed) |
| `original_label` | string | Author string label | TBD (e.g., `hate`/`non-hate`) |
| `language` | string | Language identifier | `hindi` |
| `task` | string | Task name | `hate_speech` (verify with author) |
| `split` | string | Data split | `train` / `dev` / `test` |

Future `Data/processed/hindi/train.csv`, `dev.csv`, `test.csv` MUST use this
exact schema to remain compatible with the TRUST-X training pipeline.

**DO NOT CREATE `Data/processed/hindi/` FILES UNTIL label semantics are
verified and split strategy is decided.**

---

## 13. Recommendations for Next Preprocessing Step

### Gate 1 — Blocking (complete before anything else)

- [ ] Contact HECM author: confirm meaning of label 0 and label 1
- [ ] Obtain or clarify any official split information
- [ ] Ask whether duplicates are intentional or a collection artifact

### Gate 2 — Design Decisions (before implementation)

- [ ] Deduplication strategy: remove all duplicates? keep by majority label?
- [ ] Resolution for 25 conflicting-label texts (remove / keep majority / annotate)
- [ ] Split ratio (80/10/10 or official split if provided)
- [ ] Stratify split by label to preserve 30/70 distribution

### Gate 3 — Implementation (after Gates 1 and 2)

- [ ] Optionally run MinHash LSH near-duplicate detection before splitting
- [ ] Apply deduplication and conflict resolution
- [ ] Create stratified train/dev/test split
- [ ] Write `Data/processed/hindi/{train,dev,test}.csv` in TRUST-X schema
- [ ] Verify split label distributions match expected ratios
- [ ] Add Hindi to TRUST-X pipeline (3 models x 1 language = 3 new experiments, total 12)

---

## Appendix: Supporting Tables

- [`tables/hecm_hindi_label_distribution.csv`](../tables/hecm_hindi_label_distribution.csv)
- [`tables/hecm_hindi_text_statistics.csv`](../tables/hecm_hindi_text_statistics.csv)

---

*Audit performed in read-only mode.*
*Source files HECM.txt and HECM_Class.txt were NOT modified.*
