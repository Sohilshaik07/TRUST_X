# TRUST-X: Review-2 Presentation Slide Content & Comprehensive Defense Guide

**Project Title:** TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection  
**Course:** BCSE332L — Deep Learning (Course-Based Design Project) — Review 2  
**Institution:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore  
**Academic Year:** 2024–2025  

### Project Team Members
1. **Shaik Mohammad Sohil** (Reg. No: **23BAI0133**) — Neural Architecture Design, Confidence Calibration, Loss Optimization & Master Benchmark Integration
2. **Jeevan G A** (Reg. No: **23BAI0140**) — Explainability Engineering, KernelSHAP Feature Saliency Profiling & Attribution Compactness
3. **Shashank Pinnelli (Shashank P)** (Reg. No: **23BAI0152**) — Multilingual Data Engineering, Adversarial Perturbation Operators & Linguistic Robustness Profiling

### Project Supervisor
**Dr. CHELLATAMILAN T**, Professor Grade 1, School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore

---

## PART 1: SLIDE-BY-SLIDE CONTENT & COMPREHENSIVE SPEAKER NOTES

---

### Slide 01: Title Slide
- **Slide Category:** BCSE332L — Deep Learning Project Review 2
- **Slide Type:** Dark Slate Banner (`#0D233A`)
- **Slide Title:** TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate & Offensive Speech Detection
- **Visual Elements:**
  - Dark Navy background (`#0D233A`) with Slate Blue badge (`#2F5597`) and Amber divider (`#C55A11`).
  - Two-column metallic information card detailing Team Members, Registration Numbers, Specific Responsibilities, Supervisor details, and Department.
- **On-Slide Content:**
  - Project Title & Subtitle.
  - Team: Shaik Mohammad Sohil (23BAI0133), Jeevan G A (23BAI0140), Shashank Pinnelli (23BAI0152).
  - Guide: Dr. Chellatamilan T, Professor Grade 1, SCOPE, VIT Vellore.
- **Comprehensive Speaker Script:**
  > "Good morning respected faculty members and panel evaluators. Today, our team—Shaik Mohammad Sohil (23BAI0133), Jeevan G A (23BAI0140), and Shashank Pinnelli (23BAI0152)—is proud to present our Review 2 evaluation for the Deep Learning Course Project titled 'TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection', under the esteemed guidance of Dr. Chellatamilan T.
  > 
  > Over the course of this presentation, we will walk you through the pressing social and technical challenge of toxic content in code-mixed Indian languages, our multi-source dataset curation across Telugu, Tamil, Malayalam, and Hindi, the complete transformer deep learning architectures implemented and trained under identical conditions, our key empirical findings—including a dramatic class imbalance paradox—and our novel 4-pillar evaluation framework that evaluates Predictive Performance, SHAP Explainability, Linguistic Robustness, and Confidence Calibration, culminating in a live interactive Streamlit deployment. We are excited to present our work and look forward to your valuable feedback."
- **Artifact References:** `README.md`, `reports/Individual_Contribution_*.docx`, `app/main.py`.

---

### Slide 02: Problem Statement & Central Research Question
- **Slide Category:** 1. Problem & Context
- **Slide Title:** Problem Statement & Central Research Question
- **Subtitle:** The critical limitations of single-metric optimization in multilingual toxic content moderation
- **Visual Elements:** Two-column structured layout: Left Card (The Real-World Challenge, White BG) | Right Card (Central Research Question, Light Blue BG `#EEF4FA`).
- **On-Slide Content:**
  - *The Real-World Challenge:* Over 800M active internet users in India interacting across YouTube, X, and Instagram. Communication occurs via fluid code-mixing—interleaving native Indic languages (Telugu, Tamil, Malayalam, Hindi) with English and Latin transliteration (Tanglish, Hinglish, Manglish). Toxic actors exploit deliberate phonetic respelling, leetspeak, and slang to bypass filters, causing real-world polarization and harm.
  - *The Central Research Question:* *"Can standard predictive accuracy and F1 scores reliably determine whether transformer-based models are genuinely safe, robust, explainable, and calibrated for real-world deployment on Indian code-mixed social media?"*
  - *Why Standard Metrics Fail:* High accuracy masks majority-class collapse under imbalance; models remain brittle under trivial character swaps; deep networks output overconfident 99% probabilities on errors.
- **Comprehensive Speaker Script:**
  > "To understand the motivation behind TRUST-X, consider the linguistic reality of the Indian internet. With over 800 million connected citizens, online discourse rarely happens in formal, textbook English or standard Hindi. Instead, people communicate in vibrant, code-mixed vernaculars such as Hinglish, Tanglish, Telugish, and Manglish—transcribing their native Dravidian or Indo-Aryan mother tongues into Latin script with heavy English borrowing.
  > 
  > When hate speech and online toxicity manifest in these environments, simple keyword blacklists completely collapse because words are spelled phonetically with endless regional variations. But more critically, when deep learning researchers fine-tune transformer models to address this, they almost universally optimize for a single metric: test Accuracy or Macro-F1.
  > 
  > Our central research question directly challenges this prevailing practice. We hypothesize that a model achieving a 95% accuracy score can still be fundamentally untrustworthy. It can achieve high accuracy by merely predicting the majority class, crumble when an adversary replaces the letter 'e' with '3', and assign a 99% confidence score to completely hallucinated or biased predictions. TRUST-X was conceived to bridge this dangerous gap."
- **Artifact References:** `Data/raw/`, `src/data/preprocessing.py`, `src/twi/twi.py`.

---

### Slide 03: Motivation & 7 Core Research Objectives
- **Slide Category:** 1. Problem & Context
- **Slide Title:** Motivation & 7 Core Research Objectives
- **Subtitle:** Bridging the gap between leaderboard benchmarks and real-world moderation reliability
- **Visual Elements:** Top Highlight Card (The 'Clever Hans' Paradox, Gold BG `#FEF9E7`) | Bottom Two-Column Card Grid (Objectives 1–4 and Objectives 5–7 + Deployment).
- **On-Slide Content:**
  - *Motivation (The Clever Hans Paradox):* Deep models often exploit spurious dataset artifacts and unigram frequencies rather than learning true semantic toxicity. Trustworthy AI demands multidimensional accountability.
  - *Objective 1:* Multi-source data harmonization across 4 corpora (~69,500 samples) with strict deduplication and conflict group purging.
  - *Objective 2:* Multi-architecture benchmarking of 3 diverse transformer families: mBERT (178M), IndicBERT (87M), and XLM-R (250M).
  - *Objective 3:* Class imbalance mitigation via dynamic inverse class weighting in Cross-Entropy Loss.
  - *Objective 4:* Empirical divergence analysis between Accuracy and Macro-F1 under extreme skew (up to 24:1).
  - *Objective 5:* Game-theoretic explainability profiling using KernelSHAP top-20% Explanation Compactness.
  - *Objective 6:* Adversarial linguistic robustness stress-testing across 6 deterministic perturbation operators.
  - *Objective 7:* Post-hoc calibration profiling (ECE, Brier) and formulation of Composite Trustworthiness Index (TWI) with sensitivity analysis.
  - *Deployment:* Modular Streamlit application with interactive prediction, heatmaps, and audit controls.
- **Comprehensive Speaker Script:**
  > "On Slide 3, we define our specific research objectives. Our work is grounded in overcoming what machine learning theorists call the 'Clever Hans' effect: where an NLP model appears highly capable on an academic leaderboard, but achieves its score by exploiting spurious unigram co-occurrences or dataset artifacts rather than genuine contextual understanding.
  > 
  > To address this systematically, we formulated seven core objectives:
  > First, harmonizing four heterogeneous datasets across Dravidian and Indo-Aryan families with strict deduplication.
  > Second, benchmarking three structurally distinct transformer backbones: multilingual BERT, IndicBERT with its compact ALBERT parameter-sharing design, and the high-capacity XLM-RoBERTa.
  > Third, implementing dynamic inverse class weighting into our PyTorch loss functions.
  > Fourth, empirically demonstrating why Accuracy is mathematically deceptive under extreme class skew.
  > Fifth, measuring explainability through KernelSHAP explanation compactness.
  > Sixth, stress-testing models against six character- and token-level perturbation operators.
  > And seventh, quantifying Expected Calibration Error and synthesizing all metrics into a unified Composite Trustworthiness Index, complete with a production-ready Streamlit interface."
- **Artifact References:** `src/training/trainer.py`, `src/twi/twi.py`, `app/main.py`.

---

### Slide 04: Proposed TRUST-X Framework Architecture
- **Slide Category:** 2. Architecture & Design
- **Slide Title:** Proposed TRUST-X Framework Architecture
- **Subtitle:** End-to-end pipeline from multilingual ingestion to multi-pillar evaluation and deployment
- **Visual Elements:** Two-column split: Left Card (Detailed 4-Tier Pipeline Walkthrough) | Right Card (High-resolution framework schematic embedded from `reports/extracted_images/image1.png`).
- **On-Slide Content:**
  - *Tier 1: Multilingual Ingestion & Sanitization:* Ingestion of 4 heterogeneous corpora (.xlsx, .tsv, .txt); regex cleaning of URLs and handles; script preservation; deduplication; conflict group removal.
  - *Tier 2: Transformer Fine-Tuning Engine:* Parameter-controlled fine-tuning of mBERT, IndicBERT, and XLM-R with weighted cross-entropy loss and AdamW.
  - *Tier 3: 4-Pillar Evaluation Engine:* Pillar 1 (Performance: Macro-F1), Pillar 2 (Explainability: SHAP Compactness), Pillar 3 (Robustness: 6 Perturbation Operators), Pillar 4 (Calibration: ECE & Brier score).
  - *Tier 4: Unified Metric & Interactive Deployment:* TWI composite calculation with 4 sensitivity weighting profiles, deployed in Streamlit.
- **Comprehensive Speaker Script:**
  > "Slide 4 presents the macro-architecture of TRUST-X, illustrated in the system flow diagram on the right. The framework is engineered into four modular, decoupled tiers:
  > 
  > In Tier 1, raw social media comments from diverse sources pass through our ingestion engine. Here, URLs and handles are sanitized, exact duplicate reposts are removed, conflicting annotator labels are purged, and native Indic scripts are strictly protected.
  > In Tier 2, tokenized sequences enter our deep learning engine. All three models are trained under identical hyperparameter conditions using weighted cross-entropy loss to prevent minority class neglect.
  > In Tier 3, each trained model is routed through our four independent evaluation modules: Predictive Performance, SHAP Explainability, Linguistic Robustness, and Confidence Calibration.
  > Finally, in Tier 4, these orthogonal dimensions are mapped into our Composite Trustworthiness Index (TWI) and surfaced through an interactive Streamlit application that allows real-time auditing and live testing."
- **Artifact References:** `reports/extracted_images/image1.png`, `src/training/trainer.py`, `src/twi/twi.py`.

---

### Slide 05: Dataset Overview & Linguistic Characteristics
- **Slide Category:** 3. Dataset & Preprocessing
- **Slide Title:** Dataset Overview & Linguistic Characteristics
- **Subtitle:** Four diverse corpora representing Dravidian and Indo-Aryan language families
- **Visual Elements:** 2x2 Grid of stylized cards covering Telugu, Tamil, Malayalam, and Hindi corpora + Bottom Banner highlighting the distinction between Hate Speech and Offensive Language.
- **On-Slide Content:**
  - *Telugu (HOLD Dataset):* 3,941 comments | Dravidian Family | Binary Hate & Offensive Detection | Script: Telugu script + Romanized Telugu | 65% non-hate, 35% hate.
  - *Tamil (DravidianCodeMix):* 41,879 comments | Dravidian Family | Offensive Language Detection | Script: Tamil script + Tanglish | 81% non-offensive, 19% offensive (~4.3:1 skew).
  - *Malayalam (DravidianCodeMix):* 14,482 comments | Dravidian Family | Offensive Language Detection | Script: Malayalam script + Manglish | 1,844 test set | Extreme Skew: 24:1 non-offensive to offensive ratio in evaluation split!
  - *Hindi (HECM Corpus):* 9,214 comments | Indo-Aryan Family | Hate & Extreme Content | Script: Devanagari script + Hinglish | 53% non-hate, 47% hate (~1.1:1 balanced).
  - *Key Task Distinction:* Hate Speech attacks protected identity characteristics (religion, caste, gender); Offensive Language involves profanity and vulgarity without identity targeting.
- **Comprehensive Speaker Script:**
  > "Moving to our data foundation on Slide 5: we curated four representative datasets spanning two distinct language families: the Dravidian family, represented by Telugu, Tamil, and Malayalam, and the Indo-Aryan family, represented by Hindi.
  > 
  > These corpora encompass nearly 70,000 real-world social media comments:
  > The Telugu HOLD corpus contains 3,941 comments combining Telugu script and Romanized Telugish.
  > The Tamil DravidianCodeMix corpus is our largest, with 41,879 YouTube comments dominated by Tanglish.
  > The Malayalam corpus features 14,482 comments. In its official evaluation split of 1,844 comments, there is a massive 24:1 class skew—only 76 comments are offensive, while 1,768 are non-offensive.
  > The Hindi HECM corpus contains 9,214 comments spanning Devanagari and Hinglish with balanced class polarity.
  > 
  > We also highlight the crucial distinction between Hate Speech and Offensive Language at the bottom of the slide: Hate speech attacks protected group identities like religion, caste, or gender, whereas offensive language encompasses crude profanity and aggressive colloquialisms without targeted group derogation."
- **Artifact References:** `Data/raw/`, `Data/processed/`, `src/data/loaders.py`.

---

### Slide 06: Raw Dataset Structures & Ingestion Protocols
- **Slide Category:** 3. Dataset & Preprocessing
- **Slide Title:** Raw Dataset Structures & Ingestion Protocols
- **Subtitle:** Handling heterogeneous formats, non-standard encodings, and inverted label polarities
- **Visual Elements:** Two-column card layout: Left Card (Engineering Challenges in `loaders.py`) | Right Card (Dataset Ingestion & Harmonization Matrix).
- **On-Slide Content:**
  - *Ingestion Engine (`loaders.py`):* Telugu ingested from Excel (.xlsx) requiring custom string conversion and NaN filtering; Tamil ingested from pre-split TSVs with multiline comment handling; Malayalam ingested via streaming TSV parser resolving trailing delimiters; Hindi ingested from raw line-by-line text (.txt) with custom delimiter parsing.
  - *Critical Inverted Polarity Handling:* In raw Hindi HECM, label '0' denoted Hateful and '1' denoted Non-Hateful. Our loader explicitly inverted this to match global canonical harmony: 0 = Non-Toxic, 1 = Toxic.
  - *Format Matrix:* Complete summary of file types, raw labels, harmonized mappings, and partition splits across all 4 languages.
- **Comprehensive Speaker Script:**
  > "Slide 6 dives into the real-world software engineering challenges we resolved in Tier 1. In academic benchmarks, textbooks often assume that data arrives in clean, uniform CSV tables. In reality, our four corpora arrived in four completely disparate formats: Excel spreadsheets for Telugu, TSVs with unescaped quotes for Tamil and Malayalam, and raw delimited text files for Hindi.
  > 
  > Our data ingestion engine in `src/data/loaders.py` had to address two critical anomalies:
  > First, malformed social media strings: comments on YouTube and Twitter frequently contain unescaped line breaks, trailing tabs, and unprintable control characters that crash standard CSV readers. We built robust streaming parsers to reconstruct these lines.
  > Second, and most importantly: the raw Hindi HECM corpus arrived with inverted polarity labels—label 0 meant Hateful and label 1 meant Non-Hateful. Had this not been caught and harmonized, models would have learned backwards supervisory signals. Our loader systematically inverted the Hindi labels, ensuring that across all four languages, 0 strictly denotes Non-Toxic and 1 strictly denotes Toxic or Offensive."
- **Artifact References:** `src/data/loaders.py`, `Data/raw/`.

---

### Slide 07: Data Preprocessing Pipeline
- **Slide Category:** 3. Dataset & Preprocessing
- **Slide Title:** Data Preprocessing & Integrity Protocols
- **Subtitle:** Sanitization, unicode script preservation, deduplication, and conflict resolution
- **Visual Elements:** 4-Column Sequential Pipeline Cards: Step 1 (Noise Sanitization), Step 2 (Script Preservation), Step 3 (Deduplication Audit), Step 4 (Conflict Resolution).
- **On-Slide Content:**
  - *Step 1: Noise Sanitization:* Strips URLs (`https?://\S+`) and mentions (`@username`) via regex; decodes HTML entities (`&amp;` $\rightarrow$ `&`); removes zero-width characters (`\u200b`, `\ufeff`).
  - *Step 2: Script Preservation Principle:* Preserves native Indic unicode blocks (Telugu `U+0C00–0C7F`, Tamil `U+0B80–0BFF`, Malayalam `U+0D00–0D7F`, Devanagari `U+0900–097F`) alongside Roman characters. Retains punctuation and emojis.
  - *Step 3: Deduplication Audit:* Exact-match deduplication post-normalization to eliminate viral reposts, preventing train-test contamination.
  - *Step 4: Conflict Resolution:* Purges entire conflict groups where identical text received contradictory annotations (0 and 1) from different human annotators, eliminating toxic gradient noise.
- **Comprehensive Speaker Script:**
  > "Slide 7 outlines our four-step preprocessing and data integrity pipeline implemented in `src/data/preprocessing.py`.
  > 
  > In Step 1, we sanitize superficial noise: URLs and usernames are stripped using regular expressions to ensure the transformer does not overfit to specific user handles or domain names. Zero-width spaces and HTML artifacts are decoded.
  > In Step 2, we enforce our strict 'Script Preservation Principle': unlike standard English NLP pipelines that strip non-ASCII characters, we strictly protect native Telugu, Tamil, Malayalam, and Devanagari Unicode blocks. Code-mixed text relies on the nuanced interplay between native characters and Roman transliterations.
  > In Step 3, we deduplicate exact string matches to eliminate viral retweets and bot spam that would artificially inflate validation performance.
  > In Step 4, we resolve crowd annotator conflict: when two different human annotators assign contradictory labels (0 and 1) to the exact same comment, keeping both injects toxic gradient noise during training. Our pipeline detects and purges all conflict groups to guarantee clean, unambiguous supervision."
- **Artifact References:** `src/data/preprocessing.py`, `src/data/audit.py`.

---

### Slide 08: Final Processed Data Schema & Audit Verification
- **Slide Category:** 3. Dataset & Preprocessing
- **Slide Title:** Final Processed Data Schema & Audit Verification
- **Subtitle:** Rigorous schema normalization and zero-leakage partition auditing
- **Visual Elements:** Two-Column Layout: Left Card (Standardized 7-Column DataFrame Schema) | Right Card (Verification Audit Protocols, Teal BG `#EAF8F6`).
- **On-Slide Content:**
  - *Standardized Schema (`audit.py`):*
    - `id` (String): Unique global identifier (e.g. `te_00142`, `hi_00891`).
    - `text` (String): Cleaned, normalized, script-preserved multilingual code-mixed string.
    - `label` (Integer): Canonical binary target — `0` for Non-Toxic, `1` for Toxic/Offensive.
    - `original_label` (String): Preserved raw source label for lineage traceability.
    - `language` (String): Categorical indicator (`telugu`, `tamil`, `malayalam`, `hindi`).
    - `task` (String): Task category (`hate_speech`, `offensive_language`).
    - `split` (String): Deterministic partition (`train`, `val`, `test`).
  - *Verification Audit Protocols:*
    - Verified 0 null values across all text and label fields.
    - Verified 0 string overlaps between training and test sets post-deduplication.
    - Confirmed non-empty Indic and Roman character distributions.
    - Verified binary target strictly in {0, 1}.
    - Deterministic persistence to `Data/processed/` with checksum verification.
- **Comprehensive Speaker Script:**
  > "Slide 8 details our data schema harmonization and verification assertions in `src/data/audit.py`.
  > 
  > Every single sample across all four languages is ingested into a unified 7-column schema: a globally unique identifier, the normalized text, the canonical binary label (0 or 1), the original label for provenance tracking, the language indicator, the specific task, and the partition split.
  > 
  > Before a single training epoch begins, our automated audit suite executes five verification checks: confirming zero null or empty strings, zero text leakage between train and test partitions, validated unicode script composition, binary label integrity, and reproducible parquet/CSV persistence with checksums. This guarantees that all downstream deep learning models are evaluated on verified, leak-free data."
- **Artifact References:** `src/data/audit.py`, `Data/processed/`.

---

### Slide 09: Dataset Distribution & Class Imbalance Analysis
- **Slide Category:** 3. Dataset & Preprocessing
- **Slide Title:** Dataset Distribution & Class Imbalance Analysis
- **Subtitle:** Quantifying sample sizes, split counts, and extreme majority-class skews
- **Visual Elements:** Left Card (Detailed Empirical Distribution Breakdown) | Right Card (High-resolution distribution chart embedded from `figures/audit_class_distributions.png`).
- **On-Slide Content:**
  - *Telugu:* Total = 3,941 | Train: 2,758, Val: 591, Test: 592 | Imbalance: ~65% Non-Hate vs 35% Hate (~1.9:1).
  - *Tamil:* Total = 41,879 | Train: 29,315, Val: 6,282, Test: 6,282 | Imbalance: ~81% Non-Offensive vs 19% Offensive (~4.3:1).
  - *Malayalam:* Total = 14,482 | Train: 10,137, Val: 2,172, Test: 2,173 (Official eval: 1,844) | Extreme Skew: 95.8% Non-Offensive vs 4.2% Offensive (24:1!).
  - *Hindi:* Total = 9,214 | Train: 6,449, Val: 1,382, Test: 1,383 | Imbalance: ~53% Non-Hate vs 47% Hate (~1.1:1 balanced).
  - *Critical Takeaway:* The 24:1 skew in Malayalam serves as an empirical stress-test for majority-class collapse in deep neural networks.
- **Comprehensive Speaker Script:**
  > "Slide 9 breaks down the class distributions across our four corpora, illustrated in the distribution plot on the right.
  > 
  > Notice the dramatic variation in class balance across languages:
  > Hindi is relatively well-balanced with a 53% to 47% ratio.
  > Telugu shows moderate imbalance at roughly 1.9 to 1.
  > Tamil exhibits a 4.3 to 1 imbalance, where offensive comments account for less than 20% of the dataset.
  > Most strikingly, Malayalam presents an extreme 24 to 1 majority-class skew in its official evaluation set: out of 1,844 test instances, exactly 1,768 are non-offensive and only 76 are offensive.
  > 
  > This 24:1 skew in Malayalam is not an error; it reflects real-world social media dynamics, where severe toxicity represents a rare, buried signal. This extreme distribution serves as our definitive empirical testbed for evaluating model resilience against majority-class collapse."
- **Artifact References:** `figures/audit_class_distributions.png`, `tables/dataset_summary.csv`.

---

### Slide 10: Input & Output of the Neural Network
- **Slide Category:** 4. Neural Architecture & Mathematics
- **Slide Title:** Input & Output Transformation Pipeline
- **Subtitle:** Concrete trace of a code-mixed sequence through tokenization, embeddings, and logits
- **Visual Elements:** Top Highlight Card (Concrete Code-Mixed Example Walkthrough) | Bottom 3-Card Transformation Pipeline: Card 1 (Subword Tokenizer) | Card 2 (Transformer Layers) | Card 3 (Classification Head).
- **On-Slide Content:**
  - *Input Example:* `"Yeh video bilkul bakwas hai, time waste mat karo 😡"` (Hinglish: Hindi in Latin script + Slang + Emoji) $\rightarrow$ Translation: 'This video is total rubbish, do not waste your time 😡' | Ground Truth: Toxic (1).
  - *1. Subword Tokenizer:* Subword decomposition `['[CLS]', 'Yeh', 'video', 'bil', '##kul', 'bak', '##was', 'hai', ',', 'time', 'waste', 'mat', 'karo', '[SEP]']` $\rightarrow$ Input IDs vector $\mathbf{x} \in \mathbb{Z}^{128}$ + Attention mask $\mathbf{m} \in \{0, 1\}^{128}$.
  - *2. Transformer Layers:* Summation of token, position, and segment embeddings $\rightarrow$ 12 Transformer layers with self-attention and FFN $\rightarrow$ Pooled `[CLS]` latent vector $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$.
  - *3. Classification Head:* Dropout ($p=0.1$) $\rightarrow$ Linear projection $\mathbf{z} = \mathbf{W} \mathbf{h} + \mathbf{b} \in \mathbb{R}^2$ (e.g. $[-1.82, +2.94]$) $\rightarrow$ Softmax: $[P_0 = 0.008, P_1 = 0.992] \rightarrow \hat{y} = 1$ (Toxic).
- **Comprehensive Speaker Script:**
  > "On Slide 10, we provide a concrete, step-by-step walkthrough of how a code-mixed sentence actually moves through our neural network.
  > 
  > Consider the realistic Hinglish comment: 'Yeh video bilkul bakwas hai, time waste mat karo 😡' ('This video is total rubbish, do not waste your time').
  > In Stage 1, the subword tokenizer segments the text. Notice how the colloquial Hindi slang 'bakwas' is split into subword units 'bak' and '##was'. This subword segmentation is vital: it ensures that informal colloquialisms and phonetic misspellings not present in standard dictionaries are represented without out-of-vocabulary tokens.
  > In Stage 2, these tokens are converted to embeddings, combined with positional encodings, and propagated through 12 transformer encoder blocks. Across these layers, self-attention allows 'bakwas' to attend dynamically to 'bilkul' (absolutely) and 'waste', contextually amplifying its toxicity. The aggregated representation is extracted at the 768-dimensional [CLS] token.
  > In Stage 3, the [CLS] vector passes through a 0.1 dropout layer into a linear classification head, producing raw logits $[-1.82, +2.94]$. Applying softmax yields class probabilities of 0.008 for non-toxic and 0.992 for toxic, triggering a final prediction of Class 1."
- **Artifact References:** `src/models/model_factory.py`, `src/training/trainer.py`.

---

### Slide 11: What is a Pretrained Transformer?
- **Slide Category:** 4. Neural Architecture & Mathematics
- **Slide Title:** Foundational Concepts: Pretrained Transformers
- **Subtitle:** Understanding self-supervised pretraining vs. downstream task-specific fine-tuning
- **Visual Elements:** Two-Column Comparison: Left Card (1. Self-Supervised Pretraining) | Right Card (2. Downstream Fine-Tuning in TRUST-X, Light Blue BG `#EEF4FA`).
- **On-Slide Content:**
  - *1. Self-Supervised Pretraining:* Trained on massive unlabeled web corpora (Wikipedia, Common Crawl, news) across 100+ languages without human annotations. Uses Masked Language Modeling (MLM)—predicting 15% masked tokens using bidirectional context. Learns universal grammar, cross-lingual alignment, and semantics. Computationally intensive, requiring GPU/TPU clusters.
  - *2. Downstream Fine-Tuning (TRUST-X):* Transfer learning paradigm: pretrained weights provide an informed, highly structured parameter initialization. Replaces the pretraining head with a task-specific linear layer $\mathbf{W} \in \mathbb{R}^{2 \times 768} + \mathbf{b} \in \mathbb{R}^2$. Updates all 12 transformer layers jointly using task-specific labeled data via end-to-end backpropagation. Achieves high sample efficiency in 3 epochs.
- **Comprehensive Speaker Script:**
  > "To ensure clarity for the entire panel, Slide 11 breaks down the foundational deep learning mechanism at the heart of our work: the two-stage transfer learning paradigm.
  > 
  > Stage 1 is Self-Supervised Pretraining. Instead of requiring millions of expensive human labels, transformers are trained on hundreds of gigabytes of raw text using pretext tasks like Masked Language Modeling. The network learns to predict randomly masked words using surrounding bidirectional context. In doing so, it organically acquires deep syntactic structures, cross-lingual lexical associations, and world knowledge.
  > 
  > Stage 2 is Task-Specific Fine-Tuning, which is what we execute in TRUST-X. We discard the original pretraining head and replace it with a lightweight binary classification head. We then adapt the entire network—all 12 layers and 87 to 250 million parameters—using our annotated code-mixed datasets. Because the model already understands language syntax and cross-lingual semantics, it achieves strong performance with just a few thousand training samples in 3 epochs."
- **Artifact References:** `src/models/model_factory.py`, `src/training/trainer.py`.

---

### Slide 12: Complete Neural Network Architecture
- **Slide Category:** 4. Neural Architecture & Mathematics
- **Slide Title:** Complete Neural Network Architecture
- **Subtitle:** Detailed breakdown of the 8 structural components from input tokenization to binary logits
- **Visual Elements:** Left Card (8 Structural Architectural Components) | Right Card (High-resolution architectural diagram embedded from `reports/extracted_images/image2.png`).
- **On-Slide Content:**
  - *Component 1:* Subword Tokenizer: WordPiece (mBERT) or SentencePiece (IndicBERT, XLM-R).
  - *Component 2:* Embedding Layer: Summation of Token, Positional, and Segment Embeddings into $\mathbb{R}^{768}$.
  - *Component 3:* Bidirectional Multi-Head Self-Attention: 12 parallel heads with projection dimension $d_k = 64$.
  - *Component 4:* Feed-Forward Networks (FFN): Two-layer MLP with expansion to 3,072 dimensions and GELU activation.
  - *Component 5:* Residual Skip Connections & Layer Normalization: $\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$.
  - *Component 6:* Pooled `[CLS]` Representation: Dense projection capturing sequence-level context.
  - *Component 7:* Regularization Dropout Layer ($p = 0.1$).
  - *Component 8:* Linear Classification Head: Binary projection matrix $\mathbf{W} \in \mathbb{R}^{2 \times 768}$ producing unnormalized logits $\mathbf{z} \in \mathbb{R}^2$.
- **Comprehensive Speaker Script:**
  > "Slide 12 provides the comprehensive engineering view of our neural network architecture, shown in the technical diagram on the right.
  > 
  > The architecture consists of eight precise mathematical components:
  > First, the subword tokenizer that maps code-mixed text to integer IDs.
  > Second, the embedding layer that fuses token identities, positional coordinates, and segment indicators into 768-dimensional vectors.
  > Third, the 12-head multi-head self-attention block that computes token-to-token attention maps.
  > Fourth, the position-wise feed-forward network with a 3,072-dimensional hidden expansion and GELU activation.
  > Fifth, residual skip connections and Layer Normalization following each sub-layer, ensuring smooth gradient propagation.
  > Sixth, the pooled [CLS] extraction layer that condenses sequence semantics.
  > Seventh, a 0.1 dropout regularization layer to prevent feature co-adaptation.
  > And eighth, the linear classification head that maps the 768-dimensional latent vector to two output class logits."
- **Artifact References:** `reports/extracted_images/image2.png`, `src/models/model_factory.py`.

---

### Slide 13: Inside the Transformer: Self-Attention & Contextual Embeddings
- **Slide Category:** 4. Neural Architecture & Mathematics
- **Slide Title:** Inside the Transformer: Self-Attention Mechanics
- **Subtitle:** How Scaled Dot-Product Attention captures non-local dependencies and negations
- **Visual Elements:** Top Formula Card (Scaled Dot-Product Attention Equation & Projections) | Bottom Two Cards: Left Card (Example A: Positive Valence Context, Green `#228B22`) | Right Card (Example B: Inverted Negation Context, Red `#C00000`).
- **On-Slide Content:**
  - *Self-Attention Formula:* $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$
  - *Components:* $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ are linear projections via $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{768 \times 64}$. Dot product calculates semantic relevance between all token pairs. Scaling factor $\sqrt{d_k} = \sqrt{64} = 8$ prevents dot products from exploding in high dimensions, preventing vanishing softmax gradients.
  - *Example A:* `"This movie is excellent"` $\rightarrow$ `"excellent"` attends to `"movie"`, producing a strongly positive contextual embedding.
  - *Example B:* `"This movie is not excellent"` $\rightarrow$ `"excellent"` receives heavy attention weights from `"not"`, dynamically inverting its embedding into negative territory.
  - *Relevance to Code-Mixing:* Captures dispersed negation markers ('nahi', 'mat', 'illai') separated from offensive slang by multiple colloquial words.
- **Comprehensive Speaker Script:**
  > "Slide 13 explores the mathematical core of the transformer: Scaled Dot-Product Self-Attention.
  > 
  > In the formula at the top, Queries, Keys, and Values are projected from token embeddings. The dot product between Queries and Keys measures how much attention token i should pay to token j. Dividing by the square root of key dimension 64—which is 8—is critical: without this scaling factor, dot products in high dimensions grow excessively large, pushing the softmax function into regions with near-zero gradients.
  > 
  > The comparison at the bottom explains why this is vastly superior to older embeddings like Word2Vec or TF-IDF. In Word2Vec, the word 'excellent' always has the exact same vector. But in a transformer, as shown in Example B, when 'not' precedes 'excellent', self-attention routes information directly from 'not' into 'excellent', completely inverting its vector into a negative valence. In Indian code-mixed social media, where negation words like 'nahi', 'mat', or 'illai' are often separated from profanity by several colloquial words, self-attention effortlessly captures these long-range dependencies."
- **Artifact References:** `src/models/model_factory.py`, `src/training/trainer.py`.

---

### Slide 14: Three Pretrained Architectures Compared
- **Slide Category:** 4. Neural Architecture & Mathematics
- **Slide Title:** Three Pretrained Architectures Compared
- **Subtitle:** Structural comparison of mBERT, IndicBERT, and XLM-RoBERTa
- **Visual Elements:** 3-Column Comparative Specification Cards: Column 1 (mBERT) | Column 2 (IndicBERT, Highlighted Gold) | Column 3 (XLM-RoBERTa).
- **On-Slide Content:**
  - *mBERT (`bert-base-multilingual-cased`):* 178 Million parameters | 119,547 WordPiece vocabulary | 104 languages | 12 layers, 768 hidden, 12 heads | Pretrained on Wikipedia | Standard multilingual baseline | Limitation: Equal vocabulary allocation dilutes capacity across 104 languages.
  - *IndicBERT (`ai4bharat/indic-bert`):* 87 Million parameters (Ultra-Compact!) | 200,000 SentencePiece vocabulary | 12 Indian languages + English | 12 layers, 768 hidden, 12 heads | Key Innovation: Cross-layer parameter sharing (ALBERT design) | Pretrained on 9B tokens of Indian text (AI4Bharat) | Highly parameter-efficient.
  - *XLM-RoBERTa (`xlm-roberta-base`):* 250 Million parameters (High Capacity) | 250,002 SentencePiece vocabulary | 100 languages | 12 layers, 768 hidden, 12 heads | Pretrained on 2.5TB Common Crawl (CC-100) | RoBERTa training recipe (dynamic masking, larger batches, no NSP) | Exceptional cross-lingual representation.
- **Comprehensive Speaker Script:**
  > "Slide 14 compares the three pretrained transformer backbones selected for TRUST-X.
  > 
  > First, mBERT is Google's canonical multilingual baseline. With 178 million parameters and a 119k WordPiece vocabulary, it covers 104 languages trained on Wikipedia. However, because Wikipedia text is formal, its exposure to informal Indian code-mixing is limited.
  > 
  > Second, IndicBERT represents parameter efficiency. Built by AI4Bharat on the ALBERT architecture, it shares parameters across all 12 transformer layers. Consequently, it requires only 87 million parameters—less than half of mBERT and one-third of XLM-R—while featuring a massive 200k vocabulary tailored specifically to 12 Indian languages trained on 9 billion tokens.
  > 
  > Third, XLM-RoBERTa is the heavy lifter. With 250 million parameters and a 250k SentencePiece vocabulary, it was pretrained on 2.5 terabytes of filtered Common Crawl data using RoBERTa's optimized recipe: dynamic masking, large mini-batches, and no Next Sentence Prediction objective. This gives it immense representational capacity, which proves pivotal in low-resource and code-mixed settings."
- **Artifact References:** `src/models/model_factory.py`, `configs/config.yaml`.

---

### Slide 15: Training Configuration & Hyperparameters
- **Slide Category:** 5. Training & Optimization
- **Slide Title:** Training Configuration & Hyperparameters
- **Subtitle:** Standardized, leakage-free hyperparameter protocol across all 12 experiments
- **Visual Elements:** Left Card (Standardized Hyperparameter Specifications) | Right Card (Weighted Cross-Entropy Loss Formulation & Mathematical Intuition, Gold Accent `#C55A11`).
- **On-Slide Content:**
  - *Hyperparameters:* Optimizer = AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | Base LR = $2 \times 10^{-5}$ | Weight Decay = 0.01 | LR Schedule = Linear warmup (10% steps) + Linear decay | Batch Size = 16 | Max Sequence Length = 128 tokens | Epochs = 3 | Random Seed = 42 (deterministic) | Checkpoint Selection = Best validation Macro-F1.
  - *Weighted Cross-Entropy Loss Formulation:*
    $$\mathcal{L}_{\text{WCE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \log P(y_i \mid \mathbf{x}_i)$$
  - *Dynamic Inverse Class Weighting:*
    $$w_c = \frac{N}{2 \cdot N_c}$$
  - *Gradient Scaling:* In Malayalam (24:1 skew), standard loss rewards predicting the majority class. Inverse weighting heavily penalizes minority errors ($w_{\text{minority}} \approx 12.0$ vs $w_{\text{majority}} \approx 0.52$), providing 23$\times$ stronger gradient updates to combat majority-class collapse.
- **Comprehensive Speaker Script:**
  > "Slide 15 presents our standardized training configuration. To maintain scientific validity, all 12 experiments—spanning 4 languages across 3 models—were trained using identical, rigorous hyperparameters.
  > 
  > We utilized the AdamW optimizer with a base learning rate of 2e-5, decoupled weight decay of 0.01, a batch size of 16, and a maximum sequence length of 128 tokens. Training was conducted for 3 epochs with a 10% linear warmup schedule. A deterministic seed of 42 was fixed across all random number generators. Crucially, model checkpoints were saved based on peak validation Macro-F1, never training loss or accuracy.
  > 
  > The mathematical card on the right explains our loss formulation: Weighted Cross-Entropy. In heavily imbalanced corpora like Malayalam, standard cross-entropy encourages the model to ignore toxic comments because minority errors represent a tiny fraction of the total loss. By dynamically computing inverse class weights $w_c = N / (2 \cdot N_c)$, an error on a minority toxic comment generates over 23 times larger gradient signal than a majority non-toxic error. This directly forces the optimizer to learn discriminative minority features."
- **Artifact References:** `src/training/trainer.py`, `configs/config.yaml`.

---

### Slide 16: How the Model Learns During Training
- **Slide Category:** 5. Training & Optimization
- **Slide Title:** How the Model Learns During Training
- **Subtitle:** The complete 8-step iterative forward, loss, backward, and optimization loop
- **Visual Elements:** Left Card (8-Step Training Loop Execution) | Right Card (Numerical Gradient Optimization Walkthrough, Light Blue BG `#EEF4FA`).
- **On-Slide Content:**
  - *8-Step Loop:* 1. Mini-batch loading (16 sequences) $\rightarrow$ 2. Forward pass $\rightarrow$ 3. Loss calculation via WCE $\rightarrow$ 4. Numerical loss penalty $\rightarrow$ 5. Backward pass via chain rule $\rightarrow$ 6. Gradient clipping (norm = 1.0) $\rightarrow$ 7. Parameter update via AdamW $\rightarrow$ 8. Gradient zeroing (`zero_grad()`).
  - *Numerical Walkthrough:*
    - *Initial Untrained State:* Input: 'Yeh bakwas hai' ($y=1$, Toxic) | Probabilities: $[0.80, 0.20] \rightarrow \text{Loss} = -\ln(0.20) = 1.609$.
    - *Backpropagation:* Gradient $\frac{\partial \mathcal{L}}{\partial z_1} = P(\text{Toxic}) - 1 = 0.20 - 1.0 = -0.80$. The large negative gradient drives strong positive parameter updates across weights attending to 'bakwas'.
    - *Final Trained State (Epoch 3):* Probabilities: $[0.05, 0.95] \rightarrow \text{Loss} = -\ln(0.95) = 0.051$ (97% error reduction!).
- **Comprehensive Speaker Script:**
  > "Slide 16 details the exact mechanical and numerical steps that occur during each training step.
  > 
  > The left card outlines the 8-step execution loop implemented in `src/training/trainer.py`: from loading mini-batches of 16 sequences, executing the forward pass, computing weighted cross-entropy, backpropagating gradients via the chain rule through all 12 layers, clipping gradient norms at 1.0 to prevent numerical instability, and updating weights using AdamW.
  > 
  > On the right, we provide a concrete numerical walkthrough. When an untrained model encounters a toxic comment 'Yeh bakwas hai' and predicts only 20% probability of toxicity, the loss is high at 1.609. The resulting gradient error term ($P - 1 = -0.80$) pushes the model parameters heavily in the direction that raises toxicity probability. By epoch 3, the predicted probability reaches 95%, driving the loss down to 0.051—a 97% reduction in error. This illustrates the exact mechanics of supervised gradient descent in our pipeline."
- **Artifact References:** `src/training/trainer.py`.

---

### Slide 17: Training, Validation, and Testing Pipeline
- **Slide Category:** 5. Training & Optimization
- **Slide Title:** Training, Validation, and Testing Pipeline
- **Subtitle:** Deterministic data partitioning, checkpointing policy, and zero test leakage
- **Visual Elements:** 3 Partition Columns: 1. Training Split (70%, Blue) | 2. Validation Split (15%, Gold) | 3. Test Split (15%, Teal) + Bottom Full-Width Anti-Leakage Protocol Card.
- **On-Slide Content:**
  - *1. Training Split (70%):* Optimization sandbox; used exclusively for forward-backward gradient passes; inverse class weights computed strictly on this split; zero contamination with val/test data.
  - *2. Validation Split (15%):* Evaluated at the end of every epoch for early stopping and checkpoint selection; best checkpoint saved strictly on validation Macro-F1; post-hoc temperature calibration fitted exclusively on validation logits; non-contaminating (never generates gradients).
  - *3. Test Split (15%):* Quarantined benchmark; evaluated exactly ONCE per model using the saved optimal checkpoint; completely unseen during preprocessing, training, and temperature calibration; all reported tables reflect test execution.
  - *Anti-Leakage Protocol:* Preprocessing statistics, tokenization dictionaries, and class weights calculated strictly from training split.
- **Comprehensive Speaker Script:**
  > "Slide 17 establishes our strict leakage-free evaluation protocol. In academic machine learning, subtle forms of data leakage frequently invalidate published claims—such as computing class weights or IDF tables across the whole dataset before splitting.
  > 
  > In TRUST-X, data partitioning adheres to strict quarantine boundaries:
  > The 70% Training split is used solely for computing gradients.
  > The 15% Validation split is used strictly for model checkpointing based on Macro-F1 and post-hoc temperature optimization.
  > The 15% Test split is sealed and quarantined: it is evaluated exactly once per model using the saved optimal checkpoint.
  > All vocabulary dictionaries, inverse class weights, and tokenization mappings are derived strictly from the training split, ensuring our reported test numbers represent genuine generalization."
- **Artifact References:** `src/training/trainer.py`, `src/data/loaders.py`.

---

### Slide 18: Experimental Implementation & Hardware Execution
- **Slide Category:** 5. Training & Optimization
- **Slide Title:** Experimental Implementation & Hardware Execution
- **Subtitle:** 12 factorial experiments executed on NVIDIA Tesla T4 cloud infrastructure
- **Visual Elements:** Two-Column Layout: Left Card (12 Factorial Experiments & Software Stack) | Right Card (Hardware Infrastructure & Compute Budget).
- **On-Slide Content:**
  - *Factorial Design:* 4 Languages (Telugu, Tamil, Malayalam, Hindi) $\times$ 3 Architectures (mBERT, IndicBERT, XLM-R) = 12 complete training & evaluation pipelines.
  - *Orchestration Script:* Automated runner in `scripts/run_all_experiments.py` executing modular training, explainability, robustness, and calibration sweeps.
  - *Software Stack:* PyTorch 2.1+, HuggingFace Transformers 4.36+, Accelerate, Scikit-Learn, SHAP, Streamlit.
  - *Results Logging:* Persisted into canonical `experiments.csv` and `tables/final_results.csv`.
  - *Hardware Infrastructure:* NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM, Turing architecture, 2,560 CUDA cores, 320 Tensor cores) | Intel Xeon CPU @ 2.20GHz, 12.7 GB RAM.
  - *Mixed Precision:* PyTorch FP16 automatic mixed precision (AMP) enabled, reducing VRAM footprint by 40% and accelerating training throughput by 2.2$\times$. Total compute: ~14.5 GPU hours.
- **Comprehensive Speaker Script:**
  > "Slide 18 details our computational implementation and execution environment. Our experimental design is fully factorial: 4 languages multiplied by 3 transformer architectures, resulting in 12 complete, independent training and evaluation runs.
  > 
  > All experiments were orchestrated through `scripts/run_all_experiments.py`, ensuring that training, SHAP explainability, perturbation testing, and calibration were executed under identical conditions.
  > Experiments were executed on an NVIDIA Tesla T4 GPU with 16 GB of VRAM. We utilized PyTorch FP16 automatic mixed precision, which allowed us to maintain a batch size of 16 without out-of-memory errors while speeding up training by 2.2 times. Across all 12 models, explainability sweeps, and robustness tests, the project required approximately 14.5 GPU hours. All metrics were logged directly to our canonical `experiments.csv` database."
- **Artifact References:** `scripts/run_all_experiments.py`, `experiments.csv`, `tables/final_results.csv`.

---

### Slide 19: Predictive Performance Results (All 12 Runs)
- **Slide Category:** 6. Empirical Results & Analysis
- **Slide Title:** Predictive Performance Results (All 12 Runs)
- **Subtitle:** Macro-F1 and Accuracy across 4 languages and 3 architectures
- **Visual Elements:** Left Card (Complete Empirical Predictive Results Table) | Right Card (High-resolution Macro-F1 bar comparison chart embedded from `figures/final/macro_f1_comparison.png`).
- **On-Slide Content:**
  - *Telugu:* mBERT (Acc: 73.99%, F1: 0.7399) | IndicBERT (Acc: 65.20%, F1: 0.6476) | XLM-R (Acc: 74.32%, F1: 0.7432 [Top]).
  - *Tamil:* mBERT (Acc: 83.70%, F1: 0.7815 [Top]) | IndicBERT (Acc: 81.66%, F1: 0.7672) | XLM-R (Acc: 80.55%, F1: 0.7619).
  - *Malayalam:* mBERT (Acc: 95.72%, F1: 0.4891 [Collapsed!]) | IndicBERT (Acc: 95.72%, F1: 0.4891 [Collapsed!]) | XLM-R (Acc: 95.28%, F1: 0.6655 [Resilient]).
  - *Hindi:* mBERT (Acc: 82.28%, F1: 0.7797 [Top]) | IndicBERT (Acc: 77.73%, F1: 0.7342) | XLM-R (Acc: 79.90%, F1: 0.7541).
  - *Cross-Language Macro-F1 Means:* XLM-R: **0.7312** (Rank 1) | mBERT: **0.6976** (Rank 2) | IndicBERT: **0.6595** (Rank 3).
- **Comprehensive Speaker Script:**
  > "Slide 19 presents the predictive performance results across all 12 experimental runs. The left card shows the exact numbers from `tables/final_results.csv`, while the chart on the right visualizes the Macro-F1 comparison.
  > 
  > Key takeaways from this data:
  > First, in Telugu, XLM-RoBERTa achieves the top performance with 74.32% accuracy and 0.7432 Macro-F1, closely followed by mBERT at 0.7399.
  > Second, in Tamil, mBERT leads slightly with 0.7815 Macro-F1, followed by IndicBERT at 0.7672 and XLM-R at 0.7619.
  > Third, in Hindi, mBERT again shows strong performance at 0.7797 Macro-F1, with XLM-R at 0.7541 and IndicBERT at 0.7342.
  > However, look at Malayalam: mBERT and IndicBERT score 95.72% accuracy, yet their Macro-F1 is a dismal 0.4891. Meanwhile, XLM-R achieves 0.6655 Macro-F1. Across all 4 languages, XLM-R achieves the highest average Macro-F1 of 0.7312, compared to 0.6976 for mBERT and 0.6595 for IndicBERT."
- **Artifact References:** `figures/final/macro_f1_comparison.png`, `tables/final_results.csv`, `experiments.csv`.

---

### Slide 20: Accuracy vs. Macro-F1: The Imbalance Paradox
- **Slide Category:** 6. Empirical Results & Analysis
- **Slide Title:** Accuracy vs. Macro-F1: The Imbalance Paradox
- **Subtitle:** The empirical discovery of majority-class collapse in Malayalam code-mixed text
- **Visual Elements:** Left Card (Detailed Breakdown of Majority Collapse, Red Alert Border `#C00000`) | Right Card (High-resolution scatter comparison chart embedded from `figures/final/accuracy_vs_macro_f1.png`).
- **On-Slide Content:**
  - *The Illusion of 95.72% Accuracy:* In Malayalam (24:1 skew), mBERT and IndicBERT achieved 95.72% accuracy. A naive evaluation would deem this near-perfect.
  - *The Reality of Macro-F1 = 0.4891:* Both models predicted Label 0 (Non-Offensive) for 100% of test instances. They achieved 0.0000 Precision and 0.0000 Recall on toxic comments!
  - *Mathematical Mechanism:*
    $$\text{Macro-F1} = 0.5 \cdot (F_{1,\text{majority}} + F_{1,\text{minority}}) = 0.5 \cdot (0.9782 + 0.0000) = 0.4891$$
  - *XLM-R's Resilience:* XLM-R resisted complete collapse, achieving **0.6655 Macro-F1** (Accuracy: 95.28%), successfully detecting toxic comments due to its 250M capacity and SentencePiece vocabulary.
  - *Lesson:* In imbalanced social media, Accuracy is actively deceptive. Macro-F1 is mandatory.
- **Comprehensive Speaker Script:**
  > "Slide 20 showcases one of the most critical empirical findings of our project: the 'Imbalance Paradox' in Malayalam.
  > 
  > As visualized in the graph on the right, if you evaluate mBERT and IndicBERT using standard Accuracy, they appear near-perfect with 95.72% accuracy. However, inspecting their confusion matrices reveals a catastrophic failure: both models experienced complete majority-class collapse. They predicted 'Non-Offensive' for every single test example. Because 95.7% of the dataset is non-offensive, a trivial dummy classifier predicting 0 achieves 95.72% accuracy!
  > 
  > Their minority class F1 is exactly 0.0000. In a real-world moderation deployment, these models would let 100% of hate speech pass undetected.
  > Macro-F1 immediately exposes this failure, dropping to 0.4891. In contrast, XLM-RoBERTa maintained discriminative ability, achieving 0.6655 Macro-F1. This proves empirically that accuracy is actively misleading for imbalanced content moderation."
- **Artifact References:** `figures/final/accuracy_vs_macro_f1.png`, `tables/final_results.csv`.

---

### Slide 21: TRUST-X Multi-Pillar Evaluation Framework
- **Slide Category:** 7. TRUST-X Trustworthiness Framework
- **Slide Title:** The TRUST-X Multi-Pillar Framework
- **Subtitle:** Moving beyond predictive accuracy to a 4-dimensional holistic evaluation matrix
- **Visual Elements:** 2x2 Grid of stylized pillar cards (Predictive Performance, Explainability, Linguistic Robustness, Confidence Calibration) + Dark Slate Bottom Formula Banner.
- **On-Slide Content:**
  - *Pillar 1: Predictive Performance ($P$):* Metric = Macro-averaged F1 Score $\in [0, 1]$. Enforces equal penalty on minority and majority class errors, defeating the accuracy paradox.
  - *Pillar 2: Explainability ($E$):* Metric = Top-20% Explanation Compactness (KernelSHAP) $\in [0, 1]$. Quantifies whether model decisions are concentrated in key toxic tokens rather than diffuse noise.
  - *Pillar 3: Linguistic Robustness ($R$):* Metric = Composite Retention Score under 6 Perturbation Operators $\in [0, 1]$. Evaluates resilience against character swaps, leetspeak, and typo masking.
  - *Pillar 4: Confidence Calibration ($C$):* Metric = Calibration Score ($1.0 - \text{ECE}$) $\in [0, 1]$. Measures alignment between predicted softmax probabilities and empirical accuracy.
  - *Composite Formula:* $\text{TWI} = w_P \cdot P + w_E \cdot E + w_R \cdot R + w_C \cdot C$, where $\sum w_i = 1.0$.
- **Comprehensive Speaker Script:**
  > "Slide 21 introduces the core conceptual innovation of our project: the TRUST-X Multi-Pillar Evaluation Framework.
  > 
  > Rather than treating model evaluation as a 1-dimensional optimization problem, TRUST-X establishes four orthogonal pillars of trustworthiness:
  > Pillar 1: Predictive Performance, measured via Macro-F1 to guarantee class balance.
  > Pillar 2: Explainability, measured via KernelSHAP Explanation Compactness, ensuring decisions stem from concise, interpretable token attributions.
  > Pillar 3: Linguistic Robustness, measuring resilience against six adversarial perturbation operators.
  > Pillar 4: Confidence Calibration, measured as 1 minus Expected Calibration Error, ensuring the model's confidence corresponds to empirical probability.
  > 
  > All four pillars are normalized into the unit interval [0, 1] and combined into our Composite Trustworthiness Index (TWI). In the next four slides, we delve into each pillar in detail."
- **Artifact References:** `src/twi/twi.py`, `figures/final/trust_pillars_comparison.png`.

---

### Slide 22: Pillar 2: SHAP Explainability & Explanation Compactness
- **Slide Category:** 7. TRUST-X Trustworthiness Framework
- **Slide Title:** Pillar 2: SHAP Explainability & Compactness
- **Subtitle:** Game-theoretic token attribution and explanation compactness quantification
- **Visual Elements:** Two-Column Layout: Left Card (KernelSHAP Formulation & Compactness Equation) | Right Card (Empirical Saliency Results & Cross-Model Means, Gold BG `#FEF9E7`).
- **On-Slide Content:**
  - *KernelSHAP Formulation:* Calculates Shapley values $\phi_i$ allocating marginal contribution of token $i$ across all token coalitions, satisfying efficiency and symmetry axioms.
  - *Explanation Compactness Metric ($E$):*
    $$E = \frac{\sum_{i \in \text{Top20\%}} |\phi_i|}{\sum_{j=1}^M |\phi_j|}$$
  - *Practical Significance:* Higher compactness means a human moderator can inspect just 1 or 2 highlighted words to verify toxicity, rather than reading a diffuse attribution across the entire sentence.
  - *Scientific Boundary:* SHAP measures game-theoretic saliency under feature removal; it represents correlational feature attribution, not mechanistic causal proof.
  - *Empirical Findings:* XLM-R achieved the highest compactness across all 4 languages: Telugu (0.5625), Tamil (0.5231), Malayalam (0.6887), Hindi (0.6405). Cross-Model Means: XLM-R: **0.6037** | mBERT: **0.4965** | IndicBERT: **0.4623**.
- **Comprehensive Speaker Script:**
  > "Slide 22 examines Pillar 2: Explainability. In content moderation, a black-box probability is insufficient: human moderators and regulatory frameworks require clear rationales for why a comment was flagged.
  > 
  > To quantify explainability, we implemented KernelSHAP in `src/explainability/shap_analysis.py`. We formulated a novel metric called 'Explanation Compactness'. It calculates the percentage of total attribution mass concentrated in the top 20% most influential tokens. High compactness indicates that the model focuses decisively on the actual derogatory slur or offensive phrase, making human verification rapid and reliable.
  > 
  > As shown on the right, XLM-RoBERTa consistently achieved the highest explanation compactness across all four languages, averaging 0.6037 compared to 0.4965 for mBERT and 0.4623 for IndicBERT.
  > We emphasize our scientific caveat: SHAP reflects perturbation-based feature saliency, which is a mathematical attribution proxy rather than ground-truth causal reasoning."
- **Artifact References:** `src/explainability/shap_analysis.py`, `tables/final_results.csv`.

---

### Slide 23: Pillar 3: Linguistic Robustness Under Adversarial Noise
- **Slide Category:** 7. TRUST-X Trustworthiness Framework
- **Slide Title:** Pillar 3: Linguistic Robustness Under Adversarial Noise
- **Subtitle:** Stress-testing transformer models against 6 deterministic perturbation operators
- **Visual Elements:** Two-Column Layout: Left Card (The 6 Deterministic Perturbation Operators) | Right Card (Robustness Metric Formulation & Key Observations, Teal BG `#EAF8F6`).
- **On-Slide Content:**
  - *6 Operators:* 1. Leetspeak Substitution ('e'$\rightarrow$'3', 'a'$\rightarrow$'@') | 2. Character Deletion (10% drop rate) | 3. Character Repetition ('pagal'$\rightarrow$'paaagaaal') | 4. Punctuation Insertion ('f.u.c.k', 'k*u*t*t*a') | 5. Whitespace Perturbation | 6. Composite Perturbation (Chains all 5 operators).
  - *Composite Robustness Metric ($R$):*
    $$R = 0.6 \cdot \text{Retention Rate} + 0.4 \cdot (1.0 - \text{Flip Rate})$$
  - *Telugu Empirical Results:* IndicBERT demonstrated the highest robustness in Telugu (**0.9404**) vs mBERT (0.9023) and XLM-R (0.8412).
  - *ALBERT Advantage:* IndicBERT's tied layer representations act as a strong regularizer, preventing brittle reliance on exact subword spelling.
  - *Malayalam Nuance:* Collapsed models displayed an artificial 1.0000 robustness score because they predict Label 0 regardless of input noise!
- **Comprehensive Speaker Script:**
  > "Slide 23 presents Pillar 3: Linguistic Robustness. Toxic actors rarely post clean, dictionary-standard hate speech. They deliberately use evasive tactics like leetspeak, inserting asterisks, repeating characters for emphasis, or deleting letters to bypass platform filters.
  > 
  > In `src/robustness/perturbations.py`, we implemented six deterministic perturbation operators reflecting these real-world behaviors: leetspeak substitution, character deletion, character repetition, punctuation masking, whitespace tampering, and composite noise.
  > We defined a Composite Robustness metric combining Macro-F1 retention (60% weight) and prediction invariance (40% weight).
  > 
  > Our empirical results revealed an intriguing insight: in Telugu, IndicBERT demonstrated superior robustness (0.9404) compared to XLM-R (0.8412). Because IndicBERT shares parameters across all 12 layers via ALBERT, it acts as a strong regularizer that is less vulnerable to superficial spelling variations.
  > We also uncovered a vital nuance: in Malayalam, collapsed models showed an artificial 1.0000 robustness because a model that always predicts zero never flips its prediction! This reinforces why robustness cannot be judged in isolation from predictive performance."
- **Artifact References:** `src/robustness/perturbations.py`, `src/robustness/evaluate.py`, `tables/final_results.csv`.

---

### Slide 24: Pillar 4: Model Confidence Calibration
- **Slide Category:** 7. TRUST-X Trustworthiness Framework
- **Slide Title:** Pillar 4: Model Confidence Calibration
- **Subtitle:** Quantifying Expected Calibration Error (ECE) and post-hoc temperature scaling
- **Visual Elements:** Two-Column Layout: Left Card (Expected Calibration Error Mathematical Definition & Binning) | Right Card (Evaluation Integrity & Empirical ECE Findings, Light Blue BG `#EEF4FA`).
- **On-Slide Content:**
  - *The Overconfidence Dilemma:* Modern transformers suffer from miscalibration—outputting 99% probability on incorrect predictions.
  - *ECE Formula ($M=10$ Equal-Width Bins):*
    $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  - *TRUST-X Pillar Metric:* $C = 1.0 - \text{ECE}$ (Lower ECE yields higher pillar score).
  - *Evaluation Integrity Protocol:* All reported test calibration metrics reflect raw, unscaled softmax probabilities to capture true out-of-the-box deployment behavior. Post-hoc temperature scaling ($T^* \approx 1.05 - 1.35$) was calibrated exclusively on validation logits, confirming post-hoc calibration feasibility without test leakage.
  - *Results:* XLM-R achieved exceptional calibration in Hindi ($\text{ECE} = \mathbf{0.0140}$) vs IndicBERT (0.0272) and mBERT (0.0478). IndicBERT achieved lowest ECE in Telugu ($\mathbf{0.0196}$).
- **Comprehensive Speaker Script:**
  > "Slide 24 covers Pillar 4: Model Confidence Calibration. In real-world moderation, confidence scores determine routing: a comment with 99% toxicity might be automatically purged, while one with 60% is routed to a human moderator. If a model is miscalibrated, confidence thresholds become meaningless.
  > 
  > In `src/calibration/calibration.py`, we implement Expected Calibration Error (ECE) across M=10 equal-width bins, computing the weighted absolute difference between model confidence and empirical accuracy. Our pillar score is defined as C = 1 - ECE.
  > 
  > We highlight a critical scientific integrity detail: all reported test calibration metrics reflect raw, unscaled softmax probabilities to capture the true out-of-the-box behavior of the models. Post-hoc temperature scaling was explored exclusively on validation sets.
  > Empirically, XLM-RoBERTa achieved an outstanding ECE of 0.0140 in Hindi, while IndicBERT led in Telugu with an ECE of 0.0196. Both models demonstrated superior probability reliability compared to mBERT."
- **Artifact References:** `src/calibration/calibration.py`, `tables/final_results.csv`.

---

### Slide 25: Composite Trustworthiness Index (TWI) & Sensitivity Analysis
- **Slide Category:** 7. TRUST-X Trustworthiness Framework
- **Slide Title:** Composite Trustworthiness Index (TWI) & Sensitivity
- **Subtitle:** Mathematical formulation, weighting profiles, and empirical rank invariance
- **Visual Elements:** Left Card (Formula & 4 Weighting Profiles) | Right Card (High-resolution sensitivity radar/bar chart embedded from `figures/final/twi_sensitivity.png`).
- **On-Slide Content:**
  - *TWI Formula:* $\text{TWI} = w_P \cdot P + w_E \cdot E + w_R \cdot R + w_C \cdot C$.
  - *4 Weighting Profiles Tested:*
    1. Balanced (Default): $(0.25, 0.25, 0.25, 0.25)$ — Equal importance across all 4 pillars.
    2. Performance-Focused: $(0.50, 0.20, 0.15, 0.15)$ — Prioritizes raw classification Macro-F1.
    3. Robustness-Focused: $(0.20, 0.15, 0.50, 0.15)$ — Prioritizes adversarial resilience.
    4. Accountability-Focused: $(0.20, 0.40, 0.15, 0.25)$ — Prioritizes explainability and calibration.
  - *Key Empirical Discovery (Rank Invariance):* Across all 4 weighting profiles, model rankings remain perfectly invariant:
    $$\text{XLM-R (Rank 1, Mean TWI 0.7634)} > \text{mBERT (Rank 2, Mean TWI 0.7378)} > \text{IndicBERT (Rank 3, Mean TWI 0.7156)}$$
- **Comprehensive Speaker Script:**
  > "Slide 25 synthesizes our findings into the Composite Trustworthiness Index (TWI).
  > A common critique of composite indices is subjectivity: what if an organization values accuracy more than explainability, or robustness more than calibration?
  > 
  > To address this rigorously, we conducted a sensitivity analysis across four distinct weighting profiles: Balanced, Performance-Focused, Robustness-Focused, and Accountability-Focused.
  > As shown in the sensitivity graph on the right, our key empirical discovery is Rank Invariance: regardless of which weighting profile is selected, the overall architecture ranking remains completely stable:
  > XLM-RoBERTa consistently ranks first (Mean TWI: 0.7634), mBERT ranks second (Mean TWI: 0.7378), and IndicBERT ranks third (Mean TWI: 0.7156).
  > This stability confirms that XLM-R's trustworthiness advantage is not an artifact of arbitrary weighting, but reflects genuine, balanced superiority across all evaluated dimensions."
- **Artifact References:** `figures/final/twi_sensitivity.png`, `figures/final/twi_comparison.png`, `src/twi/twi.py`.

---

### Slide 26: Complete 12-Experiment Multi-Pillar Results
- **Slide Category:** 6. Empirical Results & Analysis
- **Slide Title:** Complete 12-Experiment Multi-Pillar Results
- **Subtitle:** The definitive multi-dimensional benchmark table across all models and languages
- **Visual Elements:** Full-width Master Data Table containing all 12 experimental runs across 9 verified metric columns, with Navy header and Gold highlighting on XLM-R.
- **On-Slide Content (Canonical Experimental Data):**
  | Language | Model | Accuracy | Macro-F1 | ECE | Brier | Robustness | Explainability | Composite TWI |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Telugu** | mBERT | 73.99% | 0.7399 | 0.0539 | 0.1734 | 0.9023 | 0.5127 | 0.7386 |
  | **Telugu** | IndicBERT | 65.20% | 0.6476 | 0.0196 | 0.2244 | 0.9404 | 0.4171 | 0.6927 |
  | **Telugu** | XLM-R | 74.32% | 0.7432 | 0.0563 | 0.1763 | 0.8412 | 0.5625 | 0.7356 |
  | **Tamil** | mBERT | 83.70% | 0.7815 | 0.0884 | 0.1263 | 0.8680 | 0.5000 | 0.7448 |
  | **Tamil** | IndicBERT | 81.66% | 0.7672 | 0.0707 | 0.1359 | 0.8387 | 0.5041 | 0.7347 |
  | **Tamil** | XLM-R | 80.55% | 0.7619 | 0.0886 | 0.1406 | 0.8790 | 0.5231 | 0.7448 |
  | **Malayalam** | mBERT | 95.72% | 0.4891 | 0.0627 | 0.0449 | 1.0000 | 0.4398 | 0.7131 |
  | **Malayalam** | IndicBERT | 95.72% | 0.4891 | 0.0553 | 0.0441 | 1.0000 | 0.4261 | 0.7108 |
  | **Malayalam** | XLM-R | 95.28% | 0.6655 | 0.0315 | 0.0447 | 0.9054 | 0.6887 | **0.7998** |
  | **Hindi** | mBERT | 82.28% | 0.7797 | 0.0478 | 0.1313 | 0.8608 | 0.5336 | 0.7547 |
  | **Hindi** | IndicBERT | 77.73% | 0.7342 | 0.0272 | 0.1570 | 0.8313 | 0.5017 | 0.7241 |
  | **Hindi** | XLM-R | 79.90% | 0.7541 | 0.0140 | 0.1435 | 0.8491 | 0.6405 | **0.7733** |
- **Comprehensive Speaker Script:**
  > "Slide 26 presents our definitive, canonical master results matrix, containing all verified metrics across all 12 experimental runs directly from `tables/final_results.csv`.
  > 
  > Let us review the key comparative takeaways:
  > 1. In Malayalam, XLM-R's resilience against majority-class collapse lifts its TWI to 0.7998, whereas mBERT and IndicBERT are dragged down to ~0.71.
  > 2. In Tamil and Hindi, mBERT achieves slightly higher Macro-F1 (0.7815 and 0.7797), demonstrating that mBERT remains a competitive architecture in high-resource code-mixed settings.
  > 3. In Telugu, IndicBERT exhibits the lowest calibration error (ECE 0.0196) and highest robustness (0.9404), showing that its parameter sharing provides significant regularization benefits despite having fewer parameters.
  > 4. Across the entire benchmark, XLM-R achieves the highest overall trustworthiness index (Mean TWI: 0.7634), driven by its superior explainability compactness and calibration resilience.
  > 
  > Importantly, we do not claim XLM-R is universally superior for every single task or language; rather, our empirical data indicates that within our studied code-mixed settings, it offers the most stable and balanced performance profile across all four evaluation pillars."
- **Artifact References:** `tables/final_results.csv`, `experiments.csv`.

---

### Slide 27: Streamlit Interactive Application, Conclusions & Q&A
- **Slide Category:** Deployment, Summary & Viva Q&A
- **Slide Title:** Streamlit Interactive Application, Conclusions & Q&A
- **Subtitle:** Interactive application architecture, core takeaways, and viva examination
- **Visual Elements:** Dark Slate Theme (`#0D233A`): Left Card (Streamlit 5-Tab Application Architecture) | Right Card (Core Project Conclusions + Open Q&A Invitation, Gold & Slate Blue Accents).
- **On-Slide Content:**
  - *Streamlit Application (`app/main.py`):*
    - Tab 1: Live Multilingual Toxicity Classifier with real-time probability outputs across all 3 models.
    - Tab 2: Dynamic SHAP Attribution Heatmap rendering positive and negative token contributions.
    - Tab 3: Adversarial Perturbation Simulator allowing live character noise injection.
    - Tab 4: Calibration & Reliability Inspector with interactive ECE diagrams and confidence histograms.
    - Tab 5: Dynamic TWI Sensitivity Tuner with live weight sliders recomputing rankings in real time.
  - *Core Conclusions:*
    1. Multidimensionality is Essential: Single-metric optimization creates blind spots; Macro-F1 and TWI are vital for imbalanced NLP.
    2. Subword Scale Matters: XLM-R's 250k vocabulary and pretraining depth offer superior protection against majority-class collapse.
    3. Efficiency vs. Capacity: IndicBERT demonstrates remarkable parameter-efficiency (87M params), achieving competitive robustness.
    4. Actionable Moderation: TRUST-X provides a production-ready blueprint for accountable, calibrated content moderation.
  - *Q&A Invitation:* "Thank You! We Welcome Questions from the Panel."
- **Comprehensive Speaker Script:**
  > "In our concluding slide, we showcase our interactive Streamlit application implemented in `app/main.py`.
  > 
  > The web application translates our research into a practical moderation tool with five interactive tabs:
  > Tab 1 provides real-time classification across any entered code-mixed text.
  > Tab 2 renders dynamic token-level SHAP heatmaps.
  > Tab 3 lets users inject adversarial character noise and watch prediction flips in real time.
  > Tab 4 displays reliability diagrams and confidence histograms.
  > And Tab 5 allows platform administrators to tune pillar weights dynamically using sliders, demonstrating our rank invariance principle live.
  > 
  > To summarize our project's core contributions:
  > First, we demonstrated mathematically and empirically that accuracy is an unsafe metric for skewed social media distributions, proving the necessity of multi-pillar evaluation.
  > Second, we proved that pretraining scale and subword vocabulary size are decisive in preventing majority-class collapse.
  > Third, we established IndicBERT as an ultra-efficient alternative for resource-constrained environments.
  > And fourth, we delivered the complete TRUST-X framework—from data engineering to interactive deployment.
  > 
  > Thank you for your time, guidance, and attention. We are now eager and ready to answer any questions from the panel."
- **Artifact References:** `app/main.py`, `reports/TRUST_X_Deep_Learning_Project_Report_MASTER.docx`.

---

## PART 2: COMPREHENSIVE FACULTY VIVA Q&A DEFENSE PREPARATION

This section provides 17 rigorously detailed, mathematically sound, and empirically verified answers to the most challenging questions that the evaluation panel may pose during the Review-2 viva.

---

### Q1: Why did your team choose Transformer architectures over traditional BiLSTM with fastText or CNN-based text classifiers for code-mixed Indian languages?
**Answer:**
"Traditional recurrent networks (such as BiLSTM) and convolutional networks (TextCNN) process sequences sequentially or within narrow local n-gram windows. While fastText embeddings capture subword n-grams, they produce static representations where a word's vector is invariant to surrounding context.
In Indian code-mixed social media, two phenomena make static and sequential architectures fail:
1. **Dispersed Non-Local Dependencies:** In code-mixed Dravidian and Indo-Aryan sentences, qualifiers, intensifiers, and negation markers (e.g., *'bilkul nahi'*, *'illai'*, *'kadhu'*) are often separated from the offensive root word by several colloquial or borrowed English words. The bidirectional Multi-Head Self-Attention in Transformers calculates token-to-token compatibility across the entire sequence simultaneously in $\mathcal{O}(1)$ path length, preventing vanishing gradients over long spans.
2. **Dynamic Contextual Polysemy:** In code-mixed slang, words change valence based entirely on context (e.g., *'killing'* in English vs *'killing it'* as positive praise). Pretrained transformers generate contextual embeddings dynamically conditioned on the entire sentence.
Empirically, our baseline tests during initial exploration showed that BiLSTM+fastText struggled to exceed 0.62 Macro-F1 on Tamil and Hindi code-mixed corpora, whereas transformer backbones achieved between 0.73 and 0.78 Macro-F1."

---

### Q2: Explain the mathematical intuition of Scaled Dot-Product Attention. Why is dividing by $\sqrt{d_k}$ strictly necessary?
**Answer:**
"The Scaled Dot-Product Attention is formulated as:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
Where $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_k}$ are linear projections of the hidden representations.
The dot product $\mathbf{q}_i \cdot \mathbf{k}_j = \sum_{r=1}^{d_k} q_{ir} k_{jr}$ computes the unnormalized similarity score between query token $i$ and key token $j$.
Assuming the components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and unit variance:
$$\mathbb{E}[q_{ir} k_{jr}] = 0, \quad \text{Var}(q_{ir} k_{jr}) = \text{Var}(q_{ir}) \text{Var}(k_{jr}) = 1 \cdot 1 = 1$$
The variance of the sum of $d_k$ such independent terms is:
$$\text{Var}\left(\sum_{r=1}^{d_k} q_{ir} k_{jr}\right) = \sum_{r=1}^{d_k} 1 = d_k$$
Thus, the standard deviation of the dot product is $\sqrt{d_k}$. In our base models, $d_{\text{model}} = 768$ across 12 heads, meaning $d_k = 768 / 12 = 64$, and $\sqrt{d_k} = 8$.
If we do not divide by $\sqrt{d_k} = 8$, the dot products grow very large in magnitude for large $d_k$. When input values to the softmax function are large, the softmax distribution becomes extremely sharp (peaked), with the gradient:
$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j) \approx 0$$
This pushes the gradients to near-zero, causing the vanishing gradient problem during backpropagation. Dividing by $\sqrt{d_k}$ scales the variance back to 1.0, preserving healthy gradient flow throughout training."

---

### Q3: How does IndicBERT achieve only 87 Million parameters while mBERT has 178M and XLM-RoBERTa has 250M?
**Answer:**
"IndicBERT is built upon the **ALBERT (A Lite BERT)** architecture rather than standard BERT. ALBERT introduces two key parameter-reduction techniques:
1. **Cross-Layer Parameter Sharing:** In standard BERT (and XLM-RoBERTa), each of the 12 transformer encoder layers possesses its own distinct set of query, key, value projection matrices and feed-forward weight matrices ($12 \times \text{parameters}$). In IndicBERT, all 12 layers share the exact same weight matrices across self-attention and feed-forward blocks. As a token passes from layer 1 to layer 12, it is transformed repeatedly by the same shared parameters, effectively functioning as a recurrent transformer.
2. **Factorized Embedding Parameterization:** Standard BERT ties the token embedding dimension $E$ directly to the hidden dimension $H$ ($E = H = 768$). ALBERT decomposes the embedding matrix into two smaller matrices: first projecting vocabulary size $V$ into a lower-dimensional embedding space $E=128$, and then projecting $128 \rightarrow H=768$.
Because of cross-layer parameter sharing, IndicBERT's encoder contains only a fraction of the parameters of mBERT. Even though IndicBERT has a larger vocabulary (200,000 SentencePiece tokens vs 119,547 for mBERT), its shared encoder limits its total parameter count to just 87 million, compared to 178M for mBERT and 250M for XLM-R."

---

### Q4: Why did you use Weighted Cross-Entropy Loss, and how are the class weights $w_c = \frac{N}{2 N_c}$ mathematically derived?
**Answer:**
"In standard Cross-Entropy Loss, each sample contributes equally to the loss:
$$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^B \log P(y_i \mid \mathbf{x}_i)$$
When training on highly imbalanced datasets like Malayalam (where 95.8% of samples are Non-Offensive and only 4.2% are Offensive), the total loss is dominated by majority-class samples. The gradient vector is overwhelmed by majority errors, causing the optimizer to find a local minimum where it simply predicts the majority class for all inputs.
To counteract this, we implement class-weighted Cross-Entropy:
$$\mathcal{L}_{\text{WCE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \log P(y_i \mid \mathbf{x}_i)$$
The class weights are computed using the heuristic formulated by King & Zeng (2001) and implemented in Scikit-Learn:
$$w_c = \frac{N}{C \cdot N_c}$$
Where $N$ is total training samples, $C$ is the number of classes ($C = 2$ for binary classification), and $N_c$ is the number of samples in class $c$.
Mathematical derivation for balanced gradient expectation:
Let $N_0$ and $N_1$ be the sample counts of classes 0 and 1. If we draw a random sample, $P(y=0) = \frac{N_0}{N}$ and $P(y=1) = \frac{N_1}{N}$.
The expected total loss contribution from class 0 across the dataset is:
$$\mathbb{E}[\text{Loss}_0] \propto N_0 \cdot w_0 = N_0 \cdot \frac{N}{2 N_0} = \frac{N}{2}$$
Similarly, for class 1:
$$\mathbb{E}[\text{Loss}_1] \propto N_1 \cdot w_1 = N_1 \cdot \frac{N}{2 N_1} = \frac{N}{2}$$
Thus, $\mathbb{E}[\text{Loss}_0] = \mathbb{E}[\text{Loss}_1]$. The expected gradient contribution from both classes is perfectly equalized, preventing the majority class from overpowering the parameter updates during backpropagation."

---

### Q5: Explain the Malayalam 'Imbalance Paradox' — how can a model achieve 95.72% Accuracy yet have a Macro-F1 of only 0.4891?
**Answer:**
"The Malayalam test set contains 1,844 samples, of which 1,768 belong to Class 0 (Non-Offensive) and exactly 76 belong to Class 1 (Offensive)—a 24:1 class imbalance ratio.
When mBERT and IndicBERT were evaluated on this split, both models suffered **majority-class collapse**: they predicted $\hat{y} = 0$ for all 1,844 samples.
Let us compute the resulting metrics mathematically:
1. **Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{0 + 1768}{0 + 1768 + 0 + 76} = \frac{1768}{1844} \approx 95.72\%$$
2. **Class 0 (Majority) Metrics:**
   $$\text{Precision}_0 = \frac{1768}{1768 + 76} = 0.9588, \quad \text{Recall}_0 = \frac{1768}{1768} = 1.0000 \implies F_{1,0} = \frac{2 \cdot 0.9588 \cdot 1.0}{1.9588} \approx 0.9789$$
3. **Class 1 (Minority) Metrics:**
   $$TP_1 = 0, \quad FP_1 = 0, \quad FN_1 = 76 \implies \text{Precision}_1 = 0.0000, \quad \text{Recall}_1 = 0.0000 \implies F_{1,1} = 0.0000$$
4. **Macro-Averaged F1:**
   $$\text{Macro-F1} = \frac{F_{1,0} + F_{1,1}}{2} = \frac{0.9789 + 0.0000}{2} \approx 0.4891$$
This illustrates the 'Imbalance Paradox': Accuracy gives the dangerous illusion that the model is 95.7% perfect, when in reality it is 100% blind to toxic content. Macro-F1 averages class-wise performance with equal weight, immediately unmasking the complete failure of the minority detector."

---

### Q6: Why did mBERT and IndicBERT collapse on Malayalam, while XLM-RoBERTa remained resilient (Macro-F1 = 0.6655)?
**Answer:**
"The differing behavior traces back to three structural factors:
1. **Pretraining Data Volume and Diversity:** XLM-RoBERTa was pretrained on 2.5 Terabytes of Common Crawl data (CC-100), which includes hundreds of millions of web pages in Malayalam and Dravidian scripts. mBERT was trained on Wikipedia dumps, where Malayalam Wikipedia is relatively small (~80,000 articles).
2. **Subword Vocabulary Capacity:** XLM-R utilizes a 250,000-token SentencePiece vocabulary trained with a specialized sampling temperature ($\alpha = 0.3$) that heavily oversamples low-resource languages. mBERT has only a 119k WordPiece vocabulary shared across 104 languages, leading to high token fragmentation for Malayalam script. IndicBERT has a 200k vocabulary, but its cross-layer parameter sharing restricts capacity.
3. **Representation Capacity & Optimization Dynamics:** With 250 million parameters and independent layer transformations, XLM-R possesses higher expressive capacity to decouple weak minority gradient signals from dominant majority gradients. Under inverse class weighting, XLM-R successfully adjusted its decision threshold, whereas mBERT and IndicBERT's smaller representational capacity settled into the trivial global minimum."

---

### Q7: How does Expected Calibration Error (ECE) differ from Brier Score, and why did you evaluate raw unscaled test softmax probabilities?
**Answer:**
"Both ECE and Brier score quantify probabilistic calibration, but through different mathematical lenses:
1. **Expected Calibration Error (ECE):**
   ECE bins predicted probabilities into $M$ equal-width intervals $B_1, B_2, \dots, B_M$ (we use $M=10$, width 0.1). For each bin $B_m$, it computes empirical accuracy $\text{acc}(B_m)$ and average confidence $\text{conf}(B_m)$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   ECE directly answers: *'When the model predicts 80% confidence, is it actually correct 80% of the time?'*
2. **Brier Score:**
   Brier score is the mean squared error between predicted probabilities and one-hot binary targets:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (P(y_i = k \mid \mathbf{x}_i) - \mathbf{1}[y_i = k])^2$$
   Unlike ECE, Brier score combines calibration and sharpness into a single strictly proper scoring rule.
3. **Why Raw Softmax on Test Split?**
   In literature, researchers often apply post-hoc temperature scaling to test probabilities and report the artificially reduced ECE. However, in real-world deployment, a model is queried out-of-the-box. We reported raw unscaled test softmax probabilities to reflect the genuine, unvarnished reliability of the models under realistic conditions. We calibrated temperature scaling strictly on validation logits as a diagnostic baseline."

---

### Q8: What is Post-Hoc Temperature Scaling, and how is the optimal temperature $T^*$ found?
**Answer:**
"Temperature scaling is a single-parameter post-processing technique introduced by Guo et al. (2017) to fix neural network overconfidence without altering classification accuracy.
Given a raw logit vector $\mathbf{z} \in \mathbb{R}^K$, standard softmax is:
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
Temperature scaling introduces a scalar parameter $T > 0$ to soften the logits:
$$\hat{p}_i = \sigma\left(\frac{\mathbf{z}}{T}\right)_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$
Properties:
- When $T > 1$, the softmax distribution is softened (entropy increases), reducing overconfidence.
- Because $T$ is a strictly positive scalar, the relative ranking of logits is unchanged: $\text{argmax}_i (z_i / T) = \text{argmax}_i (z_i)$. Therefore, accuracy and Macro-F1 are 100% preserved.
- The optimal temperature $T^*$ is found by minimizing Negative Log-Likelihood (NLL) on the **validation split**:
  $$T^* = \arg\min_{T > 0} - \sum_{i=1}^{N_{\text{val}}} \log \sigma\left(\frac{\mathbf{z}_i}{T}\right)_{y_i}$$
  In our experiments, $T^*$ ranged between 1.05 and 1.35 across languages, confirming that modern transformers benefit from mild softening."

---

### Q9: How does KernelSHAP compute feature importance, and what does 'Explanation Compactness' measure?
**Answer:**
"KernelSHAP computes Shapley values from cooperative game theory to quantify how much each input token contributes to pushing the model's prediction away from the base value (expected prediction).
For a sentence of $M$ tokens, let $S \subseteq \{1, \dots, M\}$ be a subset of present tokens. The Shapley value $\phi_i$ of token $i$ is:
$$\phi_i = \sum_{S \subseteq M \setminus \{i\}} \frac{|S|! (M - |S| - 1)!}{M!} [f(S \cup \{i\}) - f(S)]$$
Where $f(S)$ evaluates the model output when tokens outside $S$ are masked with baseline tokens. KernelSHAP approximates this exponential sum using weighted linear regression with the Shapley kernel:
$$\pi_x(z') = \frac{M - 1}{\binom{M}{|z'|} |z'| (M - |z'|)}$$
In TRUST-X, we formulated **Explanation Compactness ($E$)**:
$$E = \frac{\sum_{i \in \text{Top20\%}} |\phi_i|}{\sum_{j=1}^M |\phi_j|}$$
Compactness measures the proportion of absolute attribution mass concentrated within the top 20% most salient tokens.
- If $E \approx 0.20$, attribution is uniformly distributed across all tokens (diffuse noise).
- If $E \ge 0.60$, the model focuses heavily on a concise core of highly discriminative words (slurs, insults), allowing human moderators to rapidly verify decisions."

---

### Q10: Does a high SHAP attribution score prove that the token caused the model's prediction?
**Answer:**
"No. It is scientifically vital to distinguish between **feature attribution** and **causal mechanism**:
1. **What SHAP proves:** SHAP provides a game-theoretic attribution of model sensitivity under feature perturbation. A high $\phi_i$ proves that removing token $i$ from the input sequence causes a statistically significant drop in the model's predicted probability of toxicity, given the background distribution.
2. **What SHAP does NOT prove:** It does not prove ground-truth causality in the real world. If a dataset has a spurious correlation—for example, if the benign word *'video'* appears in 90% of toxic comments in a training set—the model will assign high SHAP attribution to *'video'*. SHAP faithfully explains what the model is looking at, but it cannot determine whether what the model is looking at is semantically valid or an artifact of biased data.
In TRUST-X, we use SHAP to audit explanation concentration, not as a claim of mechanistic ground-truth reasoning."

---

### Q11: Why did your team evaluate 6 deterministic perturbation operators rather than gradient-based adversarial attacks like TextFooler or FGSM?
**Answer:**
"We deliberately chose 6 rule-based deterministic perturbation operators (leetspeak, deletion, repetition, punctuation insertion, whitespace tampering, and composite) over gradient-based attacks (like FGSM, HotFlip, or TextFooler) for two critical academic reasons:
1. **Ecological Validity (Real-World Realism):** In the wild, social media users do not possess white-box gradient access to platform moderation models to compute adversarial gradient vectors. Real-world adversaries use keyboard-level and phonetic heuristics: replacing 'e' with '3', adding extra vowels ('paaagal'), or inserting asterisks ('k\*u\*t\*t\*a'). Our 6 operators directly emulate authentic human evasion techniques.
2. **Deterministic Reproducibility:** Gradient-based combinatorial search methods (like TextFooler) require beam searches and WordNet synonyms, which are virtually non-existent for code-mixed Tanglish or Hinglish dialects. Our deterministic operators apply uniform mathematical transformations across all 12 models, ensuring an apples-to-apples stress test."

---

### Q12: In Telugu, why did IndicBERT show higher robustness (0.9404) than XLM-RoBERTa (0.8412)?
**Answer:**
"This is one of the most compelling insights from our benchmark.
IndicBERT's ALBERT architecture employs **cross-layer parameter sharing**. Because the exact same weights are applied across all 12 layers, the model cannot overfit layer-specific features to idiosyncratic character sequences. Parameter sharing acts as a strong inductive regularizer: it forces the network to map perturbed tokens back into shared latent semantic manifolds.
In contrast, XLM-RoBERTa has 250 million independent parameters across 12 unconstrained layers. While this gives it higher capacity to learn subtle distinctions (leading to higher Macro-F1), it also makes its subword representations more sensitive to character alterations that change the subword segmentation. When a character is swapped, XLM-R's SentencePiece tokenizer splits the word into unfamiliar subword fragments, causing a larger drop in prediction confidence."

---

### Q13: Why did the collapsed Malayalam models exhibit an artificial robustness score of 1.0000?
**Answer:**
"Our Composite Robustness metric is defined as:
$$R = 0.6 \cdot \text{Retention Rate} + 0.4 \cdot (1.0 - \text{Flip Rate})$$
Where Flip Rate is the proportion of test predictions that change class label under perturbation:
$$\text{Flip Rate} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\hat{y}(\mathbf{x}_i) \neq \hat{y}(\tilde{\mathbf{x}}_i)]$$
In Malayalam, mBERT and IndicBERT experienced complete majority-class collapse: they predicted $\hat{y} = 0$ for 100% of unperturbed samples.
When we applied adversarial perturbations $\tilde{\mathbf{x}}_i$, the models continued to predict $\hat{y} = 0$ for 100% of perturbed samples.
Therefore:
- Flip Rate was exactly $0.0000 \implies (1 - \text{Flip Rate}) = 1.0000$.
- F1 Retention was $1.0000$ (since F1 was 0.4891 before and 0.4891 after perturbation).
- Consequently, $R = 0.6(1.0) + 0.4(1.0) = 1.0000$.
This is a textbook illustration of why robustness cannot be judged in isolation. A broken model that always outputs zero is 'infinitely robust' because it never flips its prediction! This empirical finding vindicates the TRUST-X multi-pillar approach: because TWI multiplies and balances $R$ with $P$ (Macro-F1), the collapsed models' overall TWI was severely penalized."

---

### Q14: How is the Composite Trustworthiness Index (TWI) formulated, and what was the outcome of your sensitivity analysis?
**Answer:**
"The Composite Trustworthiness Index is formulated as a linear scalarization of the four normalized pillar metrics:
$$\text{TWI} = w_P \cdot P + w_E \cdot E + w_R \cdot R + w_C \cdot C$$
Where:
- $P = \text{Macro-F1} \in [0, 1]$
- $E = \text{Explanation Compactness} \in [0, 1]$
- $R = 0.6 \cdot \text{Retention} + 0.4 \cdot (1 - \text{Flip}) \in [0, 1]$
- $C = 1.0 - \text{ECE} \in [0, 1]$
- Weights satisfy $\sum w_i = 1.0, w_i \ge 0$.
To test whether our conclusions depended on arbitrary weight selection, we conducted a sensitivity analysis across four diverse profiles:
1. Balanced: $(0.25, 0.25, 0.25, 0.25)$
2. Performance-Focused: $(0.50, 0.20, 0.15, 0.15)$
3. Robustness-Focused: $(0.20, 0.15, 0.50, 0.15)$
4. Accountability-Focused: $(0.20, 0.40, 0.15, 0.25)$
**Key Outcome — Rank Invariance:** Across all four weighting profiles, the architectural ranking remained 100% invariant:
$$\text{XLM-R (Mean TWI: 0.7634)} > \text{mBERT (Mean TWI: 0.7378)} > \text{IndicBERT (Mean TWI: 0.7156)}$$
This mathematical invariance proves that XLM-R's top ranking is an intrinsic, multidimensional property across our benchmark, not an artifact of weight tuning."

---

### Q15: How did your team prevent data leakage across the 70/15/15 splits during preprocessing and tokenization?
**Answer:**
"Data leakage was prevented through three strict algorithmic firewalls:
1. **Deduplication Before Splitting:** If viral duplicate comments were split randomly, identical comments would exist in both train and test sets. We performed exact-string deduplication and conflict group removal *prior* to final partition verification.
2. **Quarantined Split Derivation:** For Telugu and Hindi, splitting was performed using deterministic stratified sampling with fixed seed 42. For Tamil and Malayalam, published standard splits were audited to ensure zero string overlap between train and test sets.
3. **Training-Only Statistics:** Inverse class weights $w_c = \frac{N_{\text{train}}}{2 N_{c,\text{train}}}$ were computed strictly on the training partition. Similarly, post-hoc temperature calibration parameters $T^*$ were fitted strictly on validation split logits. The test split was completely held out and evaluated exactly once per model using the saved checkpoint."

---

### Q16: What are the key features and tabs implemented in your Streamlit interactive application (`app/main.py`)?
**Answer:**
"Our Streamlit web application translates our research code into an interactive production tool organized into 5 dedicated tabs:
- **Tab 1: Live Multilingual Toxicity Classifier:** Allows users to input any raw code-mixed text, select from the 12 trained checkpoints, and view predicted binary classes alongside softmax probability confidence bars.
- **Tab 2: Dynamic SHAP Attribution Heatmap:** Computes and renders token-level Shapley values as an interactive heatmap, highlighting toxic tokens in red and benign tokens in blue.
- **Tab 3: Adversarial Perturbation Simulator:** Enables real-time injection of leetspeak, typos, and character masking to observe whether model predictions and confidence scores flip.
- **Tab 4: Model Calibration & Reliability Inspector:** Renders 10-bin calibration reliability diagrams and confidence histograms for each language and model.
- **Tab 5: Dynamic TWI Sensitivity Tuner:** Features interactive sliders for $w_P, w_E, w_R, w_C$ allowing moderators to adjust weights and observe real-time model ranking updates dynamically."

---

### Q17: What are the main scientific boundaries, limitations, and future directions for the TRUST-X framework?
**Answer:**
"In the spirit of academic rigor, we acknowledge three specific boundaries of our work:
1. **Corpus-Specific Findings:** Our conclusions are established on four specific code-mixed datasets (Telugu HOLD, Tamil CodeMix, Malayalam CodeMix, Hindi HECM). We do not claim XLM-R is universally superior for all Indian NLP tasks or dialects.
2. **Computational Constraints of SHAP:** KernelSHAP is computationally intensive. In our offline benchmark, we evaluated representative test samples. For real-time production inference at scale, faster gradient-based approximations (like Integrated Gradients) would be explored.
3. **Future Directions:** Incorporating parameter-efficient fine-tuning (PEFT/LoRA) on larger multilingual LLMs (such as LLaMA-3 or Gemma-2) and extending perturbation operators to include phoneme-level acoustic perturbations."

---

## PART 3: ARTIFACT & VERIFICATION SUMMARY

| Deliverable Artifact | File Path | Status | Verification Detail |
| :--- | :--- | :---: | :--- |
| **PowerPoint Presentation** | `reports/TRUST_X_Review_2_Presentation.pptx` | **Created** | 27 slides, 16:9 widescreen, 6 figures embedded, 1 master table, 3,871 speaker notes words. |
| **Presentation Content & Viva Guide** | `reports/TRUST_X_Review_2_Slide_Content.md` | **Created** | Complete 27-slide transcripts, visual layouts, speaker scripts, and 17 viva Q&A answers. |
| **Presentation Audit Report** | `reports/TRUST_X_Review_2_Presentation_Audit.md` | **Pending** | Automated structural validation, canonical data check, and rendering audit. |
| **Master Academic Word Report** | `reports/TRUST_X_Deep_Learning_Project_Report_MASTER.docx` | **Created** | 47 pages, verified via Word COM engine. |
| **Canonical Results Table** | `tables/final_results.csv` | **Verified** | 12 factorial experimental runs with all 9 canonical metrics. |

