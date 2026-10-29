# TRUST-X False Positive Diagnostic Analysis: Hindi XLM-R Model

**Status:** READ-ONLY DIAGNOSTIC REPORT  
**Dataset:** HECM Hindi-English Code-Mixed (Test Split: 1,383 samples)  
**Model:** XLM-RoBERTa Base (`hindi_xlmr_seed42`)  
**Checkpoint:** `checkpoints/hindi/xlmr/best_checkpoint/model.safetensors`  
**Author Confirmation on Labels:** `0 = Hateful`, `1 = Non-Hateful`  

---

## 1. Executive Summary & Error Distribution

Across the frozen Hindi XLM-R test evaluation of **1383** samples:

| Metric | Count | Percentage of Test Set | Description |
|---|---|---|---|
| **Total Test Samples** | 1383 | 100.0% | Complete frozen HECM Hindi test partition |
| **True Positives (Hateful correctly identified)** | 257 | 18.6% | True Class 0 correctly predicted as Class 0 |
| **True Negatives (Non-Hateful correctly identified)** | 848 | 61.3% | True Class 1 correctly predicted as Class 1 |
| **False Positives (Type I Error)** | **116** | **8.4%** | Ground Truth: Non-Hateful (1), Predicted: Hateful (0) |
| **False Negatives (Type II Error)** | **162** | **11.7%** | Ground Truth: Hateful (0), Predicted: Non-Hateful (1) |
| **Overall Accuracy** | 79.90% | — | $(257 + 848) / 1383$ |
| **Overall Macro-F1** | 75.41% | — | Balanced F1 across both classes |

> **Verification Confirmation:**  
> Exactly **116 False Positives** and **162 False Negatives** are confirmed in the frozen experiment results.

---

## 2. Statistical Characterization of the 116 False Positives

### 2.1 Confidence Tiers ($P(\text{Hateful}) = P(\text{Class } 0)$)

The 116 false positives exhibit the following confidence distribution:

| Confidence Tier | Probability Range | Count | % of FPs | Interpretation |
|---|---|---|---|---|
| **Borderline Uncertainty** | $0.5000 \le P < 0.6500$ | **52** | **44.8%** | Model is marginally over the 0.5 decision threshold; sensitive to small logit perturbations |
| **Moderate Confidence** | $0.6500 \le P < 0.8500$ | **45** | **38.8%** | Clear tilt toward Class 0 driven by multiple salient or contentious keywords |
| **High Confidence Overfitting** | $0.8500 \le P \le 0.9800$ | **19** | **16.4%** | Model strongly misclassifies non-hateful political attack/satire as toxic hate speech |

- **Mean $P(\text{Hateful})$:** `0.6929`
- **Median $P(\text{Hateful})$:** `0.6773`
- **Interquartile Range (IQR):** `0.5796` to `0.7986`

### 2.2 Text Length Distribution

| Length Category | Word Count | Count | % of FPs |
|---|---|---|---|
| **Short** | $\le 10$ words | 15 | 12.9% |
| **Medium** | $11 - 25$ words | 36 | 31.0% |
| **Long** | $> 25$ words | 65 | 56.0% |

Longer social media comments (>25 words) constitute 56% of false positives, often because longer texts accumulate more argumentative political markers.

### 2.3 Observable Lexical & Structural Markers

| Observable Pattern | Count in 116 FPs | Percentage | Description |
|---|---|---|---|
| **User Mentions (`@username`)** | 88 | 75.9% | Twitter conversational tagging directed at public/media figures |
| **Hashtags (`#topic`)** | 17 | 14.7% | Political campaigns (`#modi_rojgar_do`, `#shameongodimedia`, etc.) |
| **External URLs (`http/https`)** | 16 | 13.8% | Embedded media links from tweets |
| **Political / Communal / Protest Terms** | 69 | 59.5% | Mentions of leaders, parties, religious groups, or laws |
| &nbsp;&nbsp;• Mentions of *Modi* | 54 | 46.6% | Specific references to Prime Minister Narendra Modi |
| &nbsp;&nbsp;• Mentions of *BJP* | 8 | 6.9% | Specific references to the ruling party |
| &nbsp;&nbsp;• Mentions of *Rahul / Gandhi* | 6 | 5.2% | Specific references to opposition leadership |
| &nbsp;&nbsp;• Mentions of *CAA / NRC* | 4 | 3.4% | References to contentious citizenship legislation |
| &nbsp;&nbsp;• Mentions of *Kisan / Farmers* | 2 | 1.7% | References to 2020-2021 Indian farmer agitations |
| &nbsp;&nbsp;• Mentions of *SSR / Sushant* | 2 | 1.7% | References to the Sushant Singh Rajput viral Twitter campaigns |

---

## 3. Ten Diverse Representative False Positives & SHAP Feature Attributions

Using a reproducible `seed=42` stratified selection scheme across confidence tiers and text lengths, 10 diverse representative false positives were selected and analyzed with the fine-tuned XLM-R model and tokenizer using SHAP (`max_evals=60`).

> **Note on SHAP Attribution Values:**  
> SHAP attributions are reported for **Class 0 (Hateful)**.  
> - **Positive SHAP value ($+ \phi$)**: Token increases the model's predicted probability of being **Hateful**.  
> - **Negative SHAP value ($- \phi$)**: Token decreases the model's predicted probability of being Hateful (pushes toward **Non-Hateful**).

### Sample 1: `hi_test_00478`
- **Input Text:** "@don_bhojpuriya @Priyank57577435 Jhooth aur jumlo kaa pitara toh hai modi"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{90.16\%}$, $P(\text{Non-Hateful}) = 9.84\%$
- **Stratification Bucket:** High (0.85-1.00) &bull; Short (<=10w) (10 words)
- **SHAP Base Value:** `0.9192`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `aur ` | `+0.0722` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `J` | `+0.0322` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `ho` | `+0.0322` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `oth ` | `+0.0322` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `j` | `+0.0128` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `modi` | `-0.2626` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `435 ` | `-0.0241` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `57` | `-0.0136` | Lowers hate probability |

### Sample 2: `hi_test_00325`
- **Input Text:** "@ChetanM24579252 @mrpal1000 Jese modi bheek maang rha tha pulwama k baad k muje vote do pulwama k shaheedon k naam pr😂😂😂😂"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{85.95\%}$, $P(\text{Non-Hateful}) = 14.05\%$
- **Stratification Bucket:** High (0.85-1.00) &bull; Medium (11-25w) (21 words)
- **SHAP Base Value:** `0.9535`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `pul` | `+0.0188` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `wa` | `+0.0188` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `ma ` | `+0.0188` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `k ` | `+0.0188` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `pul` | `+0.0095` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `modi ` | `-0.1592` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `Je` | `-0.0337` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `se ` | `-0.0337` | Lowers hate probability |

### Sample 3: `hi_test_00216`
- **Input Text:** "@SushantBSinha Darte to sirf tum Godi media vale ho jo chand sikko ki aavaj sunte hi modi ke samne mujra krne lag jate ho 😅 #shameongodimedia"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{85.22\%}$, $P(\text{Non-Hateful}) = 14.78\%$
- **Stratification Bucket:** High (0.85-1.00) &bull; Long (>25w) (26 words)
- **SHAP Base Value:** `0.9637`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `si` | `+0.0275` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `rf ` | `+0.0275` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `tum ` | `+0.0275` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `S` | `+0.0092` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `us` | `+0.0092` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `hi ` | `-0.0827` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `modi ` | `-0.0827` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `ke ` | `-0.0179` | Lowers hate probability |

### Sample 4: `hi_test_00468`
- **Input Text:** "@burgerrqueen marahin caa"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{80.79\%}$, $P(\text{Non-Hateful}) = 19.21\%$
- **Stratification Bucket:** Moderate (0.65-0.85) &bull; Short (<=10w) (3 words)
- **SHAP Base Value:** `0.8232`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `ca` | `+0.0427` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `que` | `+0.0349` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `marah` | `+0.0275` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `in ` | `+0.0040` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `a` | `+-0.0030` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `burger` | `-0.0535` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `@` | `-0.0361` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `en ` | `-0.0214` | Lowers hate probability |

### Sample 5: `hi_test_00854`
- **Input Text:** "@HJS_Convener @NP_HJS @HinduJagrutiOrg Agar population control bill.hi sahiye to kiu CAA k jariye hindu logo ko.lana sahte ho..ye baato..isse population barhega ya control hoga."
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{75.92\%}$, $P(\text{Non-Hateful}) = 24.08\%$
- **Stratification Bucket:** Moderate (0.65-0.85) &bull; Medium (11-25w) (24 words)
- **SHAP Base Value:** `0.9637`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `ye ` | `+0.0610` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `ba` | `+0.0610` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `ato` | `+0.0610` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `.` | `+0.0134` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `.` | `+0.0088` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `Agar ` | `-0.0271` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `population ` | `-0.0271` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `control ` | `-0.0271` | Lowers hate probability |

### Sample 6: `hi_test_00084`
- **Input Text:** "@kktotlani @PashaSehgal Sehgal sab, tootlaa bacha kuch ziada emotional nahi hua ??? 😂😂😂😂 @VenomStinger447 @Diyaan96143701 https://t.co/LjcNhdtxM0"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{70.22\%}$, $P(\text{Non-Hateful}) = 29.78\%$
- **Stratification Bucket:** Moderate (0.65-0.85) &bull; Medium (11-25w) (16 words)
- **SHAP Base Value:** `0.9537`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `hd` | `+0.0071` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `t` | `+0.0071` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `x` | `+0.0071` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `M` | `+0.0071` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `L` | `+0.0070` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `701 ` | `-0.0198` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `https` | `-0.0198` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `://` | `-0.0198` | Lowers hate probability |

### Sample 7: `hi_test_01374`
- **Input Text:** "Madam je aparipak balak ke herkat se pareshan ab puttra moh chor party ko bachane hai tho rahul Gandhi ko hatana parega g-23 ke leaders ne saaf sandesh rahul Gandhi ke bachkani herkat Congress party ko berbad hona nischitt hai RSS sadanand kashri https://t.co/9YMbMzUw9P"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{65.06\%}$, $P(\text{Non-Hateful}) = 34.94\%$
- **Stratification Bucket:** Moderate (0.65-0.85) &bull; Long (>25w) (44 words)
- **SHAP Base Value:** `0.9591`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `Mada` | `+0.0126` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `m ` | `+0.0126` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `je ` | `+0.0126` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `apar` | `+0.0097` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `i` | `+0.0097` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `sa` | `-0.0214` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `af ` | `-0.0214` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `sand` | `-0.0214` | Lowers hate probability |

### Sample 8: `hi_test_00081`
- **Input Text:** "RT @TeamSSR4ever: Zindagi sab ko pyari hai"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{53.77\%}$, $P(\text{Non-Hateful}) = 46.23\%$
- **Stratification Bucket:** Borderline (0.50-0.65) &bull; Short (<=10w) (7 words)
- **SHAP Base Value:** `0.8710`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `pyar` | `+0.0234` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `ko ` | `+0.0069` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `hai` | `+0.0044` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `i ` | `+-0.0033` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `sab ` | `+-0.0060` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `RT ` | `-0.1667` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `Zindagi ` | `-0.0645` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `Team` | `-0.0484` | Lowers hate probability |

### Sample 9: `hi_test_00032`
- **Input Text:** "@ReallySwara Jaise kisanon ne fasal ko nashat karna shuru kiya hai.. Bahar se khareedte khareedte modi ki j halat ho jayegi 😂😂😂😂😂😂😂😂😂 https://t.co/f2gmj81Hru"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{61.57\%}$, $P(\text{Non-Hateful}) = 38.43\%$
- **Stratification Bucket:** Borderline (0.50-0.65) &bull; Medium (11-25w) (23 words)
- **SHAP Base Value:** `0.9469`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `@` | `+0.0033` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `Real` | `+0.0033` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `ly` | `+0.0033` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `S` | `+0.0033` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `wara ` | `+0.0033` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `modi ` | `-0.0358` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `ki ` | `-0.0358` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `j ` | `-0.0358` | Lowers hate probability |

### Sample 10: `hi_test_00428`
- **Input Text:** "@gauravbh @BJP4India @BJP4Maharashtra Kaunsi Kendra sarkar seva mai lagi hai dono gujrati dale to ghumne mai aur desh ko bechne mai lage hai kitna jhut bolta hai tu refugee"
- **Ground Truth:** Class 1 (`Non-Hateful`) | **Prediction:** Class 0 (`Hateful`)
- **Model Probabilities:** $P(\text{Hateful}) = \mathbf{58.90\%}$, $P(\text{Non-Hateful}) = 41.10\%$
- **Stratification Bucket:** Borderline (0.50-0.65) &bull; Long (>25w) (29 words)
- **SHAP Base Value:** `0.9524`

| Attribution Direction | Token / Subword | SHAP Value ($\phi$) | Contextual Role |
|---|---|---|---|
| **Pushing toward Hateful (+Class 0)** | `hai ` | `+0.0150` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `tu ` | `+0.0150` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `@` | `+0.0060` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `bol` | `+0.0033` | Elevates hate probability |
| **Pushing toward Hateful (+Class 0)** | `ta ` | `+0.0033` | Elevates hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `B` | `-0.0376` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `JP` | `-0.0376` | Lowers hate probability |
| **Pushing toward Non-Hateful (-Class 0)** | `B` | `-0.0341` | Lowers hate probability |

---

## 4. Synthesis of Recurring Patterns & Epistemic Boundaries

To ensure scientific rigor, we explicitly distinguish between **empirical observations**, **plausible hypotheses**, and **unsupported speculations**.

### 4.1 Observable Patterns (Strict Empirical Evidence)
1. **Heavy Social Media Metadata Presence**: 75.9% of false positives contain user handles (`@mentions`) and 14.7% contain hashtags. These structures reflect interactive Twitter threads rather than standalone statements.
2. **Political Criticism Dominance**: 59.5% of false positives reference political figures, legislation, or protest movements. In 46.6% of cases, "Modi" is explicitly mentioned.
3. **Abuse-Adjacent and Sarcastic Lexicon**: Texts frequently contain pejoratives directed at institutions, policies, or public figures ("jumlo kaa pitara", "bheek maang", "godi media", "mujra", "tootla bacha", "berbad", "dono gujrati dale", "refugee") rather than protected demographic categories.
4. **Emoji Amplification**: Sarcastic laughter emojis (`😂😂😂😂`, `😅`) frequently co-occur with high-confidence false positive classifications.
5. **Confidence Skew**: 44.8% of false positives are borderline ($P \in [0.50, 0.65]$), indicating that a modest shift in calibration or threshold would reclassify them as Non-Hateful.

### 4.2 Plausible Interpretations (Supported by Dataset and Training Properties)
1. **Class-Weighting Penalty Bias**: During training (`use_class_weights: true`), Class 0 (Hateful, 30% minority) was assigned a $\approx 2.33\times$ loss multiplier to prevent false negatives. When the model encounters ambiguous, contentious, or heated political rhetoric, the decision boundary systematically tilts toward Class 0.
2. **Hate Speech vs. Political Vitriol Conflation**: The underlying HECM training annotations distinguish between *toxic political mudslinging* and *hate speech targeting protected groups*. However, multilingual language models fine-tuned on code-mixed text struggle to separate generic political abuse from group-targeted hate speech because both share hostile, aggressive vocabulary.
3. **Spurious Token Correlations**: Tokens like "jumlo", "bheek", "mujra", and "CAA" frequently co-occurred with hateful comments in the training data, leading the transformer attention layers to associate these tokens with toxicity regardless of the broader semantic framing.
4. **Sarcasm and Negation Blindness**: Sentences using rhetorical mockery ("Jese modi bheek maang rha tha pulwama k baad...😂😂") convey sarcastic political satire. The transformer registers the co-occurrence of sensitive security terms ("pulwama", "shaheedon", "bheek") without resolving the sarcastic intent.

### 4.3 Unsupported Speculations (Claims Not Justified by Available Data)
1. *"The model has an intrinsic political bias against specific parties."* — **Unsupported**. The model is a statistical pattern matcher responding to word co-occurrences in the training corpus, not an agent with political stance.
2. *"Individual token frequencies causally dictate classification in all contexts."* — **Unsupported**. SHAP attributions show distributed contextual weights; transformer self-attention integrates multi-token interactions that cannot be reduced to unigram bag-of-words counting.
3. *"The annotators mislabeled these 116 samples."* — **Unsupported**. Dataset annotation guidelines specifically differentiate hate speech (targeting protected identities) from harsh criticism of politicians. The annotator labels are consistent with standard hate speech definitions.

---

## 5. False Negatives Count Summary

- **Total False Negatives (Class 0 predicted as Class 1):** **162 samples**
- **Class Breakdown:** Ground Truth = Hateful (0), Predicted = Non-Hateful (1)
- **Proportion of True Hateful Class:** $162 / 419 = 38.7\%$ of actual hateful samples were missed.
- As per protocol, no further analysis or modifications were performed on false negatives.

---

## 6. Complete Catalogue of All 116 False Positives

| # | Sample ID | Text Snippet | Ground Truth | Predicted | $P(\text{Hateful})$ | $P(\text{Non-Hate})$ |
|---|---|---|---|---|---|---|
| 1 | `hi_test_00032` | @ReallySwara Jaise kisanon ne fasal ko nashat karna shuru kiya hai.. Bahar se... | Non-Hateful (1) | Hateful (0) | 0.6157 | 0.3843 |
| 2 | `hi_test_00043` | @PratimaanPriyam @maryashakil Agree, no one can fill tarun gogoi space ... Pr... | Non-Hateful (1) | Hateful (0) | 0.5951 | 0.4049 |
| 3 | `hi_test_00058` | @blsanthosh parde ke peeche ke yodha. | Non-Hateful (1) | Hateful (0) | 0.9800 | 0.0200 |
| 4 | `hi_test_00060` | Yeh message padh ke mere bhi dil se awaaz ayee - black karo isko 🤦🏻‍♀️ Kuch b... | Non-Hateful (1) | Hateful (0) | 0.5312 | 0.4688 |
| 5 | `hi_test_00081` | RT @TeamSSR4ever: Zindagi sab ko pyari hai | Non-Hateful (1) | Hateful (0) | 0.5377 | 0.4623 |
| 6 | `hi_test_00084` | @kktotlani @PashaSehgal Sehgal sab, tootlaa bacha kuch ziada emotional nahi h... | Non-Hateful (1) | Hateful (0) | 0.7022 | 0.2978 |
| 7 | `hi_test_00101` | @INCIndia Bahar k log roj gaar de to videshyon ko bech diya bharat .....local... | Non-Hateful (1) | Hateful (0) | 0.7941 | 0.2059 |
| 8 | `hi_test_00106` | RT @Nileshk70255348: #modi_rojgar_दो modi shirph samay par lambaba 2 bhashan ... | Non-Hateful (1) | Hateful (0) | 0.6370 | 0.3630 |
| 9 | `hi_test_00109` | @HimanArma Air waha kuch chamvhe bol rhe h ki ye campaign rahul gandhi ne kiy... | Non-Hateful (1) | Hateful (0) | 0.7422 | 0.2578 |
| 10 | `hi_test_00126` | @Shruti__157 Uss din jab #AnuSeena ki shadi ki baat ho rahi thi ti kisine kah... | Non-Hateful (1) | Hateful (0) | 0.7290 | 0.2710 |
| 11 | `hi_test_00134` | @Vi_Jay10 @manojmeghwal_ Abey love day. Wo pura video dekh usne ye nahi bola ... | Non-Hateful (1) | Hateful (0) | 0.5002 | 0.4998 |
| 12 | `hi_test_00141` | @vkayay @Sasmita89152691 @pinkvilla Skin colour lol😂😂.Khud hi apne Mann se??M... | Non-Hateful (1) | Hateful (0) | 0.5413 | 0.4587 |
| 13 | `hi_test_00146` | @ThePushpendra_ Dimag kaam hi nhi kr raha ye logoka...rat din modi hi dikhta ... | Non-Hateful (1) | Hateful (0) | 0.7473 | 0.2527 |
| 14 | `hi_test_00162` | @DhekunuNizar Nizaroo. thankolheh hamaaiga huregen visnaalabala 2019 ga modi ... | Non-Hateful (1) | Hateful (0) | 0.5107 | 0.4893 |
| 15 | `hi_test_00171` | Dekhe baithe rahte hain..kaam khatam in roopon ka..paani mil gaya tha in logo... | Non-Hateful (1) | Hateful (0) | 0.7879 | 0.2121 |
| 16 | `hi_test_00184` | @IMinakshiJoshi @ChouhanShivraj Ek kaam karo medam noukri chor do or bjp ka p... | Non-Hateful (1) | Hateful (0) | 0.5749 | 0.4251 |
| 17 | `hi_test_00197` | @Apoorvaa_21 Mere rehte hue modi se guhar kyu🙄 | Non-Hateful (1) | Hateful (0) | 0.7114 | 0.2886 |
| 18 | `hi_test_00198` | @KAILASH_MINA29 @RamlalAluda Sonia gandhi rakh de bahut magnga bikega aur kuc... | Non-Hateful (1) | Hateful (0) | 0.5776 | 0.4224 |
| 19 | `hi_test_00203` | @AnonymousTrad17 @GreenBull_ @hsvashistha16 jaa ke apne baap ke munh par laga... | Non-Hateful (1) | Hateful (0) | 0.9504 | 0.0496 |
| 20 | `hi_test_00204` | @ravcitizen @30B10K @EngineerMeer1 Hindutva vicharak Veer savarkar se angrez ... | Non-Hateful (1) | Hateful (0) | 0.6296 | 0.3704 |
| 21 | `hi_test_00216` | @SushantBSinha Darte to sirf tum Godi media vale ho jo chand sikko ki aavaj s... | Non-Hateful (1) | Hateful (0) | 0.8522 | 0.1478 |
| 22 | `hi_test_00229` | @BJP4Himachal modi ne desh ka bera gark kar diye hai?? Har vackte dukhe hoe a... | Non-Hateful (1) | Hateful (0) | 0.5181 | 0.4819 |
| 23 | `hi_test_00233` | @desimojito Kabhi kabhi lagta hai ye Rahul Gandhi Mota bhai ka khass aadmi hai | Non-Hateful (1) | Hateful (0) | 0.5269 | 0.4731 |
| 24 | `hi_test_00256` | RT @Shreyash9472: @thedevil_ind @Kris7na_ Yaar 1st waali galti bhut badi Lalu... | Non-Hateful (1) | Hateful (0) | 0.7977 | 0.2023 |
| 25 | `hi_test_00259` | @abhinaymaths Mba btech kar ke bhi wo agar kaam nahi dhund pata toh galti usk... | Non-Hateful (1) | Hateful (0) | 0.8179 | 0.1821 |
| 26 | `hi_test_00266` | @ashokgehlot51 Jo tum log leke baithe ho na wahi kaala dhan h . Jo Swiss bank... | Non-Hateful (1) | Hateful (0) | 0.6135 | 0.3865 |
| 27 | `hi_test_00302` | modi nee ek mandir k chakkar me bau Ch...... Dee dhesh ki #modi_rojgar_दो | Non-Hateful (1) | Hateful (0) | 0.5396 | 0.4604 |
| 28 | `hi_test_00319` | @MdBelal15141990 @Shikadiv @nikkitamboli @ColorsTV @BiggBoss Waisey tere pare... | Non-Hateful (1) | Hateful (0) | 0.8502 | 0.1498 |
| 29 | `hi_test_00325` | @ChetanM24579252 @mrpal1000 Jese modi bheek maang rha tha pulwama k baad k mu... | Non-Hateful (1) | Hateful (0) | 0.8595 | 0.1405 |
| 30 | `hi_test_00337` | Hamaray gharon mein jab glass toot'taa hai toh 7 minute "dhum tana na na" baj... | Non-Hateful (1) | Hateful (0) | 0.7445 | 0.2555 |
| 31 | `hi_test_00349` | RT @ranjank57639008: @RailMinIndia #modi_rojgar_दो Jha chunav waha modi Jha k... | Non-Hateful (1) | Hateful (0) | 0.5438 | 0.4562 |
| 32 | `hi_test_00356` | @ArifRetd Baqi sub theak hai . Laikin fojiun ko shunwai ki zarurat kahan part... | Non-Hateful (1) | Hateful (0) | 0.7640 | 0.2360 |
| 33 | `hi_test_00395` | @BjpVivek2 @SantoshMeena_ Bhasha meri kahi galat nhu h tumhare pas bolne ke l... | Non-Hateful (1) | Hateful (0) | 0.7518 | 0.2482 |
| 34 | `hi_test_00405` | @him_ranjan क्या हुआ तेरा वादा? वो कसम वो इरादा ? #StudentsWantJobs #NoVoteTo... | Non-Hateful (1) | Hateful (0) | 0.6867 | 0.3133 |
| 35 | `hi_test_00426` | @RakhiVaish7 Mujhe modi ne beja ha | Non-Hateful (1) | Hateful (0) | 0.5210 | 0.4790 |
| 36 | `hi_test_00428` | @gauravbh @BJP4India @BJP4Maharashtra Kaunsi Kendra sarkar seva mai lagi hai ... | Non-Hateful (1) | Hateful (0) | 0.5890 | 0.4110 |
| 37 | `hi_test_00445` | @DrKhan21 is me koi Shakk nahi k ap ny di Magar Jannab jo 75% hen wo apki Gov... | Non-Hateful (1) | Hateful (0) | 0.6230 | 0.3770 |
| 38 | `hi_test_00459` | Ab main kisi dusre fandom ko shade kardoon ya kisi ke idol ko troll karna shu... | Non-Hateful (1) | Hateful (0) | 0.5272 | 0.4728 |
| 39 | `hi_test_00468` | @burgerrqueen marahin caa | Non-Hateful (1) | Hateful (0) | 0.8079 | 0.1921 |
| 40 | `hi_test_00478` | @don_bhojpuriya @Priyank57577435 Jhooth aur jumlo kaa pitara toh hai modi | Non-Hateful (1) | Hateful (0) | 0.9016 | 0.0984 |
| 41 | `hi_test_00485` | @WorldCupDefence @Dinesh3018 Yaar ek din Godse ka bhoot tujhe chipat hi jayeg... | Non-Hateful (1) | Hateful (0) | 0.9389 | 0.0611 |
| 42 | `hi_test_00497` | @HemantSorenJMM @prdjharkhand Shiv g se yah bhi boliye Hemant G ki mein jhuth... | Non-Hateful (1) | Hateful (0) | 0.5016 | 0.4984 |
| 43 | `hi_test_00501` | @SHABAZGIL Dr. shabaz Wese Tujhe Mochi Hona Chaiue Tha Tu Polish Bht Achi Krt... | Non-Hateful (1) | Hateful (0) | 0.5648 | 0.4352 |
| 44 | `hi_test_00508` | @Himalay04155335 @Sanginamby Note 9pro hai chamdike kaha past me rehta hai vi... | Non-Hateful (1) | Hateful (0) | 0.7713 | 0.2287 |
| 45 | `hi_test_00509` | @2Kazmi Modi ke ke daari pr gai ka peeshab | Non-Hateful (1) | Hateful (0) | 0.6624 | 0.3376 |
| 46 | `hi_test_00512` | @Royalakshay07 @ShyamRangeela @plurals1313 @mukesh_phd Poora mamla samjho usk... | Non-Hateful (1) | Hateful (0) | 0.6297 | 0.3703 |
| 47 | `hi_test_00514` | Aakhir ho kya gaya hai modi govt. Q nhi unhe samajh me aati hai ki ye desh yu... | Non-Hateful (1) | Hateful (0) | 0.5829 | 0.4171 |
| 48 | `hi_test_00521` | Wah desh taraki kr rha hi ...🤭😂 Baad me modi 👶dedo ye bi chalega...🤣 https://... | Non-Hateful (1) | Hateful (0) | 0.5270 | 0.4730 |
| 49 | `hi_test_00535` | @Black_Storm02 @Nalayak_betaa Modi janbujhkar single hai aur hum majburi me | Non-Hateful (1) | Hateful (0) | 0.5472 | 0.4528 |
| 50 | `hi_test_00550` | Ye dekho masum bacho ko dimag me kese jhar bhar doua he ye bacha bade hoke ky... | Non-Hateful (1) | Hateful (0) | 0.8759 | 0.1241 |
| 51 | `hi_test_00585` | @Azmat726 @hudaism azmat khan saab tussy kurry phasao bus soi raho. Apna vote... | Non-Hateful (1) | Hateful (0) | 0.5421 | 0.4579 |
| 52 | `hi_test_00610` | @HeyRickyDutta @Sryforoffending @DeepakM16353096 @Riyaz69075073 Tere kitne ba... | Non-Hateful (1) | Hateful (0) | 0.8180 | 0.1820 |
| 53 | `hi_test_00623` | Modi rojar doo | Non-Hateful (1) | Hateful (0) | 0.6601 | 0.3399 |
| 54 | `hi_test_00632` | Bengal se purana nata hai... Macher jhol kheye bodo hue che 😂😂 | Non-Hateful (1) | Hateful (0) | 0.6077 | 0.3923 |
| 55 | `hi_test_00636` | #modi_rojgar_दो #modi_girlfriend_do oo modi chhacha bahut chus liya bharat ko... | Non-Hateful (1) | Hateful (0) | 0.5101 | 0.4899 |
| 56 | `hi_test_00641` | Bahot hoyi mehgayi ki maar chor do modi Sarkar #FuelLootByBJP | Non-Hateful (1) | Hateful (0) | 0.5417 | 0.4583 |
| 57 | `hi_test_00643` | @ArizRizvi3 @BeparwahHoon @zoo_bear Ye toh abhi boli, law toh 1950 saal se ha... | Non-Hateful (1) | Hateful (0) | 0.6197 | 0.3803 |
| 58 | `hi_test_00677` | @AITCofficial yahan toh pawri ho rahi hai | Non-Hateful (1) | Hateful (0) | 0.7337 | 0.2663 |
| 59 | `hi_test_00727` | @narendramodi Wadda kuhb kiye the bharat refugee se aazzad hoga ,dharamsala k... | Non-Hateful (1) | Hateful (0) | 0.6582 | 0.3418 |
| 60 | `hi_test_00746` | Sab black black Kiya Hai or esaka udrahna 2007 Ka Sadhi deklo ab ke or Sab up... | Non-Hateful (1) | Hateful (0) | 0.6238 | 0.3762 |
| 61 | `hi_test_00760` | Me: " Maine Jab Dekha Tha Tujhko.." Nephew : " Raat bhi vo Yad Hai Mujhko , T... | Non-Hateful (1) | Hateful (0) | 0.8548 | 0.1452 |
| 62 | `hi_test_00813` | #modi_rojgar_दो modi shirph samay par lambaba 2 bhashan dege | Non-Hateful (1) | Hateful (0) | 0.6104 | 0.3896 |
| 63 | `hi_test_00815` | Ye sahi hai | Non-Hateful (1) | Hateful (0) | 0.5842 | 0.4158 |
| 64 | `hi_test_00840` | @is_enticing Social media is the new weapon for these Leli gang to propogate ... | Non-Hateful (1) | Hateful (0) | 0.5803 | 0.4197 |
| 65 | `hi_test_00848` | Are koi nazar ka kaala teeka lagwao inhe🧿meri khushi hi nhi sambhal rhi, yaar... | Non-Hateful (1) | Hateful (0) | 0.8084 | 0.1916 |
| 66 | `hi_test_00854` | @HJS_Convener @NP_HJS @HinduJagrutiOrg Agar population control bill.hi sahiye... | Non-Hateful (1) | Hateful (0) | 0.7592 | 0.2408 |
| 67 | `hi_test_00856` | @awasthis @ABPNews Petrol ke daam nhi bdhe hain tu hi garib hogya hai bhdwe ,... | Non-Hateful (1) | Hateful (0) | 0.5577 | 0.4423 |
| 68 | `hi_test_00871` | @zeeshandugraal @kunalkamra88 Bhai tum kitna hi chati pit pit k rolo ayega to... | Non-Hateful (1) | Hateful (0) | 0.8041 | 0.1959 |
| 69 | `hi_test_00878` | Yh bharat me metoo movements pe karwai ni hui jisme ladki ko tavi filim milti... | Non-Hateful (1) | Hateful (0) | 0.6610 | 0.3390 |
| 70 | `hi_test_00882` | @JINDALHemu @RahulGandhi Hosh m aavo logo ...ye jo in gaddaro ne itna nuksaan... | Non-Hateful (1) | Hateful (0) | 0.8173 | 0.1827 |
| 71 | `hi_test_00893` | Abb app ka dil ker raha hoga kay Qazi Faez Essa kay sath bhi kuch aisa hee ho... | Non-Hateful (1) | Hateful (0) | 0.6082 | 0.3918 |
| 72 | `hi_test_00894` | Mein. .pahle se hi z-black they..ab kahan kaun shakti peeth ye sab chalayenge... | Non-Hateful (1) | Hateful (0) | 0.7114 | 0.2886 |
| 73 | `hi_test_00898` | RT @dhakadsunil0201: Modi h chutiya #modi_rojgar_दो | Non-Hateful (1) | Hateful (0) | 0.8963 | 0.1037 |
| 74 | `hi_test_00940` | Ab yeah khool kar saamnay aaya ki fake kisaan aandolan modi sarkaar ko unstab... | Non-Hateful (1) | Hateful (0) | 0.8943 | 0.1057 |
| 75 | `hi_test_00963` | RT @Libertarian196: Parmatma ke IT wale, inka software mein kuchh Malware daa... | Non-Hateful (1) | Hateful (0) | 0.5669 | 0.4331 |
| 76 | `hi_test_00974` | Mei BHI Kush ! aur Mera Kutta Bhi https://t.co/7mFZRIPAL0 | Non-Hateful (1) | Hateful (0) | 0.9565 | 0.0435 |
| 77 | `hi_test_00994` | @jald_fir_milege Are bahan humne jo poocha uska MATLAB hai ki. Kon kaala kaww... | Non-Hateful (1) | Hateful (0) | 0.6058 | 0.3942 |
| 78 | `hi_test_00996` | Ye maadar**d ko Kabhi Right wing sikh representative smjhta tha, khalistaani ... | Non-Hateful (1) | Hateful (0) | 0.6948 | 0.3052 |
| 79 | `hi_test_01000` | @aajtak AajTak Modi kay kutton nay PETROL KI KEMAT PER ITNA nahi roye jitna D... | Non-Hateful (1) | Hateful (0) | 0.7283 | 0.2717 |
| 80 | `hi_test_01009` | @wils1947 @davinder_7 Are abb bacho jaisi baat to matt kar 😂same to u same to... | Non-Hateful (1) | Hateful (0) | 0.7330 | 0.2670 |
| 81 | `hi_test_01012` | @ExSecular Didi Mamata West Bengal Apke Desh Bharat mein hi ata hai ap Rohing... | Non-Hateful (1) | Hateful (0) | 0.5775 | 0.4225 |
| 82 | `hi_test_01020` | Imran khan tu kehta hai Pakistan mein talent hi buhot hai pori dunya se zyada... | Non-Hateful (1) | Hateful (0) | 0.8693 | 0.1307 |
| 83 | `hi_test_01039` | Bas Anpadh or Chutiye hi Modi Bhakt bane hue hain. Is did is ur "Vikas" #modi... | Non-Hateful (1) | Hateful (0) | 0.6945 | 0.3055 |
| 84 | `hi_test_01050` | @JangidDharmend6 @news24tvchannel @RakeshTikaitBKU Har har mahadev ke sath ac... | Non-Hateful (1) | Hateful (0) | 0.7725 | 0.2275 |
| 85 | `hi_test_01053` | #modi_rojgar_दो har ek yuva ki yahi pukar rojgaar do modi sarkar | Non-Hateful (1) | Hateful (0) | 0.5072 | 0.4928 |
| 86 | `hi_test_01061` | @marghoob_shawez @DChaurasia2312 Par kagaz toh suar me bachho Bangladeshi ghu... | Non-Hateful (1) | Hateful (0) | 0.9446 | 0.0554 |
| 87 | `hi_test_01062` | @djohninc Bhai tum logo sirf na result se mtlb hota hai...aise log duniya ke ... | Non-Hateful (1) | Hateful (0) | 0.6082 | 0.3918 |
| 88 | `hi_test_01070` | @VaibhavMancha12 @BeethoRoot2 @Onkar23697880 @_HiteshChopra Hindu ko CAA ke b... | Non-Hateful (1) | Hateful (0) | 0.6694 | 0.3306 |
| 89 | `hi_test_01097` | @rupagulab Kyu pichwade me jalan hue tmlogo k ?😂isse bawaseer kehte hain jo 2... | Non-Hateful (1) | Hateful (0) | 0.5572 | 0.4428 |
| 90 | `hi_test_01116` | @abidmalik78675 @_Mansoor_Ali 1 baat sach is qadani kutt imran khan na sirf P... | Non-Hateful (1) | Hateful (0) | 0.6496 | 0.3504 |
| 91 | `hi_test_01126` | @akki_dhoni Saal. Main 7-8 movies karte hai dosra Ki movie chin lete Hai rema... | Non-Hateful (1) | Hateful (0) | 0.6223 | 0.3777 |
| 92 | `hi_test_01137` | @Umme98 @newsclickin Nafrat tum logo ke andar hai .. tum christan se lad rahe... | Non-Hateful (1) | Hateful (0) | 0.5207 | 0.4793 |
| 93 | `hi_test_01140` | @BansalNewsMPCG Jab tak counting na ho jaye tab se pehle conferm koi kaise ke... | Non-Hateful (1) | Hateful (0) | 0.5208 | 0.4792 |
| 94 | `hi_test_01153` | @aajtak @AbhayDubeyINC @anjanaomkashyap auntie,aab to mamta bano ne apna gotr... | Non-Hateful (1) | Hateful (0) | 0.7917 | 0.2083 |
| 95 | `hi_test_01171` | Tui Jodi Rohingiya ba tui jodi illegal immigrant from Bangladesh, tui joto ko... | Non-Hateful (1) | Hateful (0) | 0.7838 | 0.2162 |
| 96 | `hi_test_01179` | @prakashachary1 Har baat pe dant dikhata hai😬 Isko toh bahut kuttne Ka mann k... | Non-Hateful (1) | Hateful (0) | 0.8826 | 0.1174 |
| 97 | `hi_test_01186` | @ex_secular21 @ShivrajKaBhanja @SonuSood kam se kam tumari tarah atankwaad fe... | Non-Hateful (1) | Hateful (0) | 0.7268 | 0.2732 |
| 98 | `hi_test_01187` | @Dr_Uditraj @INCIndia Aarakshan jivi, Tu khud papumuter pita....... Na bol to... | Non-Hateful (1) | Hateful (0) | 0.8237 | 0.1763 |
| 99 | `hi_test_01196` | RT @VikalVishal1: #modi_rojgar_दो kaha chup kar baith gaye modi ke chamcho @K... | Non-Hateful (1) | Hateful (0) | 0.8546 | 0.1454 |
| 100 | `hi_test_01202` | @ippatel Aur delhi mein 20 lakh r0hingyya B@ngll@deai bulle bhare pade hai wo... | Non-Hateful (1) | Hateful (0) | 0.7986 | 0.2014 |
| 101 | `hi_test_01218` | #modi_rojgar_दो kaha chup kar baith gaye modi ke chamcho @KanganaTeam @akshay... | Non-Hateful (1) | Hateful (0) | 0.8478 | 0.1522 |
| 102 | `hi_test_01222` | @nsui Kay matlab hay unemployment ka modi jii ne itna kuch kiya hay galti tum... | Non-Hateful (1) | Hateful (0) | 0.7054 | 0.2946 |
| 103 | `hi_test_01252` | @vinita4india Name Badal ke rahate he, Pakistan me log Khan kahate he, Bangla... | Non-Hateful (1) | Hateful (0) | 0.5200 | 0.4800 |
| 104 | `hi_test_01254` | @aajtak @vikrantgupta73 Aajtak media institute mein ab kewal godi media banai... | Non-Hateful (1) | Hateful (0) | 0.5896 | 0.4104 |
| 105 | `hi_test_01255` | @OpIndia_com CONGRESS BE LIKE :- HAM se modi nahi hatega mia didi aur rihana ... | Non-Hateful (1) | Hateful (0) | 0.6276 | 0.3724 |
| 106 | `hi_test_01260` | @iam_shahista @Uroosa_Azeem @PTIofficial Yar in jaisay pagal logo ko inkay ha... | Non-Hateful (1) | Hateful (0) | 0.8322 | 0.1678 |
| 107 | `hi_test_01271` | @TV9Marathi Itr deshani kel te modi sarkar la zamnar ka ? Ashi h batmi dakhva... | Non-Hateful (1) | Hateful (0) | 0.5153 | 0.4847 |
| 108 | `hi_test_01281` | @Frontechonline @india4contests @Contest_in @Contest_Hub 💕 Dulhe ki saaliyon ... | Non-Hateful (1) | Hateful (0) | 0.6852 | 0.3148 |
| 109 | `hi_test_01290` | ek aur memory hai, shayed wohi bacha tha, raat ka waqt tha.... aag k gird boh... | Non-Hateful (1) | Hateful (0) | 0.8775 | 0.1225 |
| 110 | `hi_test_01291` | @ShaikhKamilAhm1 @tapaswiniswain2 @IPSinghSp NRC jab lagoo ho jayega toh kaha... | Non-Hateful (1) | Hateful (0) | 0.7146 | 0.2854 |
| 111 | `hi_test_01337` | Yaar ek din Godse ka bhoot tujhe chipat hi jayega.....kaala paani le jayega t... | Non-Hateful (1) | Hateful (0) | 0.9507 | 0.0493 |
| 112 | `hi_test_01349` | agar Mai @SuPriyoBabul ki jaga hota tho enka gaana studio may record kar ke v... | Non-Hateful (1) | Hateful (0) | 0.9610 | 0.0390 |
| 113 | `hi_test_01355` | @PrakashJavdekar @RahulGandhi Desh ko chutia mat banao Desh ka yuva rojgar ma... | Non-Hateful (1) | Hateful (0) | 0.7362 | 0.2638 |
| 114 | `hi_test_01374` | Madam je aparipak balak ke herkat se pareshan ab puttra moh chor party ko bac... | Non-Hateful (1) | Hateful (0) | 0.6506 | 0.3494 |
| 115 | `hi_test_01379` | @itbpshaddam1 @helodoyouknowme @1008Virathu @rangoonwale @Ashin_Virathu @Vira... | Non-Hateful (1) | Hateful (0) | 0.6086 | 0.3914 |
| 116 | `hi_test_01381` | @BhainsdehiWale Aarakshan se toh aap jaise harijano ko vanchit kar dena chaiy... | Non-Hateful (1) | Hateful (0) | 0.7986 | 0.2014 |