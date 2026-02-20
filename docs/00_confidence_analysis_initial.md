# Confidence Score Analysis (Initial)

**Date:** February 9, 2026
**Purpose:** Preliminary analysis of log probabilities collected from fast labeling model (Mistral-7B) to assess prediction confidence across questions and sectors.

---

## Data Overview

**Labels collected:** 1,500 comments × 36 questions = 54,000 labels
**Logprobs collected:** 54,000 confidence scores
**Sectors:** Transport (500), Housing (500), Food (500)
**Questions per comment:** 7 norms + 29 survey (sector-specific)

---

## Sample Confidence Scores

**Transport sector (first comment):**
```
1.1_gate: -0.045 (95.6% confidence)
1.1.1_stance: -0.106 (89.9% confidence)
1.2.1_descriptive: -0.027 (97.3% confidence)
```

**Housing sector (first comment):**
```
1.1_gate: -0.200 (81.9% confidence)
1.1.1_stance: -0.161 (85.1% confidence)
1.2.2_injunctive: -0.018 (98.2% confidence)
```

**Food sector (first comment):**
```
1.1_gate: -0.006 (99.4% confidence)
1.1.1_stance: -0.106 (90.0% confidence)
1.2.1_descriptive: -0.002 (99.8% confidence)
```

**Observation:** Most predictions have high confidence (logprob > -0.3, i.e., >74% probability).

---

## Next Steps: Detailed Analysis

### 1. Confidence Distribution by Question

**Goal:** Identify which questions have lowest average confidence (likely correlate with lowest verification accuracy).

**Hypothesis:**
- 1.2.1_descriptive (32% accuracy) should have lower avg confidence than 100%-accuracy questions
- 1.1_gate (44% accuracy) should show bimodal distribution (confident but wrong)

**Analysis:**
```python
# For each question, calculate:
- Mean logprob
- Median logprob
- % predictions with logprob < -0.5 (low confidence)
- % predictions with logprob < -1.0 (very low confidence)
```

### 2. Confidence vs Verification Accuracy

**Goal:** Test hypothesis that low confidence predicts verification mismatch.

**Method:**
1. Load verification samples (900 samples with vllm_label and reasoning_label)
2. Match to logprobs from norms_labels.json
3. Calculate correlation between logprob and match/mismatch

**Expected result:**
- Mismatches have lower average logprob than matches
- ROC curve: logprob threshold for flagging uncertain predictions

### 3. Optimal Threshold Identification

**Goal:** Find logprob threshold for flagging uncertain predictions for human review.

**Trade-off:**
- High threshold (e.g., -0.5): flags many predictions, high recall but low precision
- Low threshold (e.g., -2.0): flags few predictions, low recall but high precision

**Optimization criterion:** Maximize F1 score for identifying verification mismatches.

### 4. Sector Differences

**Goal:** Do some sectors have systematically lower confidence?

**Hypothesis:**
- Food (diet/veganism) may have lower confidence due to emotional/ambiguous language
- Transport (EVs) may have higher confidence due to more technical/factual discussions

### 5. Question Type Differences

**Goal:** Do norms questions have lower confidence than survey questions?

**Data:**
- Norms questions: 7 questions, 32-84% verification accuracy
- Survey questions: 29 questions, 76-100% verification accuracy

**Expected result:** Norms questions have lower average confidence than survey questions.

---

## Preliminary Insights

1. **Most predictions are high confidence:** Sample logprobs range from -0.002 to -0.200 (98% to 82% probability), suggesting model is generally confident in its predictions.

2. **Low confidence ≠ low accuracy (always):** A confident but wrong prediction (e.g., 1.1_gate: 44% accuracy despite likely high confidence) indicates systematic prompt issues, not model uncertainty.

3. **Utility of confidence scores:**
   - **Flag uncertain cases:** logprob < -1.0 → human review
   - **Estimate label quality:** avg logprob per question → expected accuracy
   - **Optimize verification sampling:** prioritize low-confidence predictions for verification instead of random sampling

---

## Implementation Plan

**Week 1:**
1. Calculate confidence distribution statistics per question
2. Match verification samples to logprobs
3. Plot correlation: confidence vs verification accuracy

**Week 2:**
1. Identify optimal threshold for flagging
2. Re-run verification with targeted sampling (low-confidence predictions)
3. Compare accuracy: random sampling vs confidence-targeted sampling

**Week 3:**
1. Implement prompt optimizations from 00_prompt_optimization.md
2. Re-label with optimized prompts + collect logprobs
3. Compare: old accuracy & confidence vs new accuracy & confidence

---

## Technical Notes

**Log probability to probability conversion:**
```python
import math
logprob = -0.106
probability = math.exp(logprob)  # 0.899 = 89.9%
```

**Average logprob calculation:**
```python
# Average across all tokens in response
token_logprobs = [-0.05, -0.10, -0.15]
avg_logprob = sum(token_logprobs) / len(token_logprobs)  # -0.10
```

**Why average, not sum?**
- Different responses have different token counts
- Sum would penalize longer responses (more tokens = more negative sum)
- Average normalizes across response length

---

## Conclusion

Successfully integrated confidence scores (log probabilities) into the fast labeling pipeline. All 54,000 labels now have associated confidence estimates, enabling:
1. Identification of uncertain predictions
2. Quality estimation without full verification
3. Optimized verification sampling strategies

Next steps: Detailed analysis linking confidence to verification accuracy, followed by targeted prompt improvements for lowest-confidence/lowest-accuracy questions.
