# Mistral-7B Response Parsing: Bugs Found and Fixed

## Background

Reddit comments in our dataset were labeled by Mistral-7B (via vLLM) using a
hierarchical social norms schema. Each inference call returned a raw text response
that was then parsed by `_parse_single_choice` to extract a structured answer.

Raw responses are stored per-record under `raw_responses[qid]` in each checkpoint
file (`paper4data/checkpoints/checkpoint_*.json`), which means labels can be
re-derived from raw text without re-running the LLM.

---

## Bugs Identified

### Bug 1: "X or Y" Ambiguity

**Question affected:** `1.3.1_reference_group`
**Options:** `coworkers`, `family`, `friends`, `local community`, `neighbors`,
`online community`, `other`, `other reddit user`, `partner/spouse`, `political tribe`

Mistral frequently hedged its answer with phrasing like:

> "coworkers or other (no possessive)"
> "coworkers or other (Sunpower is a company)"
> "friends or family (depending on the relationship)"
> "neighbors or local community (implied in 'find an electrician you trust')"

**Old behavior:** The original parser used longest-first substring matching on the
first sentence. For "coworkers or other (no possessive)", it matched "coworkers"
(length 9) before "other" (length 5).

**Actual intent:** When Mistral writes "X or Y", the second option (Y) is the
conservative fallback — the model is expressing uncertainty and defaulting to the
more general category. In the coworkers/other case, "other" is the correct label
because there is no possessive indicator.

**Scale:** 3,129 labels changed across 69,000 records (~4.5% of `1.3.1` labels).

---

### Bug 2: Special HTML Tokens

**Affected question:** Various
**Pattern:** Responses like `<s>NO</s>` where Mistral's tokenizer emitted
markdown/HTML wrapper tokens around the actual answer.

**Old behavior:** The substring matcher would find "no" inside the full string
`<s>no</s>`, which happens to work for most simple yes/no cases. However, for
cases like `<s>coworkers</s>` the HTML noise could interfere with option matching
order.

**Fix:** Strip all `<[^>]+>` tokens before matching.
**Scale:** 1 label changed (rare edge case).

---

### Bug 3: Missing `map_to` Application (Pre-existing checkpoint issue)

**Questions affected:** `1.1_gate`, `1.3.3_second_order`

The checkpoints stored raw option strings instead of the schema's `map_to` values:
- `1.1_gate`: stored `"no"` / `"yes"` instead of `"0"` / `"1"`
- `1.3.3_second_order`: stored `"none"` / `"weak"` / `"strong"` instead of `"0"` / `"1"` / `"2"`

This also revealed an existing parsing failure in `1.3.3_second_order` where the
old code returned the default `options[0]` ("none") for responses like `"weak."`,
incorrectly labeling ~68,998 records.

**Fix:** The re-parse script applies `map_to` consistently via the corrected parser.

---

## Parser Changes

### `_parse_single_choice` in `00_vLLM_hierarchical.py` and `shared_utilities.py`

**Before:**
```python
c = content.strip().lower()
# ... first-sentence match, then full-text fallback
```

**After:**
```python
c = content.strip()
# Strip HTML/XML tokens (e.g. <s>NO</s>)
c = re.sub(r'<[^>]+>', ' ', c)
c = ' '.join(c.split()).lower()

# Fix: "optionA or optionB" → prefer optionB (model's refined answer)
for o1 in sorted(options, key=len, reverse=True):
    for o2 in sorted(options, key=len, reverse=True):
        if o1.lower() != o2.lower():
            if (o1.lower() + ' or ' + o2.lower()) in first:
                m = o2.lower()
                return map_to.get(m, m) if map_to else m

# ... then existing first-sentence + full-text fallback
```

The "prefer Y" rule is applied **before** the longest-first substring match,
so it takes priority whenever the model writes an "X or Y" hedge.

**Known limitation:** "pro or against" would map to "against", which may not always
be correct. However, in practice this pattern is rare compared to "coworkers or other"
and the hedge interpretation (prefer the conservative/more general option) is still
the better default than the original bug.

---

## Re-parsing Procedure

A standalone script (`reparse_checkpoints.py`) was written to:
1. Load question schemas (options + map_to) from `schema/` directory
2. Iterate all 167 checkpoint files
3. Re-parse each `raw_responses[qid]` with the fixed parser
4. Update `answers[qid]` in-place if the new parse differs
5. Save corrected checkpoints back in-place

```
Total labels changed: 142,852
  HTML-token fixes:   1
  "X or Y" fixes:     3,219
Changes by question:
  1.1_gate:                  68,999  (map_to fix: "no"→"0", "yes"→"1")
  1.3.3_second_order:        68,998  (map_to + parsing failure fix)
  1.3.1_reference_group:      3,129  ("X or Y" fix)
  1.1.1_stance:               1,624  (various parsing improvements)
  1.3.1b_perceived_...:          45
  ... (27 other questions, 1–23 each)
```

---

## Before / After Accuracy

Verification is based on 7,355 samples labeled by GPT-OSS-20B (reasoning model)
as ground truth. `vllm_label` fields were updated from corrected checkpoints before
recomputing metrics.

| Question | Before Acc | After Acc | Before κ | After κ |
|---|---|---|---|---|
| `1.1_gate` | 0.565 | 0.565 | 0.132 | 0.132 |
| `1.1.1_stance` | 0.489 | 0.489 | 0.358 | 0.358 |
| `1.2.1_descriptive` | 0.504 | 0.504 | 0.258 | 0.258 |
| `1.2.2_injunctive` | 0.531 | 0.531 | 0.296 | 0.296 |
| `1.3.1_reference_group` | **0.286** | **0.352** | — | 0.227 |
| `1.3.1b_perceived_reference_stance` | 0.580 | 0.580 | 0.370 | 0.370 |
| `1.3.3_second_order` | — | 0.379 | — | 0.080 |
| **Overall** | **0.660** | **0.665** | **0.392** | **0.394** |

**Key improvement:** `1.3.1_reference_group` accuracy increased by **+6.6 percentage
points** (0.286 → 0.352) due to the "X or Y" fix correctly routing ambiguous
"coworkers or other" responses to "other".

---

## Files Changed

| File | Change |
|------|--------|
| `00_vLLM_hierarchical.py` | Fixed `_parse_single_choice`: HTML strip + "X or Y" rule |
| `shared_utilities.py` | Same fix applied |
| `paper4data/checkpoints/checkpoint_*.json` | Re-parsed `raw_responses` → updated `answers` (142,852 labels) |
| `paper4data/norms_labels_checkpoints_only.json` | Rebuilt from corrected checkpoints |
| `paper4data/00_verification_samples_ckpt.json` | 372 `vllm_label` fields updated |
| `paper4data/00_verification_results_ckpt.json` | Recomputed accuracy metrics |
| `docs/mistral_parsing.md` | This document |
