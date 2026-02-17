# Social Norms in Climate Discussions — Dashboard & Methodology

**Live dashboard:** [chowdhary-sandeep.github.io/climate_change__4housing_transport_food](https://chowdhary-sandeep.github.io/climate_change__4housing_transport_food/)
**File:** `temp.py` → generates `temp.html`

---

## 1. Introduction

Climate mitigation requires more than technological solutions—it depends on collective behavioural shifts shaped by social and cultural processes [1, 2]. Social norms—shared expectations about what is typical or approved within a group—have emerged as powerful levers for encouraging climate-relevant behaviours [3, 4]. Descriptive norms (what people *do*) and injunctive norms (what people *should* do) interact to shape energy conservation [5], dietary choices, and technology adoption [6]. Second-order normative beliefs—beliefs about what others *believe* others do—have proven critical predictors of behaviour change, sometimes more so than first-order norms [7]. Trending norms, where a behaviour is growing even if still a minority practice, can be communicated to accelerate adoption [8].

Three domains sit at the heart of individual-level climate mitigation: **transport** (electric vehicles), **food** (plant-based diets), and **housing** (residential solar photovoltaics). Together, these sectors account for a large share of household carbon footprints, and public discourse around them is rapidly evolving.

Yet most of what we know about climate norms comes from surveys and experiments [3, 5, 7]—instruments that are expensive, slow, and limited in temporal resolution [9]. Social media platforms, especially Reddit, host continuous, large-scale, naturalistic discussion where normative claims surface organically. No study has systematically extracted the *structure* of social norms from this discourse—measuring not just sentiment but the type of norm invoked, the reference group cited, and the stance expressed—across multiple climate-action sectors simultaneously.

### 1.1 Knowledge Gap

Traditional measures of societal readiness—public opinion polls [10], climate opinion maps [11], media content analyses [12], policy adoption indices [13], technology deployment rates [14]—can be slow, costly, or limited in temporal granularity. Social-media sentiment studies exist [15] but typically treat sentiment as a single dimension, ignoring the richer norm taxonomy that behavioural science has shown to matter (descriptive vs. injunctive, first- vs. second-order, reference group specificity). Furthermore, no prior work has compared the *frame-level* content of online discourse with established survey instruments to identify where public attention and survey assumptions diverge.

### 1.2 This Work

We present an AI-enabled pipeline to measure societal readiness for climate solutions by extracting a structured social norms taxonomy from large-scale Reddit discourse. Our contributions are:

1. **Hierarchical norm extraction** — Each comment is classified on 7 norm dimensions derived from IPCC AR6 Chapter 5 [2]: norm signal presence (gate), author stance, descriptive norm, injunctive norm, reference group, perceived reference stance, and second-order normative belief.
2. **Sector-specific survey frame labeling** — Comments are simultaneously labeled against sector-specific survey questions (13 food/diet factors, 6 EV factors, 10 solar factors) drawn from Gallup, Pew, Yale, and Green Fox Energy surveys [10, 11, 16], enabling direct comparison of Reddit discourse proportions with survey endorsement rates.
3. **Temporal trend analysis** — Year-stratified sampling (2010–2024) enables tracking of how norm types, stances, reference groups, and factor salience shift over a 15-year window.
4. **Verification pipeline** — A reasoning model (GPT-OSS-20B) re-labels 1,800 samples (50 per question) to quantify label reliability, with logprob-based confidence analysis.

---

## 2. Data

### 2.1 Reddit Corpus

We collected comments and submissions from **16 subreddits** spanning three behavioural sectors plus nine general climate subreddits. Posts were filtered using sector-specific keyword regex patterns (60 transport keywords, 31 housing keywords, 40 food keywords).

| Sector | Subreddits | Comments + Submissions | Keyword-matched |
|--------|-----------|----------------------:|----------------:|
| **Food** | r/vegan, r/veganarchism, r/vegancirclejerk | 13,226,049 | 2,535,937 |
| **Transport** | r/electricvehicles, r/ElectricScooters, r/Electricmotorcycles | 5,550,944 | 1,040,821 |
| **Housing** | r/solar | 681,926 | 126,481 |
| **Climate** (cross-sector) | r/climate, r/ClimateActionPlan, r/climatechange, r/ClimateChaos, r/climatedisalarm, r/ClimateMemes, r/ClimateOffensive, r/ClimateShitposting, r/climateskeptics | 607,244 | — |
| **Total** | **16 subreddits** | **20,066,163** | **3,703,239** |

### 2.2 Sampling Strategy

From the 3.7 million keyword-matched comments, we sampled **9,000 comments** (3,000 per sector) with **equal representation across years** (2010–2024). Food sector achieves a uniform 200 comments/year; transport and housing have slightly more in recent years (reflecting Reddit's growth) but maintain ≥57 comments even in the earliest year (2010).

### 2.3 Labeling Schema

Each of the 9,000 comments receives **two types of labels**:

**Social Norms (7 questions per comment, all sectors):**

| ID | Question | Response Options |
|----|----------|-----------------|
| 1.1_gate | Does the comment reference a social norm? | yes / no |
| 1.1.1_stance | Author's stance toward the climate action | pro / against / against particular but pro / neither-mixed / pro but lack of options |
| 1.2.1_descriptive | Descriptive norm present? | explicitly present / absent / unclear |
| 1.2.2_injunctive | Injunctive norm present? | present / absent / unclear |
| 1.3.1_reference_group | Which social group is referenced? | family / partner-spouse / friends / coworkers / neighbors / local community / political tribe / online community / other reddit user / other |
| 1.3.1b_perceived_stance | Stance attributed to the reference group | pro / against / neither-mixed |
| 1.3.3_second_order | Second-order normative belief | 0 (none) / 1 (weak) / 2 (strong) |

**Survey Frame Questions (sector-specific, binary yes/no):**

| Sector | # Questions | Source Surveys | Examples |
|--------|----------:|----------------|---------|
| Food | 13 | Gallup 2019; Yale & GMU 2020 | health motivation, animal welfare, taste, social prompting, identity |
| Transport | 6 | Pew 2024 | environment, purchase cost, operating cost, infrastructure, driving experience, reliability |
| Housing | 10 | Green Fox Energy 2025 | cost savings, confidence, environmental benefit, obsolescence, payback, installer trust |

**Total labels produced:** 9,000 × 7 norms + 3,000 × 13 food + 3,000 × 6 transport + 3,000 × 10 housing = **150,000 individual labels**.

---

## 3. Pipeline

### 3.1 Data Collection (`reddit_Paper4_EVs.ipynb`)

1. Load CSV exports of Reddit comments and submissions from 16 subreddits (via Pushshift/Arctic Shift archives).
2. Filter using sector-specific regex keyword matching (case-insensitive, morphological variants).
3. Deduplicate by comment body per sector.
4. Extract year from `created_utc` Unix timestamp.
5. Cache to `paper4data/sector_to_comments_cache.json` (3.7M records with `id`, `body`, `year`).

### 3.2 Hierarchical Labeling (`00_vLLM_hierarchical.py`)

1. Load comment cache; sample 3,000 per sector with equal yearly representation.
2. For each comment, query a local vLLM instance (Mistral-7B-Instruct) with:
   - **Norms questions**: Hierarchical cascade—gate question first, then stance, norms, reference group, second-order beliefs.
   - **Survey questions**: Sector-specific binary prompts drawn from `00_vllm_survey_question_final.json`.
3. Collect both the label and the **log-probability** (confidence score) for each response.
4. Apply safety net: recheck "against" stance labels for potential "pro but lack of options" misclassification.
5. Output: `paper4data/norms_labels.json` — per-sector arrays of `{comment, year, answers, logprobs}`.

### 3.3 Prompt Optimization (`00_prompt_optimization.md`)

Evidence-guided prompt refinement based on systematic analysis of verification mismatches:
- **Descriptive norm prompt**: Fast model was too restrictive—added explicit examples for self-reports, statistics, observed group behaviour.
- **Gate question prompt**: Fast model was too broad—excluded corporate mentions, technical facts, questions.
- **Perceived stance prompt**: Fast model defaulted to "neither/mixed"—added inference rules for tone, framing, sarcasm.
- **Reference group prompt**: Fast model over-assigned specific groups—emphasized possessive indicators (my/our) as requirement.

### 3.4 Verification (`00_LMstudio_verifier_v2.py`)

1. Sample 50 comments per question (1,800 total across 36 questions).
2. Re-label with a larger reasoning model (GPT-OSS-20B via LM Studio) using **identical prompts**.
3. Compute: accuracy, precision, recall, F1, Cohen's kappa, category estimation errors.
4. Output: `paper4data/00_verification_results.json` (aggregated metrics) and `paper4data/00_verification_samples.json` (sample-level data with both labels).

**Verification Results Summary:**

| Metric | Value |
|--------|------:|
| Mean accuracy | 86.1% |
| Std accuracy | 12.1% |
| Min accuracy | 56% |
| Max accuracy | 100% |
| Mean Cohen's κ | 0.318 |
| Total samples verified | 1,800 |
| Empty responses | 0% |

---

## 4. Dashboard

### 4.1 Data Sources

| File | Contents |
|------|----------|
| `paper4data/norms_labels.json` | Full labeled dataset: 9,000 records with `comment`, `year`, `answers` (7–21 labels), `logprobs` |
| `paper4data/00_verification_results.json` | Aggregated verification metrics by question |
| `paper4data/00_verification_samples.json` | 1,800 re-labeled samples with both model labels |
| `00_vllm_survey_question_final.json` | Survey question metadata with wordings and source surveys |

### 4.2 Libraries

| Library | Version | Role |
|---------|---------|------|
| **Plotly.js** | 2.27.0 | Gauge, treemap, Sankey, bar charts, scatter plots, stacked area charts |
| **D3.js** | v7 | Bubble pack layout, radial/spoke charts |

---

## 5. Methods & Computations

### M1. Year-Bucketed Proportion (%)

**Used by:** Temporal Stance charts (Tab 4), Temporal Norm Dimension charts (Tab 4), Temporal Survey Factor charts (Tab 4)

1. Iterate all records in `norms_labels.json[sector]`.
2. Filter: skip records with `year < 2010` or `year is None`.
3. For each year in `[2010..2024]`, count occurrences of each category value for the target field.
4. Compute: `proportion = count_of_category / total_in_year × 100` (rounded to 1 decimal).
5. If a year has 0 total records, proportion = 0.

| Target field | Categories |
|---|---|
| `1.1.1_stance` (author stance) | pro, against, against particular but pro, neither/mixed, pro but lack of options |
| `1.2.1_descriptive` | explicitly present, absent, unclear |
| `1.2.2_injunctive` | present, absent, unclear |
| `1.3.3_second_order` | 2 (strong), 1 (weak), 0 (none) |
| `1.3.1b_perceived_reference_stance` | pro, against, neither/mixed |
| Survey questions (per `qid`) | Binary: "yes" vs not-yes → single `% yes` per year |

---

### M2. Animated Stacked Area Chart (Plotly)

**Used by:** All 9 Temporal charts in Tab 4

1. Takes pre-computed year-bucketed proportions from M1.
2. Creates one `scatter` trace per category with `mode:'lines'`, `stackgroup:'one'`.
3. X-axis = years `[2010..2024]`, Y-axis = percentage `[0..100]`.
4. Colors: fixed palette for stance/norms, rotating 13-colour palette for survey.

---

### M3. Animated Horizontal Bar Chart (Plotly)

**Used by:** Accuracy by Question (Tab 5), Top Estimation Errors (Tab 5), High-Confidence Accuracy (Tab 5)

| Chart | Source | Computation |
|---|---|---|
| **Accuracy by Question** | `verification_results.by_task.{norms,survey}.{qid}.accuracy` | Sorted ascending. Colour by threshold (green ≥85%, orange 70–85%, red <70%). |
| **Top Estimation Errors** | `verification_results...category_estimation.{cat}.estimation_error` | Filter |error| ≥ 2pp, sort descending, top 40. |
| **High-Conf Accuracy** | Derived from verification samples | Per-question accuracy restricted to samples with confidence > 0.9. |

---

### M4. Gauge Chart — Norm Signal Prevalence (Tab 1)

**Tool:** Plotly pie chart with `hole: 0.65`, clipped to top half.

**Data:** Hardcoded sector percentages: Food 40.2%, Transport 11.6%, Housing 16.2%.

**Computation:** Average = `(40.2 + 11.6 + 16.2) / 3 = 22.7%`. From gate question `1.1_gate` across each sector's 3,000 comments.

---

### M5. Confidence Analysis (Tab 5)

1. For each verification sample, extract logprob → convert to probability: `conf = exp(logprob)`.
2. Aggregate: mean confidence for matched vs mismatched samples, binned mismatch rates, per-question high-confidence accuracy.

---

### M6. Sankey Diagram — Norm Classification Flow (Tab 1)

**Structure:** 19 nodes (3 sectors + 4 dimensions + 12 categories). 48 links (16 per sector).

**Interaction:** Sector toggle buttons highlight one sector's flows (0.45 opacity) and dim others (0.06). Category labels update to show per-sector counts.

---

### M7. Treemap — Author Stance (Tab 2)

**Hierarchy:** Root → Sector → Stance. Colours: green (pro) → muted green → neutral → orange → red (against).

| Sector | Pro | Against | Against part. but pro | Neither/Mixed | Pro but lack of options |
|---|---|---|---|---|---|
| Food | 735 | 252 | 83 | 1,404 | 526 |
| Transport | 372 | 274 | 66 | 1,754 | 534 |
| Housing | 531 | 243 | 69 | 1,555 | 602 |

---

### M8. Bubble Pack — Reference Groups (Tab 1)

**Tool:** D3 force simulation. Radius: `r = sqrt(count) × 2.2`. 9 categories across 3 sectors.

---

### M9. Radial / Spoke Charts — Survey Factors (Tab 3)

**Tool:** D3 `d3.arc()`. Spokes show `% yes` for each factor.

| Sector | # Factors | Max scale |
|---|---:|---:|
| Food | 13 | 25% |
| Transport | 6 | 10% |
| Housing | 10 | 15% |

---

### M10. Verification Stat Boxes (Tab 5)

| Stat | Source | Colour logic |
|---|---|---|
| Accuracy | `summary.mean_accuracy` | Green ≥85%, else orange |
| Cohen's κ | `summary.mean_kappa` | Red <0.4, else green |
| No Response | `summary.empty_response_pct` | Green |
| Total Samples | `summary.total_samples` | Neutral |
| Conf (Match) | Mean exp(logprob) matched | Green |
| Conf (Mismatch) | Mean exp(logprob) mismatched | Orange |
| Acc (Conf>0.9) | High-conf accuracy | Green ≥90%, else orange |
| High-Conf Samples | Count conf > 0.9 | Neutral |

---

### M11. Examples Tab (Tab 6)

For each of 11 questions, show up to 4 mismatches and 2 matches from 50 verification samples per question. Low-accuracy questions (<70%) expanded by default.

---

## 6. Tab → Plot → Method Reference

| Tab | Plot | Method(s) |
|-----|------|-----------|
| 1. Norms | Gauge (norm signal %) | M4 |
| 1. Norms | Sankey (classification flow) | M6 |
| 1. Norms | Bubble pack (reference groups) | M8 |
| 2. Author Stance | Treemap | M7 |
| 3. Factors & Barriers | Radial charts (×3 sectors) | M9 |
| 4. Temporal | Stance over time (×3) | M1 → M2 |
| 4. Temporal | Norm dimensions over time (×3, togglable) | M1 → M2 |
| 4. Temporal | Survey factors over time (×3) | M1 → M2 |
| 5. Verification | Stat boxes | M10 |
| 5. Verification | Accuracy by question | M3 |
| 5. Verification | Top estimation errors | M3 |
| 5. Verification | Confidence bin bar + scatter | M5 |
| 5. Verification | High-confidence accuracy | M3 + M5 |
| 6. Examples | Sample comparisons | M11 |

---

## References

[1] IPCC, "Climate Change 2022: Mitigation of Climate Change," Chapter 5 — Demand, Services and Social Aspects of Mitigation, 2022.

[2] S. Eker *et al.*, "Social and cultural processes in climate mitigation," in *IPCC AR6 WG3 Chapter 5*, 2022.

[3] R. Cialdini, R. Reno, and C. Kallgren, "A focus theory of normative conduct: Recycling the concept of norms to reduce littering in public places," *Journal of Personality and Social Psychology*, vol. 58, pp. 1015–1026, 1990.

[4] D. Miller and D. Prentice, "Changing norms to change behavior," *Annual Review of Psychology*, vol. 67, pp. 339–361, 2016.

[5] J. Bonan, C. Cattaneo, G. d'Adda, and M. Tavoni, "The interaction of descriptive and injunctive norms in promoting energy conservation," *Nature Energy*, vol. 5, pp. 900–909, 2020.

[6] W. Abrahamse and L. Steg, "Social influence approaches to encourage resource conservation: A meta-analysis," *Global Environmental Change*, vol. 23, pp. 1773–1785, 2013.

[7] J. Jachimowicz, O. Hauser, J. O'Brien, E. Sherman, and A. Galinsky, "The critical role of second-order normative beliefs in predicting energy conservation," *Nature Human Behaviour*, vol. 2, pp. 757–764, 2018.

[8] C. Mortensen, R. Neel, R. Cialdini, C. Jaeger, and R. Jacobson, "Trending norms: A lever for encouraging behaviors performed by the minority," *Social Psychological and Personality Science*, vol. 10, pp. 201–210, 2019.

[9] S. Chowdhary and S. Eker, "Societal Readiness for Climate Solutions: Insights from Reddit Discourse on Adoption of Electric Vehicles, Plant-based Diets and Solar Photovoltaics," *International Institute for Applied Systems Analysis*, working paper, 2025.

[10] Pew Research Center, "Climate change public opinion survey," 2022/2024.

[11] Yale Program on Climate Change Communication, "Yale climate opinion maps," 2023.

[12] M. T. Boykoff and T. J. Roberts, "Media coverage of climate change: meta-analysis of content and trends," *Global Environmental Change*, vol. 17, no. 5, pp. 1–14, 2007.

[13] OECD, "Climate policy database," 2023.

[14] IEA, "World energy outlook," 2023.

[15] A. Carlyle, J. Smith, and M. Lee, "Automated sentiment analysis of climate discourse on social media," *Environmental Communication*, vol. 15, no. 6, pp. 738–758, 2021.

[16] Green Fox Energy, "Residential solar adoption survey," 2025.

[17] W. Pearce, S. Niederer, S. O'Neill, and M. Hurlstone, "Climate change on social media: the role of platforms in public engagement," *Wiley Interdisciplinary Reviews: Climate Change*, vol. 10, p. e604, 2019.

[18] C. Sunstein, *How Change Happens*, MIT Press, 2019.
