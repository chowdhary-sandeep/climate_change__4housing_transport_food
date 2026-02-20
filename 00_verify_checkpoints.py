"""
00_verify_checkpoints.py

Runs 00_LMstudio_verifier_v2 logic on norms_labels_full.json (123k checkpoint data).
- 100 comments per question/label (or max available)
- Random sample from full checkpoint pool
- Outputs: paper4data/00_verification_results_ckpt.json
           paper4data/00_verification_samples_ckpt.json

Starts with a 5-label timing benchmark to estimate total runtime.
"""

import json, os, random, asyncio, aiohttp, time
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from tqdm import tqdm

from shared_utilities import (
    load_api_config,
    call_llm_single_choice,
    NORMS_SYSTEM, SURVEY_SYSTEM,
    NORMS_QUESTIONS, SURVEY_QUESTIONS_BY_SECTOR,
)

# ── Config ────────────────────────────────────────────────────────────────────
API_CONFIG_PATH        = "local_LLM_api_from_vLLM.json"
NORMS_LABELS_PATH      = "paper4data/norms_labels_full.json"
OUTPUT_PATH            = "paper4data/00_verification_results_ckpt.json"
SAMPLES_OUTPUT_PATH    = "paper4data/00_verification_samples_ckpt.json"
SAMPLES_PER_QUESTION   = 200
MAX_CONCURRENT         = 4     # conservative for 20B reasoning model
REASONING_EFFORT       = "medium"
RANDOM_SEED            = 42
BENCHMARK_N            = 5     # labels to time before estimating full run


def get_model_config():
    config = load_api_config(API_CONFIG_PATH)
    key    = config.get("verification_model_key", "7")
    mc     = config["available_models"][key]
    return mc["base_url"], mc["model_name"]

BASE_URL, MODEL_NAME = get_model_config()


# ── Sampling ──────────────────────────────────────────────────────────────────
def sample_for_verification(data, n=SAMPLES_PER_QUESTION, seed=RANDOM_SEED):
    random.seed(seed)
    sampled = {"norms": defaultdict(list), "survey": defaultdict(list)}

    all_comments = []
    for sector, comments in data.items():
        for c in comments:
            if "answers" in c:
                all_comments.append({**c, "sector": sector})

    for q in NORMS_QUESTIONS:
        qid = q["id"]
        eligible = [c for c in all_comments if qid in c["answers"]]
        sampled["norms"][qid] = random.sample(eligible, min(n, len(eligible)))

    for sector, qs in SURVEY_QUESTIONS_BY_SECTOR.items():
        sc = [c for c in all_comments if c["sector"] == sector]
        for q in qs:
            qid = q["id"]
            eligible = [c for c in sc if qid in c["answers"]]
            sampled["survey"][qid] = random.sample(eligible, min(n, len(eligible)))

    return sampled


# ── Relabeling ────────────────────────────────────────────────────────────────
async def label_one(session, semaphore, comment, question, system_prompt):
    async with semaphore:
        reasoning_label, raw, _ = await call_llm_single_choice(
            session=session,
            text=comment["comment"],
            question=question,
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            sector=comment.get("sector"),
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=1024,
            reasoning_effort=REASONING_EFFORT,
        )
        return {
            **comment,
            "question_id": question["id"],
            "vllm_label": comment["answers"].get(question["id"], ""),
            "reasoning_label": reasoning_label,
            "raw_reasoning_response": raw,
        }


async def relabel_all(sampled, max_concurrent=MAX_CONCURRENT):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks_list = []
    meta = []

    for qid, comments in sampled["norms"].items():
        q = next((x for x in NORMS_QUESTIONS if x["id"] == qid), None)
        if not q: continue
        for c in comments:
            tasks_list.append((c, q, NORMS_SYSTEM))
            meta.append(("norms", qid))

    for qid, comments in sampled["survey"].items():
        q = None
        for qs in SURVEY_QUESTIONS_BY_SECTOR.values():
            q = next((x for x in qs if x["id"] == qid), None)
            if q: break
        if not q: continue
        for c in comments:
            tasks_list.append((c, q, SURVEY_SYSTEM))
            meta.append(("survey", qid))

    total = len(tasks_list)
    print(f"\nTotal labels to verify: {total}")
    print(f"Model: {MODEL_NAME}  |  Concurrency: {max_concurrent}\n")

    results = []
    start = time.time()

    async with aiohttp.ClientSession() as session:
        coros = [label_one(session, semaphore, c, q, sys) for c, q, sys in tasks_list]

        with tqdm(total=total, unit="labels",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for i in range(0, total, 50):
                batch = coros[i:i+50]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)
                pbar.update(len(batch))
                elapsed = time.time() - start
                rate = len(results) / elapsed if elapsed > 0 else 0
                eta = (total - len(results)) / rate if rate > 0 else 0
                pbar.set_postfix_str(f"{rate:.1f}/s | ETA {eta/60:.1f}min")

    relabeled = {"norms": defaultdict(list), "survey": defaultdict(list)}
    for i, r in enumerate(results):
        if isinstance(r, Exception): continue
        t, qid = meta[i]
        relabeled[t][qid].append(r)

    return {"norms": dict(relabeled["norms"]), "survey": dict(relabeled["survey"])}


# ── Timing benchmark ──────────────────────────────────────────────────────────
async def benchmark(sampled, n=BENCHMARK_N):
    """Time n labels to estimate full run duration."""
    print(f"\n[BENCHMARK] Timing {n} labels to estimate full run...")
    # grab first n labels from norms
    bench_items = []
    for qid, comments in sampled["norms"].items():
        for c in comments:
            bench_items.append((c, next(q for q in NORMS_QUESTIONS if q["id"] == qid), NORMS_SYSTEM))
            if len(bench_items) >= n:
                break
        if len(bench_items) >= n:
            break

    semaphore = asyncio.Semaphore(1)   # serial for accurate timing
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [label_one(session, semaphore, c, q, sys) for c, q, sys in bench_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start
    ok = sum(1 for r in results if not isinstance(r, Exception))

    # total work
    total_norms  = sum(len(v) for v in sampled["norms"].values())
    total_survey = sum(len(v) for v in sampled["survey"].values())
    total = total_norms + total_survey

    rate_serial = ok / elapsed if elapsed > 0 else 0
    rate_parallel = rate_serial * MAX_CONCURRENT  # approx with concurrency
    eta_s = total / rate_parallel if rate_parallel > 0 else 0

    print(f"  Benchmark: {ok}/{n} OK in {elapsed:.1f}s -> {rate_serial:.2f} labels/s serial")
    print(f"  With concurrency={MAX_CONCURRENT}: ~{rate_parallel:.1f} labels/s")
    print(f"  Total labels: {total} ({total_norms} norms + {total_survey} survey)")
    print(f"  Estimated time: {eta_s/60:.0f} min  ({eta_s/3600:.1f} hrs)")
    print()
    return elapsed / ok if ok > 0 else None


# ── Metrics ───────────────────────────────────────────────────────────────────
def normalize(s): return str(s).lower().strip()

def calc_metrics(comments, qid, short_form=None):
    valid = [c for c in comments
             if normalize(c["reasoning_label"]) and normalize(c["vllm_label"])]
    n_empty = len(comments) - len(valid)
    if not valid:
        return {"question_id": qid, "n_samples_valid": 0,
                "n_empty_responses": n_empty, "accuracy": 0.0}

    y_true = [normalize(c["reasoning_label"]) for c in valid]
    y_pred = [normalize(c["vllm_label"])      for c in valid]
    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    try:    kappa = cohen_kappa_score(y_true, y_pred)
    except: kappa = 0.0

    cat_est = {}
    for i, lbl in enumerate(labels):
        rt = Counter(y_true)[lbl] / len(y_true) * 100
        ft = Counter(y_pred)[lbl] / len(y_pred) * 100
        diff = ft - rt
        cat_est[lbl] = {"reasoning_pct": round(rt,1), "fast_model_pct": round(ft,1),
                        "error": round(diff,1),
                        "type": "accurate" if abs(diff)<=2 else ("over" if diff>2 else "under")}

    return {
        "question_id": qid,
        "question_short_form": short_form or qid,
        "n_samples_total": len(comments),
        "n_samples_valid": len(valid),
        "n_empty_responses": n_empty,
        "empty_pct": round(n_empty/len(comments)*100,1),
        "accuracy": round(acc, 3),
        "macro_precision": round(float(np.mean(prec)), 3),
        "macro_recall":    round(float(np.mean(rec)),  3),
        "macro_f1":        round(float(np.mean(f1)),   3),
        "cohen_kappa":     round(kappa, 3),
        "category_estimation": cat_est,
    }

def calc_all_metrics(relabeled):
    out = {"norms": {}, "survey": {}}
    for qid, coms in relabeled["norms"].items():
        out["norms"][qid] = calc_metrics(coms, qid)
    for qid, coms in relabeled["survey"].items():
        out["survey"][qid] = calc_metrics(coms, qid, qid.replace("_"," "))

    accs   = [m["accuracy"]     for t in out.values() for m in t.values() if m.get("accuracy")]
    kappas = [m["cohen_kappa"]  for t in out.values() for m in t.values()
              if m.get("cohen_kappa") and not np.isnan(m["cohen_kappa"])]
    tot_n  = sum(m.get("n_samples_total",0) for t in out.values() for m in t.values())
    tot_e  = sum(m.get("n_empty_responses",0) for t in out.values() for m in t.values())

    summary = {
        "total_questions": len(accs),
        "norms_questions": len(out["norms"]),
        "survey_questions": len(out["survey"]),
        "total_samples": tot_n,
        "total_empty": tot_e,
        "empty_pct": round(tot_e/tot_n*100,1) if tot_n else 0,
        "empty_response_pct": round(tot_e/tot_n*100,1) if tot_n else 0,
        "mean_accuracy": round(float(np.mean(accs)),3) if accs else 0,
        "std_accuracy":  round(float(np.std(accs)),3)  if accs else 0,
        "min_accuracy":  round(float(np.min(accs)),3)  if accs else 0,
        "max_accuracy":  round(float(np.max(accs)),3)  if accs else 0,
        "mean_kappa":    round(float(np.mean(kappas)),3) if kappas else 0,
        "data_source": NORMS_LABELS_PATH,
        "samples_per_question": SAMPLES_PER_QUESTION,
        "model": MODEL_NAME,
    }
    return {"summary": summary, "by_task": out}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("\n" + "="*70)
    print("CHECKPOINT VERIFICATION  -  GPT-OSS-20B  -  200 samples/question  -  reasoning=medium")
    print(f"Data: {NORMS_LABELS_PATH}")
    print("="*70)

    print("\n[1/5] Loading 123k checkpoint data...")
    with open(NORMS_LABELS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    total = sum(len(v) for v in data.values())
    print(f"  {total:,} labeled comments across {len(data)} sectors")

    print(f"\n[2/5] Sampling {SAMPLES_PER_QUESTION}/question from checkpoint pool...")
    sampled = sample_for_verification(data, SAMPLES_PER_QUESTION, RANDOM_SEED)
    n_norms  = sum(len(v) for v in sampled["norms"].values())
    n_survey = sum(len(v) for v in sampled["survey"].values())
    print(f"  {n_norms} norms labels + {n_survey} survey labels = {n_norms+n_survey} total")

    print("\n[3/5] Benchmarking model speed...")
    await benchmark(sampled, BENCHMARK_N)

    # auto-confirmed by user

    print("\n[4/5] Verifying all labels...")
    relabeled = await relabel_all(sampled, MAX_CONCURRENT)

    print("\n[5/5] Computing metrics & saving...")
    results = calc_all_metrics(relabeled)
    s = results["summary"]
    print(f"  Mean accuracy : {s['mean_accuracy']:.3f}  (+/-{s['std_accuracy']:.3f})")
    print(f"  Mean kappa    : {s['mean_kappa']:.3f}")
    print(f"  Empty rate    : {s['empty_pct']:.1f}%")

    os.makedirs("paper4data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(SAMPLES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(relabeled, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"  Saved: {SAMPLES_OUTPUT_PATH}")

    # Summary table
    print("\n" + "-"*70)
    print(f"{'Question':<38} {'Acc':>6} {'k':>7} {'N':>5}  {'Type'}")
    print("-"*70)
    for qid, m in sorted(results["by_task"]["norms"].items()):
        kstr = f"{m['cohen_kappa']:.3f}" if not np.isnan(m.get('cohen_kappa',0)) else "n/a"
        print(f"{qid[:37]:<38} {m['accuracy']:>6.3f} {kstr:>7} {m['n_samples_valid']:>5}  norms")
    for qid, m in sorted(results["by_task"]["survey"].items()):
        kstr = f"{m['cohen_kappa']:.3f}" if not np.isnan(m.get('cohen_kappa',0)) else "n/a"
        print(f"{qid[:37]:<38} {m['accuracy']:>6.3f} {kstr:>7} {m['n_samples_valid']:>5}  survey")
    print("-"*70)
    print(f"\nDone. Results in {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
