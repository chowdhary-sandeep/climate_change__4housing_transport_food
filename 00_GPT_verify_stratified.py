"""
00_GPT_verify_stratified.py

Verification stratified per (question x label x sector).
- 100 samples per (qid, label, sector) cell, or max available
- Ensures every Examples tab cell has enough verified samples for 2-agree + 1-disagree
- reasoning_effort=medium, concurrency=4
- Incremental checkpointing: resumes if interrupted
- Outputs: paper4data/00_verification_results_v2.json
           paper4data/00_verification_samples_v2.json
"""

import json, os, random, asyncio, aiohttp, time
from collections import defaultdict, Counter
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
API_CONFIG_PATH      = "schema/local_LLM_api_from_vLLM.json"
NORMS_LABELS_PATH    = "paper4data/norms_labels_checkpoints_only.json"
OUTPUT_PATH          = "paper4data/00_verification_results_v2.json"
SAMPLES_OUTPUT_PATH  = "paper4data/00_verification_samples_v2.json"
CHECKPOINT_PATH      = "paper4data/00_GPT_verify_stratified_checkpoint.json"
SAMPLES_PER_CELL     = 50
MAX_CONCURRENT       = 4
REASONING_EFFORT     = "medium"
RANDOM_SEED          = 42


def get_model_config():
    config = load_api_config(API_CONFIG_PATH)
    key = config.get("verification_model_key", "7")
    mc  = config["available_models"][key]
    return mc["base_url"], mc["model_name"]

BASE_URL, MODEL_NAME = get_model_config()


# ── Sampling: per (qid, label, sector) ───────────────────────────────────────
def build_sample_plan(data, seed=RANDOM_SEED, n=SAMPLES_PER_CELL):
    """
    Returns list of (task_type, qid, question, system_prompt, sector, label, [comment_records])
    """
    random.seed(seed)
    plan = []

    # Norms: apply to all sectors
    for q in NORMS_QUESTIONS:
        qid = q["id"]
        for sec, recs in data.items():
            eligible = [r for r in recs if "answers" in r and qid in r["answers"]]
            by_label = defaultdict(list)
            for r in eligible:
                lbl = str(r["answers"][qid]).strip()
                if lbl:
                    by_label[lbl].append(r)
            for lbl, pool in by_label.items():
                sampled = random.sample(pool, min(n, len(pool)))
                plan.append(("norms", qid, q, NORMS_SYSTEM, sec, lbl, sampled))

    # Survey: sector-specific
    for sec, qs in SURVEY_QUESTIONS_BY_SECTOR.items():
        recs = data.get(sec, [])
        for q in qs:
            qid = q["id"]
            eligible = [r for r in recs if "answers" in r and qid in r["answers"]]
            by_label = defaultdict(list)
            for r in eligible:
                lbl = str(r["answers"][qid]).strip()
                if lbl:
                    by_label[lbl].append(r)
            for lbl, pool in by_label.items():
                sampled = random.sample(pool, min(n, len(pool)))
                plan.append(("survey", qid, q, SURVEY_SYSTEM, sec, lbl, sampled))

    return plan


# ── Single label call ─────────────────────────────────────────────────────────
async def label_one(session, semaphore, rec, question, system_prompt, sector, qid, vllm_label):
    async with semaphore:
        reasoning_label, raw, _ = await call_llm_single_choice(
            session=session,
            text=rec["comment"],
            question=question,
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            sector=sector,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=1024,
            reasoning_effort=REASONING_EFFORT,
        )
        return {
            "comment":          rec["comment"],
            "sector":           sector,
            "year":             rec.get("year"),
            "answers":          rec.get("answers", {}),
            "logprobs":         rec.get("logprobs", {}),
            "question_id":      qid,
            "vllm_label":       vllm_label,
            "reasoning_label":  reasoning_label,
            "raw_reasoning_response": raw,
        }


# ── Run one cell (qid, label, sector) ────────────────────────────────────────
async def run_cell(session, semaphore, task_type, qid, question, system_prompt,
                   sector, label, recs):
    coros = [label_one(session, semaphore, r, question, system_prompt,
                       sector, qid, str(r["answers"][qid]).strip())
             for r in recs]
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]


# ── Metrics per question ──────────────────────────────────────────────────────
def normalize(s): return str(s).lower().strip()

def calc_metrics(comments, qid):
    valid = [c for c in comments
             if normalize(c.get("reasoning_label","")) and normalize(c.get("vllm_label",""))]
    n_empty = len(comments) - len(valid)
    if not valid:
        return {"question_id": qid, "n_samples_valid": 0,
                "n_empty_responses": n_empty, "accuracy": 0.0,
                "empty_response_pct": round(n_empty/max(len(comments),1)*100,1)}

    y_true = [normalize(c["reasoning_label"]) for c in valid]
    y_pred = [normalize(c["vllm_label"])      for c in valid]
    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    try:    kappa = cohen_kappa_score(y_true, y_pred)
    except: kappa = 0.0

    cat_est = {}
    for i, lbl in enumerate(labels):
        rt = Counter(y_true)[lbl] / len(y_true) * 100
        ft = Counter(y_pred)[lbl] / len(y_pred) * 100
        diff = ft - rt
        cat_est[lbl] = {
            "reasoning_pct": round(rt,1), "fast_model_pct": round(ft,1),
            "error": round(diff,1),
            "type": "accurate" if abs(diff)<=2 else ("over" if diff>2 else "under")
        }
    return {
        "question_id":      qid,
        "n_samples_total":  len(comments),
        "n_samples_valid":  len(valid),
        "n_empty_responses": n_empty,
        "empty_response_pct": round(n_empty/len(comments)*100,1),
        "accuracy":         round(acc,3),
        "macro_precision":  round(float(np.mean(prec)),3),
        "macro_recall":     round(float(np.mean(rec)),3),
        "macro_f1":         round(float(np.mean(f1)),3),
        "cohen_kappa":      round(kappa,3),
        "category_estimation": cat_est,
    }

def calc_all_metrics(relabeled):
    out = {"norms": {}, "survey": {}}
    for tt in ["norms","survey"]:
        for qid, coms in relabeled[tt].items():
            out[tt][qid] = calc_metrics(coms, qid)

    accs   = [m["accuracy"]    for t in out.values() for m in t.values() if m.get("accuracy")]
    kappas = [m["cohen_kappa"] for t in out.values() for m in t.values()
              if m.get("cohen_kappa") and not np.isnan(m.get("cohen_kappa",float("nan")))]
    tot_n  = sum(m.get("n_samples_total",0) for t in out.values() for m in t.values())
    tot_e  = sum(m.get("n_empty_responses",0) for t in out.values() for m in t.values())

    return {
        "summary": {
            "total_questions":   len(accs),
            "norms_questions":   len(out["norms"]),
            "survey_questions":  len(out["survey"]),
            "total_samples":     tot_n,
            "total_empty":       tot_e,
            "empty_response_pct": round(tot_e/tot_n*100,1) if tot_n else 0,
            "mean_accuracy":     round(float(np.mean(accs)),3)  if accs   else 0,
            "std_accuracy":      round(float(np.std(accs)),3)   if accs   else 0,
            "min_accuracy":      round(float(np.min(accs)),3)   if accs   else 0,
            "max_accuracy":      round(float(np.max(accs)),3)   if accs   else 0,
            "mean_kappa":        round(float(np.mean(kappas)),3) if kappas else 0,
            "data_source":       NORMS_LABELS_PATH,
            "samples_per_cell":  SAMPLES_PER_CELL,
            "model":             MODEL_NAME,
            "sampling":          "per (question x label x sector)",
        },
        "by_task": out,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("\n" + "="*70)
    print("PER-LABEL VERIFICATION  -  GPT-OSS-20B  -  100 samples/(q x label x sector)")
    print(f"Model: {MODEL_NAME} | reasoning_effort={REASONING_EFFORT} | concurrency={MAX_CONCURRENT}")
    print("="*70)

    # Load data
    print("\n[1] Loading 123k checkpoint data...")
    with open(NORMS_LABELS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {sum(len(v) for v in data.values()):,} records")

    # Load checkpoint if exists (resume support)
    done_keys = set()
    relabeled = {"norms": defaultdict(list), "survey": defaultdict(list)}
    if os.path.exists(CHECKPOINT_PATH):
        print(f"  Resuming from checkpoint: {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            ckpt = json.load(f)
        for tt in ["norms","survey"]:
            for qid, samples in ckpt.get(tt, {}).items():
                relabeled[tt][qid].extend(samples)
        done_keys = set(ckpt.get("done_keys", []))
        print(f"  Already done: {len(done_keys)} cells")

    # Build plan
    print("\n[2] Building sample plan...")
    plan = build_sample_plan(data, RANDOM_SEED, SAMPLES_PER_CELL)
    total_cells = len(plan)
    total_labels = sum(len(cell[-1]) for cell in plan)
    pending = [(tt,qid,q,sp,sec,lbl,recs) for (tt,qid,q,sp,sec,lbl,recs) in plan
               if f"{tt}:{qid}:{lbl}:{sec}" not in done_keys]
    pending_labels = sum(len(cell[-1]) for cell in pending)

    print(f"  Total cells: {total_cells} | Total labels: {total_labels:,}")
    print(f"  Pending: {len(pending)} cells | {pending_labels:,} labels")
    if pending_labels > 0:
        print(f"  Est. time @ 1.2s/label concurrency={MAX_CONCURRENT}: ~{pending_labels*1.2/MAX_CONCURRENT/60:.0f} min")

    if not pending:
        print("  Nothing to do - all cells already verified!")
    else:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        start = time.time()
        done = 0

        async with aiohttp.ClientSession() as session:
            with tqdm(total=pending_labels, unit="labels",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

                for (tt, qid, q, sp, sec, lbl, recs) in pending:
                    cell_key = f"{tt}:{qid}:{lbl}:{sec}"
                    cell_results = await run_cell(session, semaphore, tt, qid, q, sp, sec, lbl, recs)
                    relabeled[tt][qid].extend(cell_results)
                    done_keys.add(cell_key)
                    done += len(cell_results)
                    pbar.update(len(recs))

                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (pending_labels - done) / rate if rate > 0 else 0
                    pbar.set_postfix_str(f"{rate:.1f}/s | ETA {eta/60:.0f}min")

                    # Save checkpoint after each cell
                    ckpt_data = {"done_keys": list(done_keys),
                                 "norms": {k: v for k,v in relabeled["norms"].items()},
                                 "survey": {k: v for k,v in relabeled["survey"].items()}}
                    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                        json.dump(ckpt_data, f, ensure_ascii=False)

    # Final metrics + save
    print("\n[3] Computing metrics...")
    relabeled_final = {
        "norms":  dict(relabeled["norms"]),
        "survey": dict(relabeled["survey"]),
    }
    results = calc_all_metrics(relabeled_final)
    s = results["summary"]
    print(f"  Mean accuracy : {s['mean_accuracy']:.3f} (+/-{s['std_accuracy']:.3f})")
    print(f"  Mean kappa    : {s['mean_kappa']:.3f}")
    print(f"  Total samples : {s['total_samples']:,}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(SAMPLES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(relabeled_final, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"  Saved: {SAMPLES_OUTPUT_PATH}")

    # Summary table
    print("\n" + "-"*70)
    print(f"{'Question':<38} {'Acc':>6} {'k':>7} {'N':>6}  Type")
    print("-"*70)
    for qid, m in sorted(results["by_task"]["norms"].items()):
        kv = m.get("cohen_kappa", 0)
        kstr = f"{kv:.3f}" if not np.isnan(kv) else "n/a"
        print(f"{qid[:37]:<38} {m['accuracy']:>6.3f} {kstr:>7} {m['n_samples_valid']:>6}  norms")
    for qid, m in sorted(results["by_task"]["survey"].items()):
        kv = m.get("cohen_kappa", 0)
        kstr = f"{kv:.3f}" if not np.isnan(kv) else "n/a"
        print(f"{qid[:37]:<38} {m['accuracy']:>6.3f} {kstr:>7} {m['n_samples_valid']:>6}  survey")
    print("-"*70)
    print(f"\nDone. Results -> {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
