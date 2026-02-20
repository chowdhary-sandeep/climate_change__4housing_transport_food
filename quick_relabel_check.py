"""
quick_relabel_check.py

Re-label a sample of old checkpoint records with the new first-sentence parser.
Shows old_label vs new_label for the most bug-prone questions.
Focus: coworkers, stance, injunctive
"""

import json, random, asyncio, aiohttp, sys
from shared_utilities import (
    load_api_config, call_llm_single_choice,
    NORMS_SYSTEM, NORMS_QUESTIONS,
)

random.seed(42)
API_CONFIG_PATH = "schema/local_LLM_api_from_vLLM.json"

# Load API config - use labeling model (key "1" = Mistral 7B)
config = load_api_config(API_CONFIG_PATH)
key = config.get("default_model_key", "1")
mc = config["available_models"][key]
BASE_URL = mc["base_url"]
MODEL_NAME = mc["model_name"]

# Questions to test (the ones with known bugs)
TARGET_QIDS = [
    "1.3.1_reference_group",   # coworkers bug
    "1.1.1_stance",            # against/pro in explanation
    "1.2.2_injunctive",        # absent/present in explanation
    "1.1_gate",                # yes/no in explanation
]

MAX_CONCURRENT = 8
N_SAMPLE = 40  # per focus area


def load_old_checkpoints(n_ckpts=5):
    """Load first N old checkpoints."""
    import os
    files = sorted([f for f in os.listdir("paper4data/checkpoints_old") if f.endswith(".json")])[:n_ckpts]
    all_recs = {"transport": [], "housing": [], "food": []}
    for fname in files:
        with open(f"paper4data/checkpoints_old/{fname}", encoding="utf-8") as f:
            data = json.load(f)
        for sector, recs in data.items():
            all_recs[sector].extend(recs)
    return all_recs


async def relabel_one(session, semaphore, rec, sector, q):
    async with semaphore:
        ans, raw, _ = await call_llm_single_choice(
            session=session,
            text=rec["comment"],
            question=q,
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            sector=sector,
            system_prompt=NORMS_SYSTEM,
            temperature=0.1,
            max_tokens=256,
        )
        return {
            "sector": sector,
            "comment": rec["comment"][:300],
            "old_label": rec["answers"].get(q["id"], "?"),
            "new_label": ans,
            "raw": raw[:200],
            "match": rec["answers"].get(q["id"], "?") == ans,
        }


async def main():
    print(f"Loading old checkpoints...")
    all_recs = load_old_checkpoints(n_ckpts=3)
    total = sum(len(v) for v in all_recs.values())
    print(f"  Loaded {total:,} records")

    # Build question map
    q_map = {q["id"]: q for q in NORMS_QUESTIONS}

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        for qid in TARGET_QIDS:
            if qid not in q_map:
                print(f"\nSkipping {qid} (not in norms questions)")
                continue
            q = q_map[qid]
            options_str = " / ".join(q["options"])
            print(f"\n{'='*70}")
            print(f"Q: {qid}  (options: {options_str})")
            print(f"{'='*70}")

            # Find records where old label is something interesting
            focus_recs = []
            for sector, recs in all_recs.items():
                for r in recs:
                    old = r["answers"].get(qid, "")
                    if old:
                        focus_recs.append((sector, r))

            sample = random.sample(focus_recs, min(N_SAMPLE, len(focus_recs)))
            print(f"  Sampled {len(sample)} records")

            coros = [relabel_one(session, semaphore, r, sec, q) for sec, r in sample]
            results = await asyncio.gather(*coros)

            changed = [r for r in results if not r["match"]]
            print(f"  Label changes: {len(changed)}/{len(results)} ({len(changed)/len(results)*100:.0f}%)")

            # Show changed cases
            if changed:
                print(f"\n  CHANGED LABELS (old -> new):")
                for r in changed[:12]:
                    print(f"    [{r['sector']}] OLD={r['old_label']!r:20s} NEW={r['new_label']!r}")
                    print(f"      comment: {r['comment'][:120]}")
                    print(f"      raw:     {r['raw'][:120]}")
                    print()

            # Show distribution old vs new
            from collections import Counter
            old_dist = Counter(r["old_label"] for r in results)
            new_dist = Counter(r["new_label"] for r in results)
            print(f"  Old distribution: {dict(old_dist)}")
            print(f"  New distribution: {dict(new_dist)}")

            # Check coworkers specifically
            if qid == "1.3.1_reference_group":
                coworker_old = [r for r in results if r["old_label"] == "coworkers"]
                coworker_new = [r for r in coworker_old if r["new_label"] == "coworkers"]
                print(f"\n  Coworkers old: {len(coworker_old)}, still coworkers with new parser: {len(coworker_new)}")
                for r in coworker_old[:5]:
                    print(f"    [{r['sector']}] OLD=coworkers NEW={r['new_label']!r}")
                    print(f"      comment: {r['comment'][:100]}")
                    print(f"      raw:     {r['raw'][:100]}")


if __name__ == "__main__":
    asyncio.run(main())
