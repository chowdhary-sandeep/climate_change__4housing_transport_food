"""Update verification sample vllm_label fields from corrected checkpoint data."""
import json

# ── Load corrected checkpoint data → build lookup ────────────────────────────
print("Loading merged checkpoint data...")
with open('paper4data/norms_labels_checkpoints_only.json', encoding='utf-8') as f:
    merged = json.load(f)

# Build lookup: (comment_text, sector) -> answers dict
lookup = {}
for sector, records in merged.items():
    for rec in records:
        comment = rec.get('comment', '')
        answers = rec.get('answers', {})
        key = (comment, sector)
        lookup[key] = answers

print(f"Lookup built: {len(lookup)} (comment, sector) pairs")

# ── Load verification samples ─────────────────────────────────────────────────
with open('paper4data/00_verification_samples_ckpt.json', encoding='utf-8') as f:
    vs = json.load(f)

total_updated = 0
not_found = 0
updates_by_qid = {}

for task_type in ['norms', 'survey']:
    if task_type not in vs:
        continue
    for qid, samples in vs[task_type].items():
        updated_count = 0
        for sample in samples:
            comment = sample.get('comment', '')
            sector = sample.get('sector', '')
            key = (comment, sector)
            if key in lookup:
                answers = lookup[key]
                if qid in answers:
                    new_label = str(answers[qid])
                    old_label = sample.get('vllm_label', '')
                    if new_label != old_label:
                        sample['vllm_label'] = new_label
                        total_updated += 1
                        updated_count += 1
            else:
                not_found += 1
        if updated_count:
            updates_by_qid[qid] = updated_count

print(f"\nTotal vllm_label fields updated: {total_updated}")
print(f"(comment, sector) keys not found in checkpoint: {not_found}")
print("\nUpdates by question:")
for qid, cnt in sorted(updates_by_qid.items(), key=lambda x: -x[1]):
    print(f"  {qid}: {cnt}")

# ── Save back ─────────────────────────────────────────────────────────────────
with open('paper4data/00_verification_samples_ckpt.json', 'w', encoding='utf-8') as f:
    json.dump(vs, f, ensure_ascii=False, indent=2)
print("\nSaved updated verification samples.")
