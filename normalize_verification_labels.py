"""Normalize vllm_label and reasoning_label in verification samples.

Problems found:
1. diet_10, diet_13: reasoning_label "1" should be "yes" (yes/no question, no map_to)
2. 1.1_gate: some vllm_label "yes"/"no" not mapped to "0"/"1" (records not in checkpoint)
3. 1.3.3_second_order: some vllm_label "none" not mapped to "0"

Fix: for each question, define canonical label space (the map_to output if map_to exists,
else the raw options). Normalize both vllm_label and reasoning_label to that space.
"""
import json
from collections import Counter

# ── Load schemas ─────────────────────────────────────────────────────────────
q_info = {}  # qid -> {options, map_to, canonical}

with open('schema/00_vllm_ipcc_social_norms_schema.json', encoding='utf-8') as f:
    norms_schema = json.load(f)
for q in norms_schema.get('norms_questions', []):
    qid = q.get('id')
    if qid:
        opts = q.get('options', [])
        mto = q.get('map_to') or {}
        # canonical space = map_to values if map_to exists, else options
        canonical = list(mto.values()) if mto else [o.lower() for o in opts]
        q_info[qid] = {'options': opts, 'map_to': mto, 'canonical': canonical}

with open('schema/00_vllm_survey_question_final.json', encoding='utf-8') as f:
    survey_schema = json.load(f)
for sector_name, sector_data in survey_schema.items():
    if sector_name == 'survey_system':
        continue
    if isinstance(sector_data, dict):
        for topic, topic_data in sector_data.items():
            if isinstance(topic_data, dict) and 'questions' in topic_data:
                for q in topic_data['questions']:
                    qid = q.get('id')
                    if qid and qid not in q_info:
                        opts = q.get('options', ['yes', 'no'])
                        mto = q.get('map_to') or {}
                        canonical = list(mto.values()) if mto else [o.lower() for o in opts]
                        q_info[qid] = {'options': opts, 'map_to': mto, 'canonical': canonical}

# ── Build normalizer for each question ───────────────────────────────────────
def make_normalizer(info):
    """Return a function that maps any label string to the canonical space."""
    opts_lower = [o.lower() for o in info['options']]
    mto = {k.lower(): v.lower() for k, v in info['map_to'].items()} if info['map_to'] else {}
    canonical_lower = [c.lower() for c in info['canonical']]

    # Build reverse map: canonical → canonical (identity) + raw → canonical
    norm_map = {}
    for c in canonical_lower:
        norm_map[c] = c  # already canonical
    for raw, mapped in mto.items():
        norm_map[raw.lower()] = mapped.lower()

    # For yes/no questions without map_to: add numeric aliases
    if 'yes' in opts_lower and 'no' in opts_lower and not mto:
        norm_map['1'] = 'yes'
        norm_map['0'] = 'no'
        norm_map['true'] = 'yes'
        norm_map['false'] = 'no'
    # For yes/no questions WITH map_to (gate): "yes"→"1", "no"→"0" via mto, also "1"→"yes" reverse?
    # We keep canonical as map_to values, so "1"→"1", "0"→"0", "yes"→"1", "no"→"0"

    def normalize(label):
        s = str(label).strip().lower()
        return norm_map.get(s, s)  # return as-is if not found

    return normalize

normalizers = {qid: make_normalizer(info) for qid, info in q_info.items()}

# ── Load and fix verification samples ────────────────────────────────────────
with open('paper4data/00_verification_samples_ckpt.json', encoding='utf-8') as f:
    vs = json.load(f)

vllm_changed = 0
reasoning_changed = 0
changes_log = []

for task_type in ['norms', 'survey']:
    for qid, samples in vs.get(task_type, {}).items():
        norm = normalizers.get(qid)
        if not norm:
            continue
        for s in samples:
            # Fix vllm_label
            old_vl = str(s.get('vllm_label', '')).strip().lower()
            new_vl = norm(old_vl)
            if new_vl != old_vl:
                s['vllm_label'] = new_vl
                vllm_changed += 1
                changes_log.append(f'  vllm {qid}: {repr(old_vl)} -> {repr(new_vl)}')

            # Fix reasoning_label
            old_rl = str(s.get('reasoning_label', '')).strip().lower()
            new_rl = norm(old_rl)
            if new_rl != old_rl:
                s['reasoning_label'] = new_rl
                reasoning_changed += 1
                changes_log.append(f'  reasoning {qid}: {repr(old_rl)} -> {repr(new_rl)}')

print(f'vllm_label changes:     {vllm_changed}')
print(f'reasoning_label changes: {reasoning_changed}')
if changes_log:
    print('\nAll changes:')
    for line in changes_log[:50]:
        print(line)
    if len(changes_log) > 50:
        print(f'  ... ({len(changes_log) - 50} more)')

with open('paper4data/00_verification_samples_ckpt.json', 'w', encoding='utf-8') as f:
    json.dump(vs, f, ensure_ascii=False, indent=2)
print('\nSaved.')

# ── Verify result ─────────────────────────────────────────────────────────────
print('\n=== Checking remaining label-space issues ===')
for task_type in ['norms', 'survey']:
    for qid, samples in vs.get(task_type, {}).items():
        info = q_info.get(qid, {})
        canonical = set(c.lower() for c in info.get('canonical', []))
        if not canonical:
            continue
        rl_odd = [str(s.get('reasoning_label', '')).strip().lower()
                  for s in samples
                  if str(s.get('reasoning_label', '')).strip().lower() not in canonical]
        vl_odd = [str(s.get('vllm_label', '')).strip().lower()
                  for s in samples
                  if str(s.get('vllm_label', '')).strip().lower() not in canonical]
        if rl_odd or vl_odd:
            print(f'{qid}: odd reasoning={Counter(rl_odd)}, odd vllm={Counter(vl_odd)}')
print('Done.')
