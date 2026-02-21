"""Re-parse all checkpoint raw_responses with the fixed _parse_single_choice."""
import json, re, glob, os
from collections import defaultdict


def _parse_single_choice(content, options, map_to):
    """Fixed parser: HTML stripping + 'X or Y' → prefer Y."""
    c = content.strip()
    # Remove thinking blocks
    for pat in [r'<think>.*?</think>', r'<thinking>.*?</thinking>',
                r'<reason>.*?</reason>', r'<reasoning>.*?</reasoning>']:
        c = re.sub(pat, ' ', c, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining HTML/XML tokens (e.g. <s>NO</s>)
    c = re.sub(r'<[^>]+>', ' ', c)
    c = ' '.join(c.split()).lower()

    def _longest_first(text):
        for opt in sorted(options, key=len, reverse=True):
            if opt.lower() in text:
                return opt.lower()
        return None

    parts = re.split(r'[.\n]', c)
    first = parts[0].strip() if parts else ''

    # Fix: 'optionA or optionB' -> prefer optionB (model's refined answer)
    for o1 in sorted(options, key=len, reverse=True):
        for o2 in sorted(options, key=len, reverse=True):
            if o1.lower() != o2.lower():
                if (o1.lower() + ' or ' + o2.lower()) in first:
                    m = o2.lower()
                    return map_to.get(m, m) if map_to else m

    match = _longest_first(first)
    if match:
        return map_to.get(match, match) if map_to else match

    match = _longest_first(c)
    if match:
        return map_to.get(match, match) if map_to else match

    return options[0] if options else ''


# ── Build question lookup ────────────────────────────────────────────────────
q_lookup = {}  # qid -> {options, map_to}

with open('schema/00_vllm_ipcc_social_norms_schema.json', encoding='utf-8') as f:
    norms_schema = json.load(f)
for q in norms_schema.get('norms_questions', []):
    qid = q.get('id')
    if qid:
        q_lookup[qid] = {'options': q.get('options', []), 'map_to': q.get('map_to')}

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
                    if qid and qid not in q_lookup:
                        q_lookup[qid] = {
                            'options': q.get('options', ['yes', 'no']),
                            'map_to': q.get('map_to')
                        }

print(f'Loaded {len(q_lookup)} question definitions')

# ── Process all checkpoint files ─────────────────────────────────────────────
ckpt_files = sorted(glob.glob('paper4data/checkpoints/checkpoint_*.json'))
print(f'Found {len(ckpt_files)} checkpoint files')

total_changed = 0
changes_by_qid = defaultdict(int)
html_fix_count = 0
or_fix_count = 0

for ckpt_path in ckpt_files:
    with open(ckpt_path, encoding='utf-8') as f:
        data = json.load(f)

    file_changed = False
    for sector, records in data.items():
        if not isinstance(records, list):
            continue
        for rec in records:
            raw = rec.get('raw_responses', {})
            answers = rec.get('answers', {})
            if not raw:
                continue
            for qid, raw_text in raw.items():
                if qid not in q_lookup:
                    continue
                opts = q_lookup[qid]['options']
                mto = q_lookup[qid]['map_to']
                if not opts:
                    continue
                new_ans = _parse_single_choice(raw_text, opts, mto)
                old_ans = answers.get(qid, '')
                if new_ans != old_ans:
                    answers[qid] = new_ans
                    file_changed = True
                    total_changed += 1
                    changes_by_qid[qid] += 1
                    # Track type of fix
                    if re.search(r'<[^>]+>', raw_text):
                        html_fix_count += 1
                    # Check for "X or Y" pattern
                    for o1 in sorted(opts, key=len, reverse=True):
                        for o2 in sorted(opts, key=len, reverse=True):
                            if o1.lower() != o2.lower():
                                c_tmp = re.sub(r'<[^>]+>', ' ', raw_text)
                                c_tmp = ' '.join(c_tmp.split()).lower()
                                parts_tmp = re.split(r'[.\n]', c_tmp)
                                first_tmp = parts_tmp[0].strip()
                                if (o1.lower() + ' or ' + o2.lower()) in first_tmp:
                                    or_fix_count += 1

    if file_changed:
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

print(f'\nTotal labels changed: {total_changed}')
print(f'  HTML-token fixes:   {html_fix_count}')
print(f'  "X or Y" fixes:    {or_fix_count}')
print('\nChanges by question:')
for qid, cnt in sorted(changes_by_qid.items(), key=lambda x: -x[1]):
    print(f'  {qid}: {cnt}')
