"""
Generate dashboard_checkpoints.html from current paper4data/checkpoints/ data.

  - Merges all checkpoint JSONs into norms_labels_checkpoints_only.json
  - Executes 00_vLLM_dashboard_engine.py (patched to use merged data)
  - Injects docs/DASHBOARD_DOCUMENTATION.html as a Docs tab
  - Outputs dashboard_checkpoints.html

No dependency on temp.py or temp.html.
"""
import re, os, json
from collections import defaultdict

# ── 1. Merge all checkpoint files ─────────────────────────────────────────────
ckpt_dir = 'paper4data/checkpoints'
merged = defaultdict(list)
for fname in sorted(os.listdir(ckpt_dir)):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(ckpt_dir, fname), encoding='utf-8') as f:
        data = json.load(f)
    for sector, items in data.items():
        merged[sector].extend(items)

total = sum(len(v) for v in merged.values())
per_sector = total // max(len(merged), 1)
print(f"Merged {total:,} items ({per_sector:,}/sector) from {ckpt_dir}")

merged_path = 'paper4data/norms_labels_checkpoints_only.json'
with open(merged_path, 'w', encoding='utf-8') as f:
    json.dump(dict(merged), f, ensure_ascii=False, indent=0)

# ── 2. Read and patch 00_vLLM_dashboard_engine.py ────────────────────────────
with open('00_vLLM_dashboard_engine.py', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
    'paper4data/norms_labels.json',
    merged_path
)
# Update subtitle comment count dynamically
code = code.replace(
    '9,000 comments &mdash; LLM-labeled with verification',
    f'{total:,} comments ({per_sector:,}/sector) &mdash; LLM-labeled with verification'
)
code = code.replace(
    'open("temp.html", "w", encoding="utf-8")',
    'open("dashboard_checkpoints.html", "w", encoding="utf-8")'
)
code = code.replace(
    'Generated temp.html',
    'Generated dashboard_checkpoints.html'
)
# Fix schema paths (engine may have pre-move paths)
code = code.replace(
    '"00_vllm_survey_question_final.json"',
    '"schema/00_vllm_survey_question_final.json"'
)
code = code.replace(
    '"00_vllm_ipcc_social_norms_schema.json"',
    '"schema/00_vllm_ipcc_social_norms_schema.json"'
)
code = code.replace(
    '"local_LLM_api_from_vLLM.json"',
    '"schema/local_LLM_api_from_vLLM.json"'
)

# ── 3. Execute patched engine ─────────────────────────────────────────────────
ns = {'__name__': '__main__'}
exec(compile(code, '00_vLLM_dashboard_engine.py', 'exec'), ns, ns)

# ── 3. Load generated HTML ────────────────────────────────────────────────────
with open('dashboard_checkpoints.html', encoding='utf-8') as f:
    html = f.read()

# ── 4. Load and extract docs content ─────────────────────────────────────────
with open('docs/DASHBOARD_DOCUMENTATION.html', encoding='utf-8') as f:
    docs_html = f.read()

style_m = re.search(r'<style>(.*?)</style>', docs_html, re.DOTALL)
docs_css_raw = style_m.group(1) if style_m else ''

body_m = re.search(r'<body>(.*?)</body>', docs_html, re.DOTALL)
docs_body = body_m.group(1).strip() if body_m else ''

# ── 4b. Patch docs body for checkpoint data values ────────────────────────────
# 21,300 total (7,100/sector)
docs_body = docs_body.replace('9,000 comments (3,000/sector)', '21,300 comments (7,100/sector)')
docs_body = docs_body.replace('9,000 &times; 7 norms', '21,300 &times; 7 norms')
docs_body = docs_body.replace('3,000 &times; 13', '7,100 &times; 13')
docs_body = docs_body.replace('3,000 &times; 6', '7,100 &times; 6')
docs_body = docs_body.replace('3,000 &times; 10', '7,100 &times; 10')
docs_body = docs_body.replace('<b>150,000 individual labels</b>', '<b>~277,000 individual labels</b>')
docs_body = docs_body.replace('>9,000<', '>21,300<')
docs_body = docs_body.replace('>150K<', '>~277K<')
docs_body = docs_body.replace('3,000/sector with equal yearly', '7,100/sector with equal yearly')
docs_body = docs_body.replace('sample 3,000/sector', 'sample 7,100/sector')
docs_body = docs_body.replace('9,000 records with comment', '21,300 records with comment')
docs_body = docs_body.replace('Labeled Comments</div>', 'Labeled Comments (checkpoint)</div>')

# ── 5. Scope docs CSS to #tab-docs ────────────────────────────────────────────
# Drop html/body/* resets — irrelevant when nested
docs_css = re.sub(r'(?:html|body|\*)\s*\{[^}]*\}', '', docs_css_raw)
# Drop @media blocks (responsive tweaks not needed inside tab)
docs_css = re.sub(r'@media[^{]*\{(?:[^{}]*|\{[^}]*\})*\}', '', docs_css, flags=re.DOTALL)
# Prefix every remaining selector with #tab-docs for specificity isolation
def prefix_sel(m):
    sels = m.group(1)
    body = m.group(2)
    new_sels = ', '.join(f'#tab-docs {s.strip()}' for s in sels.split(',') if s.strip())
    return f'{new_sels}{{{body}}}'
docs_css = re.sub(r'([^{}\n/][^{}]*)\{([^}]*)\}', prefix_sel, docs_css)

# ── 6. Inject Docs tab button ─────────────────────────────────────────────────
html = html.replace(
    "<button class=\"tab-btn\" onclick=\"showTab('examples')\">Examples</button>",
    "<button class=\"tab-btn\" onclick=\"showTab('examples')\">Examples</button>\n"
    "<span class=\"tab-sep\"></span>\n"
    "<button class=\"tab-btn\" onclick=\"showTab('docs')\">Docs</button>"
)

# ── 7. Inject scoped CSS into page <style> ────────────────────────────────────
html = html.replace(
    '</style>',
    f'\n/* === Docs tab scoped styles === */\n{docs_css}\n</style>',
    1
)

# ── 8. Inject docs tab panel ─────────────────────────────────────────────────
docs_panel = (
    '\n<!-- DOCS TAB -->\n'
    '<div class="tab-content" id="tab-docs" style="background:#0a1628;padding:20px 0">\n'
    '<div style="max-width:1100px;margin:0 auto;padding:0 20px;line-height:1.7">\n'
    + docs_body +
    '\n</div>\n</div>\n'
)
html = html.replace('</body>', docs_panel + '</body>')

# ── 9. Write final file ───────────────────────────────────────────────────────
with open('dashboard_checkpoints.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('dashboard_checkpoints.html written.')
