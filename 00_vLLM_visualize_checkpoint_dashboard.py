"""
Generate dashboard_checkpoints.html:
  - Uses norms_labels_full.json (123k checkpoint data, 41k/sector) for temporal analysis
  - Merges DASHBOARD_DOCUMENTATION.html as a Docs tab
  - Outputs dashboard_checkpoints.html
"""
import re, os

# ── 1. Read and patch temp.py source ─────────────────────────────────────────
with open('temp.py', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
    'paper4data/norms_labels.json',
    'paper4data/norms_labels_full.json'
)
code = code.replace(
    '9,000 comments',
    '123,000 comments (checkpoint data)'
)
code = code.replace(
    'open("temp.html", "w", encoding="utf-8")',
    'open("dashboard_checkpoints.html", "w", encoding="utf-8")'
)
code = code.replace(
    'Generated temp.html',
    'Generated dashboard_checkpoints.html'
)

# ── 2. Execute patched code ───────────────────────────────────────────────────
ns = {'__name__': '__main__'}
exec(compile(code, '<patched_temp>', 'exec'), ns, ns)

# ── 3. Load generated HTML ────────────────────────────────────────────────────
with open('dashboard_checkpoints.html', encoding='utf-8') as f:
    html = f.read()

# ── 4. Load and extract docs content ─────────────────────────────────────────
with open('DASHBOARD_DOCUMENTATION.html', encoding='utf-8') as f:
    docs_html = f.read()

style_m = re.search(r'<style>(.*?)</style>', docs_html, re.DOTALL)
docs_css_raw = style_m.group(1) if style_m else ''

body_m = re.search(r'<body>(.*?)</body>', docs_html, re.DOTALL)
docs_body = body_m.group(1).strip() if body_m else ''

# ── 4b. Patch docs body for checkpoint data values ────────────────────────────
# 123k total (41k/sector), ~2.1M labels
docs_body = docs_body.replace('9,000 comments (3,000/sector)', '123,000 comments (41,000/sector)')
docs_body = docs_body.replace('9,000 &times; 7 norms', '123,000 &times; 7 norms')
docs_body = docs_body.replace('3,000 &times; 13', '41,000 &times; 13')
docs_body = docs_body.replace('3,000 &times; 6', '41,000 &times; 6')
docs_body = docs_body.replace('3,000 &times; 10', '41,000 &times; 10')
docs_body = docs_body.replace('<b>150,000 individual labels</b>', '<b>~2.1M individual labels</b>')
docs_body = docs_body.replace('>9,000<', '>123,000<')
docs_body = docs_body.replace('>150K<', '>~2.1M<')
docs_body = docs_body.replace('3,000/sector with equal yearly', '41,000/sector with equal yearly')
docs_body = docs_body.replace('sample 3,000/sector', 'sample 41,000/sector')
docs_body = docs_body.replace('9,000 records with comment', '123,000 records with comment')
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
