"""
00_vLLM_visualize.py

Load norms_labels.json (from 00_vLLM_hierarchical.py --norms) and generate:
- 00_dashboardv2.html — pie and bar charts per question/sector
- 00_dashboard_examples.html — one example comment per (question, category, sector)

Input JSON format: { "food": [ { "comment_index", "comment", "answers": { "1.1_gate": "1", ... } }, ... ], "transport": ..., "housing": ... }
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any

# Map answer codes to display labels (dashboard and examples)
CODE_TO_LABEL = {
    "1.1_gate": {"0": "No", "1": "Yes"},
    "1.2.1_descriptive": {"0": "none", "1": "implied", "2": "explicit"},
    "1.2.2_injunctive": {
        "0": "none",
        "1": "implied approval",
        "2": "implied disapproval",
        "3": "explicit approval",
        "4": "explicit disapproval",
    },
    "1.3.3_second_order": {"0": "none", "1": "weak", "2": "strong"},
}

SECTOR_DISPLAY = {"food": "FOOD", "transport": "TRANSPORT", "housing": "HOUSING"}
QUESTION_TITLES = {
    "1.1_gate": "Norm signal present",
    "1.1.1_stance": "Author stance",
    "1.2.1_descriptive": "Descriptive norm",
    "1.2.2_injunctive": "Injunctive norm",
    "1.3.1_reference_group": "Reference group",
    "1.3.1b_perceived_reference_stance": "Perceived reference stance",
    "1.3.2_mechanism": "Mechanism",
    "1.3.3_second_order": "Second-order normative belief",
}

# Exact prompt and choices sent to the LLM (mirrors 00_vLLM_hierarchical.NORMS_QUESTIONS for display)
NORMS_PROMPTS = {
    "1.1_gate": {
        "prompt": "Does this comment or post reference what others do or approve, or any social norm (descriptive or injunctive)? Answer with exactly one word: yes or no.",
        "options": ["yes", "no"],
    },
    "1.1.1_stance": {
        "prompt": "What is the author's stance toward {sector_topic}? Answer with exactly one of: against, against particular but pro, neither/mixed, pushing for.",
        "options": ["against", "against particular but pro", "neither/mixed", "pushing for"],
        "sector_specific": True,
        "sector_topic": {"transport": "EVs", "food": "veganism or vegetarianism / diet", "housing": "solar"},
    },
    "1.2.1_descriptive": {
        "prompt": "Does the text express a descriptive norm (what people do / how common something is)? Answer with exactly one of: none, implied, explicit.",
        "options": ["none", "implied", "explicit"],
    },
    "1.2.2_injunctive": {
        "prompt": "Does the text express an injunctive norm (what people should do / approval or disapproval)? Answer with exactly one of: none, implied approval, implied disapproval, explicit approval, explicit disapproval.",
        "options": ["none", "implied approval", "implied disapproval", "explicit approval", "explicit disapproval"],
    },
    "1.3.1_reference_group": {
        "prompt": "Who is the reference group (who the author refers to as doing or approving something)? Answer with exactly one of: coworkers, family, friends, local community, neighbors, online community, other, other reddit user, partner/spouse, people like me, political tribe.",
        "options": ["coworkers", "family", "friends", "local community", "neighbors", "online community", "other", "other reddit user", "partner/spouse", "people like me", "political tribe"],
    },
    "1.3.1b_perceived_reference_stance": {
        "prompt": "What stance does the author attribute to that reference group? Answer with exactly one of: against, neither/mixed, pushing for.",
        "options": ["against", "neither/mixed", "pushing for"],
    },
    "1.3.2_mechanism": {
        "prompt": "What mechanism is used to convey the norm or social pressure? Answer with exactly one of: blame/shame, community standard, identity/status signaling, other, praise, rule/virtue language, social comparison.",
        "options": ["blame/shame", "community standard", "identity/status signaling", "other", "praise", "rule/virtue language", "social comparison"],
    },
    "1.3.3_second_order": {
        "prompt": "Does the text express second-order normative beliefs (beliefs about what others think one should do)? Answer with exactly one of: none, weak, strong.",
        "options": ["none", "weak", "strong"],
    },
}


def load_norms_labels(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def answer_to_label(qid: str, value: str) -> str:
    if qid in CODE_TO_LABEL and value in CODE_TO_LABEL[qid]:
        return CODE_TO_LABEL[qid][value]
    return value


def compute_counts(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """counts[question_id][sector][answer_label] = count"""
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for sector, items in data.items():
        for rec in items:
            ans = rec.get("answers") or {}
            for qid, val in ans.items():
                label = answer_to_label(qid, str(val).strip())
                counts[qid][sector][label] += 1
    return counts


def _chart_id(qid: str, sector: str = "") -> str:
    """Safe HTML id and JS variable name: no dots (e.g. 1.1_gate -> 1_1_gate)."""
    safe_q = qid.replace(".", "_")
    return f"{safe_q}_{sector}" if sector else safe_q


def build_dashboard_html(counts: Dict[str, Dict[str, Dict[str, int]]], out_path: str) -> None:
    """Write dashboard HTML with Plotly charts (inline JSON + plotly.js)."""
    sectors = ["food", "transport", "housing"]
    question_order = [
        "1.1_gate",
        "1.1.1_stance",
        "1.3.1_reference_group",
        "1.3.1b_perceived_reference_stance",
        "1.2.1_descriptive",
        "1.2.2_injunctive",
        "1.3.2_mechanism",
        "1.3.3_second_order",
    ]
    colors = {
        "Yes": "#3498db",
        "No": "#9b59b6",
        "pushing for": "#2ecc71",
        "against": "#e74c3c",
        "against particular but pro": "#e67e22",
        "neither/mixed": "#f1c40f",
        "none": "#95a5a6",
        "implied": "#673ab7",
        "explicit": "#3f51b5",
        "implied approval": "#3498db",
        "implied disapproval": "#9b59b6",
        "explicit approval": "#34495e",
        "explicit disapproval": "#8e44ad",
        # Reference group (distinct per category)
        "family": "#2c3e50",
        "partner/spouse": "#9b59b6",
        "friends": "#3498db",
        "coworkers": "#8e44ad",
        "neighbors": "#2980b9",
        "local community": "#1abc9c",
        "political tribe": "#5d6d7e",
        "online community": "#95a5a6",
        "other reddit user": "#e91e63",
        "people like me": "#16a085",
        "other": "#546e7a",
        # Mechanism
        "social comparison": "#5c6bc0",
        "praise": "#66bb6a",
        "blame/shame": "#ef5350",
        "community standard": "#26a69a",
        "identity/status signaling": "#ffa726",
        "rule/virtue language": "#ab47bc",
        # Second-order
        "weak": "#42a5f5",
        "strong": "#ff7043",
    }
    # Fallback palette for any label not in colors (distinct hues)
    _palette_extra = [
        "#7e57c2", "#ec407a", "#26c6da", "#9ccc65", "#ffca28",
        "#8d6e63", "#78909c", "#5c6bc0", "#66bb6a", "#ef5350",
    ]

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Norms Hierarchical Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
html { background: #0d1117; }
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 16px; background: #0d1117; color: #e6edf3; min-height: 100vh; }
h1 { text-align: center; color: #e6edf3; }
h2 { margin-top: 32px; color: #c9d1d9; }
.chart { margin: 16px 0; background: #161b22; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); padding: 12px; }
.charts-row { display: flex; flex-wrap: nowrap; gap: 16px; margin: 16px 0; }
.charts-row .chart { flex: 1; min-width: 0; margin: 0; }
a { color: #58a6ff; }
</style>
</head>
<body>
<h1>Norms Hierarchical Dashboard</h1>
<p style="text-align:center"><a href="00_dashboard_examples.html">Example comments by category</a></p>
"""
    ]

    for qid in question_order:
        if qid not in counts:
            continue
        title = QUESTION_TITLES.get(qid, qid)
        html_parts.append(f"<h2>{title}</h2>\n")

        c = counts[qid]
        if qid == "1.1_gate":
            # Build all three donuts, then output divs then scripts so all containers exist before any Plotly runs
            gate_figs = []
            for sector in sectors:
                yes_count = c.get(sector, {}).get("Yes", 0) + c.get(sector, {}).get("1", 0)
                no_count = c.get(sector, {}).get("No", 0) + c.get(sector, {}).get("0", 0)
                sector_title = SECTOR_DISPLAY.get(sector, sector)
                layout = {
                    "title": {"text": sector_title, "font": {"color": "#e6edf3"}},
                    "showlegend": True,
                    "height": 360,
                    "paper_bgcolor": "#161b22",
                    "plot_bgcolor": "#161b22",
                    "font": {"color": "#e6edf3"},
                }
                fig = {
                    "data": [{
                        "labels": ["Yes", "No"],
                        "values": [yes_count, no_count],
                        "type": "pie",
                        "hole": 0.5,
                        "marker": {"colors": [colors.get("Yes", "#3498db"), colors.get("No", "#9b59b6")]},
                        "customdata": [[sector, "1"], [sector, "0"]],
                    }],
                    "layout": layout,
                }
                cid = _chart_id(qid, sector)
                gate_figs.append((cid, fig))
            html_parts.append('<div class="charts-row">\n')
            for cid, _ in gate_figs:
                html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
            html_parts.append('</div>\n')
            for cid, fig in gate_figs:
                html_parts.append(
                    f'<script>var fig_{cid} = {json.dumps(fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
                )
        else:
            labels_seen = set()
            for sector in sectors:
                for label in c.get(sector, {}).keys():
                    labels_seen.add(label)
            labels_order = [l for l in colors if l in labels_seen] + [l for l in sorted(labels_seen) if l not in colors]
            # Assign distinct color per label (from colors dict or fallback palette)
            label_to_color = {}
            unseen_idx = 0
            for label in labels_order:
                if label in colors:
                    label_to_color[label] = colors[label]
                else:
                    label_to_color[label] = _palette_extra[unseen_idx % len(_palette_extra)]
                    unseen_idx += 1
            x_by_label = {label: [c.get(s, {}).get(label, 0) for s in sectors] for label in labels_order}
            y = [SECTOR_DISPLAY.get(s, s) for s in sectors]
            traces = []
            for label in labels_order:
                x = x_by_label[label]
                if any(x):
                    traces.append({
                        "x": x,
                        "y": y,
                        "name": label,
                        "type": "bar",
                        "orientation": "h",
                        "marker": {"color": label_to_color[label]},
                        "text": [str(v) if v else "" for v in x],
                        "textposition": "inside",
                    })
            if traces:
                fig = {
                    "data": traces,
                    "layout": {
                        "barmode": "stack",
                        "height": 280,
                        "margin": {"l": 100},
                        "xaxis": {"title": "Count", "color": "#e6edf3", "gridcolor": "#30363d"},
                        "yaxis": {"color": "#e6edf3", "gridcolor": "#30363d"},
                        "showlegend": True,
                        "paper_bgcolor": "#161b22",
                        "plot_bgcolor": "#161b22",
                        "font": {"color": "#e6edf3"},
                        "legend": {"font": {"color": "#e6edf3"}},
                    },
                }
                cid = _chart_id(qid)
                html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
                html_parts.append(
                    f'<script>var fig_{cid} = {json.dumps(fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
                )

    html_parts.append("</body>\n</html>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Wrote dashboard: {out_path}")


def build_examples_html(data: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    """One example comment per (question, category, sector)."""
    sectors = ["food", "transport", "housing"]
    question_order = [
        "1.1_gate",
        "1.1.1_stance",
        "1.3.1_reference_group",
        "1.3.1b_perceived_reference_stance",
        "1.2.1_descriptive",
        "1.2.2_injunctive",
        "1.3.2_mechanism",
    ]
    # Collect by (qid, label, sector) -> first comment text
    by_cat: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for sector, items in data.items():
        for rec in items:
            comment = (rec.get("comment") or "").strip()
            if not comment:
                continue
            ans = rec.get("answers") or {}
            for qid, val in ans.items():
                label = answer_to_label(qid, str(val).strip())
                if sector not in by_cat[qid][label]:
                    by_cat[qid][label][sector] = comment[:500] + ("..." if len(comment) > 500 else "")

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Norms Hierarchical - Example Comments</title>
<style>
* { box-sizing: border-box; }
html { background: #0d1117; }
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 16px; background: #0d1117; color: #e6edf3; min-height: 100vh; }
h1 { text-align: center; color: #e6edf3; }
.back-link { text-align: center; margin-bottom: 24px; }
.back-link a { color: #58a6ff; }
.question-section { max-width: 1400px; margin: 0 auto 40px; background: #161b22; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); padding: 24px; }
.question-title { font-size: 18px; font-weight: 600; color: #c9d1d9; margin-bottom: 20px; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
.category-group { margin-bottom: 24px; }
.category-title { font-weight: 600; font-size: 14px; margin-bottom: 12px; padding: 8px 12px; border-radius: 6px; display: inline-block; color: #e6edf3; }
.sectors-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.sector-column { display: flex; flex-direction: column; }
.sector-header { font-weight: 600; font-size: 12px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; }
.example-comment { padding: 14px; border-radius: 8px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; border-left: 4px solid #58a6ff; background: #0d1117; color: #e6edf3; }
.prompt-dropdown { margin-bottom: 20px; }
.prompt-dropdown summary { cursor: pointer; color: #58a6ff; font-size: 14px; user-select: none; }
.prompt-dropdown summary:hover { text-decoration: underline; }
.prompt-box { margin-top: 12px; padding: 14px; border-radius: 8px; background: #0d1117; border: 1px solid #30363d; font-size: 13px; }
.prompt-box .prompt-text { color: #e6edf3; line-height: 1.6; margin-bottom: 12px; }
.prompt-box .choices-label { color: #8b949e; font-weight: 600; margin-bottom: 6px; }
.prompt-box .choices-list { color: #c9d1d9; list-style: none; padding-left: 0; }
.prompt-box .choices-list li { margin: 4px 0; }
</style>
</head>
<body>
<h1>Example Comments by Category</h1>
<div class="back-link"><a href="00_dashboardv2.html">&lt;- Back to Dashboard</a></div>
"""
    ]

    for qid in question_order:
        if qid not in by_cat:
            continue
        title = QUESTION_TITLES.get(qid, qid)
        html_parts.append(f'<div class="question-section"><div class="question-title">{title}</div>\n')
        # Dropdown: exact prompt and choices sent to the LLM
        prompt_info = NORMS_PROMPTS.get(qid)
        if prompt_info:
            html_parts.append('<details class="prompt-dropdown"><summary>Read exact prompt &amp; choices sent to the LLM</summary>\n')
            html_parts.append('<div class="prompt-box">\n')
            prompt_text = prompt_info["prompt"]
            if prompt_info.get("sector_specific") and prompt_info.get("sector_topic"):
                html_parts.append('<div class="choices-label">Prompt uses only the comment\'s sector topic (not all three):</div>\n')
                st = prompt_info["sector_topic"]
                for sec, topic in st.items():
                    html_parts.append(f'<div class="prompt-text"><strong>{sec}:</strong> {html_escape(topic)}</div>\n')
                html_parts.append('<div class="choices-label">Prompt template:</div>\n')
            html_parts.append(f'<div class="prompt-text">{html_escape(prompt_text)}</div>\n')
            html_parts.append('<div class="choices-label">Valid choices:</div>\n<ul class="choices-list">\n')
            for opt in prompt_info["options"]:
                html_parts.append(f'<li>{html_escape(opt)}</li>\n')
            html_parts.append('</ul>\n</div>\n</details>\n')
        for label in sorted(by_cat[qid].keys()):
            html_parts.append(f'<div class="category-group"><div class="category-title">{label}</div><div class="sectors-row">\n')
            for sector in sectors:
                text = by_cat[qid][label].get(sector, "No examples")
                html_parts.append(
                    f'<div class="sector-column"><div class="sector-header">{SECTOR_DISPLAY.get(sector, sector)}</div>'
                    f'<div class="example-comment">{html_escape(text)}</div></div>\n'
                )
            html_parts.append("</div></div>\n")
        html_parts.append("</div>\n")

    html_parts.append("</body>\n</html>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Wrote examples: {out_path}")


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Build norms dashboard and examples from norms_labels.json")
    p.add_argument("--input", default="paper4data/norms_labels.json", help="Input JSON from 00_vLLM_hierarchical.py --norms")
    p.add_argument("--dashboard", default="00_dashboardv2.html", help="Output dashboard HTML path")
    p.add_argument("--examples", default="00_dashboard_examples.html", help="Output examples HTML path")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        print("Run first: python 00_vLLM_hierarchical.py --label-only --norms --limit-total 1000")
        return

    data = load_norms_labels(args.input)
    counts = compute_counts(data)
    build_dashboard_html(counts, args.dashboard)
    build_examples_html(data, args.examples)
    print("Done.")


if __name__ == "__main__":
    main()
