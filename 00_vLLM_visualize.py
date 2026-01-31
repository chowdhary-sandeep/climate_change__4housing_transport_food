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
    }

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
            for sector in sectors:
                yes_count = c.get(sector, {}).get("Yes", 0) + c.get(sector, {}).get("1", 0)
                no_count = c.get(sector, {}).get("No", 0) + c.get(sector, {}).get("0", 0)
                layout = {
                    "title": SECTOR_DISPLAY.get(sector, sector),
                    "showlegend": True,
                    "height": 360,
                    "paper_bgcolor": "#161b22",
                    "plot_bgcolor": "#161b22",
                    "font": {"color": "#e6edf3"},
                    "title": {"font": {"color": "#e6edf3"}},
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
                html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
                html_parts.append(
                    f'<script>var fig_{cid} = {json.dumps(fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
                )
        else:
            labels_seen = set()
            for sector in sectors:
                for label in c.get(sector, {}).keys():
                    labels_seen.add(label)
            labels_order = [l for l in colors if l in labels_seen] + [l for l in sorted(labels_seen) if l not in colors]
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
                        "marker": {"color": colors.get(label, "#607d8b")},
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
