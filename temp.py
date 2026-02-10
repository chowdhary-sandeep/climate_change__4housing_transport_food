"""
temp.py - Generate beautiful alternative visualizations for dashboard plots
Inspired by informationisbeautiful.net
"""
import json

# ═══════════════════════════════════════════════════════════════════
# Data extracted from dashboard
# ═══════════════════════════════════════════════════════════════════

NORM_SIGNAL = {
    "Food": {"yes": 201, "no": 299, "total": 500},
    "Transport": {"yes": 58, "no": 442, "total": 500},
    "Housing": {"yes": 81, "no": 419, "total": 500},
}

AUTHOR_STANCE = {
    "categories": ["pro", "against", "against particular\\nbut pro", "neither/mixed", "pro but lack\\nof options"],
    "Food":      [107, 43, 15, 236, 99],
    "Transport": [61, 45, 9, 291, 94],
    "Housing":   [75, 48, 17, 253, 107],
}

REFERENCE_GROUP = {
    "categories": ["family", "partner/spouse", "friends", "coworkers", "neighbors",
                   "local community", "political tribe", "online community", "other reddit user", "other"],
    "Food":      [22, 12, 50, 15, 3, 10, 0, 6, 19, 363],
    "Transport": [6, 1, 2, 56, 6, 17, 0, 2, 20, 390],
    "Housing":   [10, 2, 3, 58, 8, 24, 3, 1, 16, 375],
}

DESCRIPTIVE_NORM = {
    "categories": ["explicitly present", "absent", "unclear"],
    "Food":      [142, 133, 225],
    "Transport": [77, 350, 73],
    "Housing":   [70, 328, 102],
}

INJUNCTIVE_NORM = {
    "categories": ["present", "unclear", "absent"],
    "Food":      [40, 25, 435],
    "Transport": [27, 7, 466],
    "Housing":   [47, 11, 442],
}

SECOND_ORDER = {
    "categories": ["strong", "weak", "none"],
    "Food":      [58, 104, 338],
    "Transport": [54, 79, 367],
    "Housing":   [83, 83, 334],
}

# ═══════════════════════════════════════════════════════════════════
# Build HTML
# ═══════════════════════════════════════════════════════════════════

html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Dashboard Visualization Experiments</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html { background: #0a1628; }
body { font-family: "Inter", "Segoe UI", system-ui, sans-serif; background: #0a1628; color: #e0e0e0; padding: 20px; }
h1 { text-align: center; font-size: 1.6em; color: #fff; margin-bottom: 5px; font-weight: 300; letter-spacing: 2px; }
.subtitle { text-align: center; color: #6a8caf; font-size: 0.85em; margin-bottom: 25px; }

/* Tabs */
.tabs { display: flex; gap: 4px; margin-bottom: 0; flex-wrap: wrap; justify-content: center; }
.tab-btn {
    padding: 10px 22px; cursor: pointer; border: none;
    background: #12203a; color: #6a8caf; font-size: 0.85em;
    border-radius: 8px 8px 0 0; transition: all 0.3s;
    font-family: inherit; font-weight: 500;
}
.tab-btn:hover { background: #1a3050; color: #a0c4e8; }
.tab-btn.active { background: #1a3050; color: #fff; border-bottom: 2px solid #5ab4ac; }
.tab-content { display: none; background: #0f1d33; border-radius: 0 8px 8px 8px; padding: 25px; min-height: 600px; }
.tab-content.active { display: block; }

/* Variation grid */
.var-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
.var-card {
    background: #12203a; border-radius: 12px; padding: 20px;
    border: 1px solid #1a3050; transition: border-color 0.3s;
}
.var-card:hover { border-color: #5ab4ac; }
.var-label { font-size: 0.75em; color: #5ab4ac; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; }
.var-title { font-size: 1.1em; color: #fff; margin-bottom: 15px; font-weight: 400; }
.chart-container { width: 100%; }

/* Bubble chart specific */
.bubble-chart { width: 100%; height: 480px; }
.bubble-chart svg { width: 100%; height: 100%; }
.bubble-label { fill: #fff; font-family: Inter, sans-serif; pointer-events: none; text-anchor: middle; }
.bubble-sector-label { fill: #6a8caf; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }

/* Waffle specific */
.waffle-grid { display: grid; grid-template-columns: repeat(20, 1fr); gap: 2px; max-width: 340px; margin: 0 auto; }
.waffle-cell { aspect-ratio: 1; border-radius: 2px; transition: transform 0.2s; }
.waffle-cell:hover { transform: scale(1.3); z-index: 10; }

/* Treemap label */
.treemap-title { font-size: 0.95em; color: #b0c4d8; margin-bottom: 8px; text-align: center; }

/* Sankey annotations */
.sankey-note { font-size: 0.8em; color: #6a8caf; text-align: center; margin-top: 8px; }

/* Legend */
.legend { display: flex; gap: 15px; justify-content: center; margin-top: 10px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.8em; color: #b0c4d8; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
</style>
</head>
<body>
<h1>VISUALIZATION EXPERIMENTS</h1>
<p class="subtitle">Alternative chart designs for the Norms Dashboard &mdash; inspired by informationisbeautiful.net</p>

<div class="tabs">
    <button class="tab-btn active" onclick="showTab('norm-signal')">Norm Signal</button>
    <button class="tab-btn" onclick="showTab('author-stance')">Author Stance</button>
    <button class="tab-btn" onclick="showTab('reference-group')">Reference Group</button>
    <button class="tab-btn" onclick="showTab('norms-overview')">Norms Overview</button>
</div>

<!-- ════════════════════════════════════════════════════════════ -->
<!-- TAB 1: NORM SIGNAL PRESENT -->
<!-- ════════════════════════════════════════════════════════════ -->
<div class="tab-content active" id="tab-norm-signal">
<div class="var-grid">

<div class="var-card">
<div class="var-label">Variation 1</div>
<div class="var-title">Semi-Circular Gauge</div>
<div id="gauge-chart" class="chart-container"></div>
</div>

<div class="var-card">
<div class="var-label">Variation 2</div>
<div class="var-title">Treemap</div>
<div id="treemap-norm" class="chart-container"></div>
</div>

<div class="var-card">
<div class="var-label">Variation 3</div>
<div class="var-title">Waffle Charts</div>
<div style="display:flex; flex-direction:column; gap:18px;">
WAFFLE_PLACEHOLDER
</div>
</div>

</div>
</div>

<!-- ════════════════════════════════════════════════════════════ -->
<!-- TAB 2: AUTHOR STANCE -->
<!-- ════════════════════════════════════════════════════════════ -->
<div class="tab-content" id="tab-author-stance">
<div class="var-grid">

<div class="var-card">
<div class="var-label">Variation 1</div>
<div class="var-title">Sankey Flow</div>
<div id="sankey-stance" class="chart-container"></div>
<p class="sankey-note">Flow of author stances across all 1500 comments</p>
</div>

<div class="var-card">
<div class="var-label">Variation 2</div>
<div class="var-title">Butterfly Chart</div>
<div id="butterfly-stance" class="chart-container"></div>
</div>

<div class="var-card">
<div class="var-label">Variation 3</div>
<div class="var-title">Proportional Treemap</div>
<div id="treemap-stance" class="chart-container"></div>
</div>

</div>
</div>

<!-- ════════════════════════════════════════════════════════════ -->
<!-- TAB 3: REFERENCE GROUP -->
<!-- ════════════════════════════════════════════════════════════ -->
<div class="tab-content" id="tab-reference-group">
<div class="var-grid">

<div class="var-card" style="grid-column: 1 / 3;">
<div class="var-label">Variation 1</div>
<div class="var-title">Animated Bubble Pack</div>
<div id="bubble-pack" class="bubble-chart"></div>
<div class="legend" id="bubble-legend"></div>
</div>

<div class="var-card">
<div class="var-label">Variation 2</div>
<div class="var-title">Treemap</div>
<div id="treemap-ref" class="chart-container"></div>
</div>

</div>
</div>

<!-- ════════════════════════════════════════════════════════════ -->
<!-- TAB 4: NORMS OVERVIEW -->
<!-- ════════════════════════════════════════════════════════════ -->
<div class="tab-content" id="tab-norms-overview">
<div class="var-grid">

<div class="var-card">
<div class="var-label">Variation 1</div>
<div class="var-title">Heatmap Matrix</div>
<div id="heatmap-norms" class="chart-container"></div>
</div>

<div class="var-card">
<div class="var-label">Variation 2</div>
<div class="var-title">Lollipop Chart</div>
<div id="lollipop-norms" class="chart-container"></div>
</div>

<div class="var-card">
<div class="var-label">Variation 3</div>
<div class="var-title">Radial Bars</div>
<div id="radial-norms" class="chart-container"></div>
</div>

</div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
// Tab switching
// ═══════════════════════════════════════════════════════════════
function showTab(id) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    event.target.classList.add('active');
    // Trigger resize for plotly charts
    window.dispatchEvent(new Event('resize'));
    // Re-init bubbles if switching to reference group
    if (id === 'reference-group') { setTimeout(initBubbles, 100); }
}

var plotlyConfig = {displayModeBar: false, responsive: true};
var darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {color: '#b0c4d8', family: 'Inter, sans-serif', size: 11},
    margin: {t: 30, b: 40, l: 50, r: 20},
};

// ═══════════════════════════════════════════════════════════════
// TAB 1: NORM SIGNAL - Gauge
// ═══════════════════════════════════════════════════════════════
(function() {
    var food_pct = 40.2, transport_pct = 11.6, housing_pct = 16.2;
    var avg = ((food_pct + transport_pct + housing_pct) / 3);

    Plotly.newPlot('gauge-chart', [{
        values: [food_pct, transport_pct, housing_pct, 100],
        labels: ['Food', 'Transport', 'Housing', ''],
        marker: {
            colors: ['#5ab4ac', '#af8dc3', '#f4a460', 'rgba(0,0,0,0)'],
            line: {color: '#0a1628', width: 3}
        },
        hole: 0.70, direction: 'clockwise', sort: false, rotation: 270,
        textposition: 'none', type: 'pie', showlegend: false,
        hovertemplate: '<b>%{label}</b><br>%{value:.1f}% norm signal<extra></extra>'
    }], Object.assign({}, darkLayout, {
        height: 380,
        annotations: [
            {text: '<b>' + avg.toFixed(1) + '%</b>', x: 0.5, y: 0.58, font: {size: 44, color: '#fff'}, showarrow: false},
            {text: 'Avg Norm Signal', x: 0.5, y: 0.42, font: {size: 12, color: '#6a8caf'}, showarrow: false},
            {text: '<span style="color:#5ab4ac">\\u25cf</span> Food ' + food_pct + '%', x: 0.22, y: 0.15, font: {size: 11}, showarrow: false, xanchor: 'left'},
            {text: '<span style="color:#af8dc3">\\u25cf</span> Transport ' + transport_pct + '%', x: 0.40, y: 0.15, font: {size: 11}, showarrow: false, xanchor: 'left'},
            {text: '<span style="color:#f4a460">\\u25cf</span> Housing ' + housing_pct + '%', x: 0.62, y: 0.15, font: {size: 11}, showarrow: false, xanchor: 'left'},
        ]
    }), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 1: NORM SIGNAL - Treemap (bike-sharing style)
// ═══════════════════════════════════════════════════════════════
(function() {
    Plotly.newPlot('treemap-norm', [{
        type: 'treemap',
        labels: [
            'All Comments',
            'Food', 'Transport', 'Housing',
            'Food Yes', 'Food No',
            'Transport Yes', 'Transport No',
            'Housing Yes', 'Housing No'
        ],
        parents: [
            '', 'All Comments', 'All Comments', 'All Comments',
            'Food', 'Food', 'Transport', 'Transport', 'Housing', 'Housing'
        ],
        values: [0, 0, 0, 0, 201, 299, 58, 442, 81, 419],
        text: ['1500 comments', '500', '500', '500',
               '<b>201</b><br>40.2%', '<b>299</b><br>59.8%',
               '<b>58</b><br>11.6%', '<b>442</b><br>88.4%',
               '<b>81</b><br>16.2%', '<b>419</b><br>83.8%'],
        textinfo: 'label+text',
        hovertemplate: '<b>%{label}</b><br>%{text}<extra></extra>',
        marker: {
            colors: ['#0f1d33', '#0f1d33', '#0f1d33', '#0f1d33',
                     '#5ab4ac', '#2a4a5a', '#af8dc3', '#3a2a4a', '#f4a460', '#5a3a2a'],
            line: {color: '#0a1628', width: 2}
        },
        textfont: {color: '#fff', size: 12},
        pathbar: {visible: false},
        branchvalues: 'total',
    }], Object.assign({}, darkLayout, {
        height: 380,
        margin: {t: 10, b: 10, l: 10, r: 10},
        treemapcolorway: ['#5ab4ac', '#af8dc3', '#f4a460'],
    }), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 2: AUTHOR STANCE - Sankey (plastics-style flow)
// ═══════════════════════════════════════════════════════════════
(function() {
    // Nodes: 0=All, 1=Food, 2=Transport, 3=Housing,
    // 4=pro, 5=against, 6=against-but-pro, 7=neither, 8=pro-lack
    var labels = ['All Comments<br>1500', 'Food<br>500', 'Transport<br>500', 'Housing<br>500',
                  'Pro', 'Against', 'Against but Pro', 'Neither / Mixed', 'Pro but Lack of Options'];
    var nodeColors = ['#1a3050', '#5ab4ac', '#af8dc3', '#f4a460',
                      '#8fcc8f', '#ff9aa8', '#ffb87a', '#ffe87a', '#8fbfd9'];

    var source = [0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3];
    var target = [1,2,3, 4,5,6,7,8, 4,5,6,7,8, 4,5,6,7,8];
    var value  = [500,500,500, 107,43,15,236,99, 61,45,9,291,94, 75,48,17,253,107];
    var linkColors = [
        'rgba(90,180,172,0.3)', 'rgba(175,141,195,0.3)', 'rgba(244,164,96,0.3)',
        'rgba(143,204,143,0.4)', 'rgba(255,154,168,0.4)', 'rgba(255,184,122,0.4)', 'rgba(255,232,122,0.4)', 'rgba(143,191,217,0.4)',
        'rgba(143,204,143,0.4)', 'rgba(255,154,168,0.4)', 'rgba(255,184,122,0.4)', 'rgba(255,232,122,0.4)', 'rgba(143,191,217,0.4)',
        'rgba(143,204,143,0.4)', 'rgba(255,154,168,0.4)', 'rgba(255,184,122,0.4)', 'rgba(255,232,122,0.4)', 'rgba(143,191,217,0.4)',
    ];

    Plotly.newPlot('sankey-stance', [{
        type: 'sankey', orientation: 'h',
        node: {
            label: labels, color: nodeColors, pad: 20, thickness: 25,
            line: {color: '#0a1628', width: 1}
        },
        link: {source: source, target: target, value: value, color: linkColors}
    }], Object.assign({}, darkLayout, {height: 450, margin: {t: 10, b: 10, l: 10, r: 10}}), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 2: AUTHOR STANCE - Butterfly Chart
// ═══════════════════════════════════════════════════════════════
(function() {
    var cats = ['pro', 'against', 'against\\nparticular\\nbut pro', 'neither/\\nmixed', 'pro but\\nlack of\\noptions'];
    var food = [107, 43, 15, 236, 99];
    var transport = [61, 45, 9, 291, 94];
    var housing = [75, 48, 17, 253, 107];

    // Show Food on left (negative), Housing on right (positive), Transport in middle
    Plotly.newPlot('butterfly-stance', [
        {y: cats, x: food.map(v => -v), type: 'bar', orientation: 'h', name: 'Food',
         marker: {color: '#5ab4ac'}, text: food.map(v => v), textposition: 'outside', textfont: {size: 9, color: '#5ab4ac'},
         hovertemplate: '<b>Food</b><br>%{y}: %{text}<extra></extra>'},
        {y: cats, x: housing, type: 'bar', orientation: 'h', name: 'Housing',
         marker: {color: '#f4a460'}, text: housing.map(v => v), textposition: 'outside', textfont: {size: 9, color: '#f4a460'},
         hovertemplate: '<b>Housing</b><br>%{y}: %{x}<extra></extra>'},
    ], Object.assign({}, darkLayout, {
        height: 450, barmode: 'overlay', bargap: 0.25,
        margin: {l: 80, r: 60, t: 30, b: 50},
        xaxis: {zeroline: true, zerolinecolor: '#3a5070', zerolinewidth: 2,
                showticklabels: false, showgrid: false},
        yaxis: {tickfont: {size: 10, color: '#b0c4d8'}, gridcolor: 'rgba(0,0,0,0)'},
        legend: {orientation: 'h', x: 0.5, xanchor: 'center', y: -0.05, font: {size: 10}},
        annotations: [{text: '<b>Food</b> \\u2190', x: 0.05, y: 1.04, xref: 'paper', yref: 'paper', showarrow: false, font: {color: '#5ab4ac', size: 11}},
                      {text: '\\u2192 <b>Housing</b>', x: 0.95, y: 1.04, xref: 'paper', yref: 'paper', showarrow: false, font: {color: '#f4a460', size: 11}}]
    }), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 2: AUTHOR STANCE - Treemap
// ═══════════════════════════════════════════════════════════════
(function() {
    var stanceColors = {'pro':'#8fcc8f','against':'#ff9aa8','against particular but pro':'#ffb87a','neither/mixed':'#ffe87a','pro but lack of options':'#8fbfd9'};
    var labels = ['Stances'];
    var parents = [''];
    var values = [0];
    var colors = ['#0f1d33'];
    var texts = ['1500 total'];

    var sectors = ['Food', 'Transport', 'Housing'];
    var sectorData = {
        'Food': {vals: [107,43,15,236,99], color: '#5ab4ac'},
        'Transport': {vals: [61,45,9,291,94], color: '#af8dc3'},
        'Housing': {vals: [75,48,17,253,107], color: '#f4a460'}
    };
    var stanceNames = ['pro','against','against particular but pro','neither/mixed','pro but lack of options'];

    sectors.forEach(function(s) {
        labels.push(s);
        parents.push('Stances');
        values.push(0);
        colors.push(sectorData[s].color + '40');
        texts.push('500 comments');

        stanceNames.forEach(function(st, i) {
            var v = sectorData[s].vals[i];
            labels.push(s + ' - ' + st);
            parents.push(s);
            values.push(v);
            colors.push(stanceColors[st]);
            texts.push('<b>' + v + '</b> (' + (v/5).toFixed(0) + '%)');
        });
    });

    Plotly.newPlot('treemap-stance', [{
        type: 'treemap', labels: labels, parents: parents, values: values,
        text: texts, textinfo: 'label+text', branchvalues: 'total',
        marker: {colors: colors, line: {color: '#0a1628', width: 2}},
        textfont: {color: '#fff', size: 10},
        pathbar: {visible: true, textfont: {size: 10}},
        hovertemplate: '<b>%{label}</b><br>%{text}<extra></extra>',
    }], Object.assign({}, darkLayout, {height: 450, margin: {t: 30, b: 10, l: 10, r: 10}}), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 3: REFERENCE GROUP - Animated Bubble Pack (D3)
// ═══════════════════════════════════════════════════════════════
var bubbleData = {
    categories: ['family','partner/spouse','friends','coworkers','neighbors','local community','political tribe','online community','other reddit user'],
    colors: ['#a8b8c2','#c49fc4','#8fbfd9','#c692c6','#7cadc6','#7cc2b8','#a8b1b8','#b8b8b8','#ffa8c2'],
    Food:      [22, 12, 50, 15, 3, 10, 0, 6, 19],
    Transport: [6, 1, 2, 56, 6, 17, 0, 2, 20],
    Housing:   [10, 2, 3, 58, 8, 24, 3, 1, 16],
};

function initBubbles() {
    var container = document.getElementById('bubble-pack');
    if (!container) return;
    container.innerHTML = '';

    var width = container.clientWidth || 700;
    var height = 480;
    var svg = d3.select('#bubble-pack').append('svg').attr('viewBox', '0 0 ' + width + ' ' + height);

    var sectors = ['Food', 'Transport', 'Housing'];
    var sectorColors = {Food: '#5ab4ac', Transport: '#af8dc3', Housing: '#f4a460'};
    var colWidth = width / 3;

    // Create nodes for each sector
    var allNodes = [];
    sectors.forEach(function(sector, si) {
        var cx = colWidth * si + colWidth / 2;

        // Sector label
        svg.append('text').attr('x', cx).attr('y', 25)
           .attr('class', 'bubble-sector-label').attr('text-anchor', 'middle')
           .text(sector.toUpperCase());

        bubbleData.categories.forEach(function(cat, ci) {
            var val = bubbleData[sector][ci];
            if (val === 0) return;
            allNodes.push({
                sector: sector, sectorIdx: si, category: cat,
                value: val, r: Math.sqrt(val) * 4.5,
                color: bubbleData.colors[ci],
                cx: cx, cy: height / 2,
                x: cx + (Math.random() - 0.5) * 80,
                y: height / 2 + (Math.random() - 0.5) * 80,
            });
        });
    });

    // Draw separator lines
    svg.selectAll('.sep').data([1, 2]).enter().append('line')
        .attr('x1', function(d) { return colWidth * d; })
        .attr('x2', function(d) { return colWidth * d; })
        .attr('y1', 40).attr('y2', height - 10)
        .attr('stroke', '#1a3050').attr('stroke-width', 1).attr('stroke-dasharray', '4,4');

    // Draw bubbles
    var bubbles = svg.selectAll('.bubble').data(allNodes).enter().append('g');

    bubbles.append('circle')
        .attr('cx', function(d) { return d.x; })
        .attr('cy', function(d) { return d.y; })
        .attr('r', 0)
        .attr('fill', function(d) { return d.color; })
        .attr('fill-opacity', 0.85)
        .attr('stroke', function(d) { return d.color; })
        .attr('stroke-opacity', 0.3)
        .attr('stroke-width', 2)
        .transition().duration(800).delay(function(d, i) { return i * 30; })
        .attr('r', function(d) { return d.r; });

    bubbles.append('text')
        .attr('x', function(d) { return d.x; })
        .attr('y', function(d) { return d.y - 2; })
        .attr('class', 'bubble-label')
        .attr('font-size', function(d) { return Math.max(8, Math.min(d.r * 0.6, 13)) + 'px'; })
        .attr('opacity', 0)
        .text(function(d) { return d.value >= 8 ? d.category : ''; })
        .transition().duration(600).delay(function(d, i) { return 800 + i * 30; })
        .attr('opacity', 1);

    bubbles.append('text')
        .attr('x', function(d) { return d.x; })
        .attr('y', function(d) { return d.y + (d.r > 20 ? 12 : 8); })
        .attr('class', 'bubble-label')
        .attr('font-size', function(d) { return Math.max(9, Math.min(d.r * 0.55, 14)) + 'px'; })
        .attr('font-weight', '700')
        .attr('opacity', 0)
        .text(function(d) { return d.value; })
        .transition().duration(600).delay(function(d, i) { return 800 + i * 30; })
        .attr('opacity', 1);

    // Force simulation for organic movement
    var sim = d3.forceSimulation(allNodes)
        .force('x', d3.forceX(function(d) { return d.cx; }).strength(0.08))
        .force('y', d3.forceY(height / 2 + 20).strength(0.05))
        .force('collide', d3.forceCollide(function(d) { return d.r + 3; }).strength(0.9))
        .force('charge', d3.forceManyBody().strength(-2))
        .alpha(0.8)
        .on('tick', function() {
            bubbles.select('circle').attr('cx', function(d) { return d.x; }).attr('cy', function(d) { return d.y; });
            bubbles.selectAll('text').attr('x', function(d) { return d.x; });
            bubbles.select('text:nth-child(2)').attr('y', function(d) { return d.y - 2; });
            bubbles.select('text:nth-child(3)').attr('y', function(d) { return d.y + (d.r > 20 ? 12 : 8); });
        });

    // Build legend
    var legendEl = document.getElementById('bubble-legend');
    legendEl.innerHTML = '';
    bubbleData.categories.forEach(function(cat, i) {
        legendEl.innerHTML += '<span class="legend-item"><span class="legend-dot" style="background:' + bubbleData.colors[i] + '"></span>' + cat + '</span>';
    });
}

// ═══════════════════════════════════════════════════════════════
// TAB 3: REFERENCE GROUP - Treemap
// ═══════════════════════════════════════════════════════════════
(function() {
    var cats = ['family','partner/spouse','friends','coworkers','neighbors','local community','online community','other reddit user'];
    var catColors = ['#a8b8c2','#c49fc4','#8fbfd9','#c692c6','#7cadc6','#7cc2b8','#b8b8b8','#ffa8c2'];
    var food = [22,12,50,15,3,10,6,19];
    var transport = [6,1,2,56,6,17,2,20];
    var housing = [10,2,3,58,8,24,1,16];

    var labels = ['Reference Groups'];
    var parents = [''];
    var values = [0];
    var colors = ['#0f1d33'];

    cats.forEach(function(cat, i) {
        var total = food[i] + transport[i] + housing[i];
        labels.push(cat + '<br><b>' + total + '</b>');
        parents.push('Reference Groups');
        values.push(total);
        colors.push(catColors[i]);
    });

    Plotly.newPlot('treemap-ref', [{
        type: 'treemap', labels: labels, parents: parents, values: values,
        branchvalues: 'total', textinfo: 'label',
        marker: {colors: colors, line: {color: '#0a1628', width: 3}},
        textfont: {color: '#fff', size: 13},
        hovertemplate: '<b>%{label}</b><br>Count: %{value}<extra></extra>',
    }], Object.assign({}, darkLayout, {height: 450, margin: {t: 10, b: 10, l: 10, r: 10}}), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 4: NORMS OVERVIEW - Heatmap
// ═══════════════════════════════════════════════════════════════
(function() {
    // Percentages for each norm type across sectors
    var sectors = ['Food', 'Transport', 'Housing'];
    var metrics = [
        'Descriptive: Present', 'Descriptive: Absent', 'Descriptive: Unclear',
        'Injunctive: Present', 'Injunctive: Unclear', 'Injunctive: Absent',
        '2nd Order: Strong', '2nd Order: Weak', '2nd Order: None'
    ];
    var z = [
        [28.4, 15.4, 14.0],
        [26.6, 70.0, 65.6],
        [45.0, 14.6, 20.4],
        [8.0, 5.4, 9.4],
        [5.0, 1.4, 2.2],
        [87.0, 93.2, 88.4],
        [11.6, 10.8, 16.6],
        [20.8, 15.8, 16.6],
        [67.6, 73.4, 66.8],
    ];
    var textvals = z.map(function(row) { return row.map(function(v) { return v.toFixed(1) + '%'; }); });

    Plotly.newPlot('heatmap-norms', [{
        z: z, x: sectors, y: metrics, type: 'heatmap',
        colorscale: [[0, '#0a1628'], [0.15, '#1a3050'], [0.4, '#5ab4ac'], [0.7, '#8fcc8f'], [1, '#ffe87a']],
        text: textvals, texttemplate: '%{text}', textfont: {size: 11, color: '#fff'},
        hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
        showscale: true, colorbar: {title: '%', titleside: 'right', tickfont: {color: '#6a8caf'}, titlefont: {color: '#6a8caf'}},
    }], Object.assign({}, darkLayout, {
        height: 450,
        margin: {l: 160, r: 80, t: 20, b: 50},
        xaxis: {side: 'bottom', tickfont: {size: 12}},
        yaxis: {tickfont: {size: 10}, autorange: 'reversed'},
    }), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 4: NORMS OVERVIEW - Lollipop
// ═══════════════════════════════════════════════════════════════
(function() {
    var items = [
        {label: 'Descriptive: Present', food: 28.4, transport: 15.4, housing: 14.0},
        {label: 'Injunctive: Present', food: 8.0, transport: 5.4, housing: 9.4},
        {label: '2nd Order: Strong', food: 11.6, transport: 10.8, housing: 16.6},
        {label: '2nd Order: Weak', food: 20.8, transport: 15.8, housing: 16.6},
        {label: 'Descriptive: Unclear', food: 45.0, transport: 14.6, housing: 20.4},
        {label: 'Injunctive: Unclear', food: 5.0, transport: 1.4, housing: 2.2},
    ];

    var traces = [];
    var sectorConf = [{key:'food', name:'Food', color:'#5ab4ac'}, {key:'transport', name:'Transport', color:'#af8dc3'}, {key:'housing', name:'Housing', color:'#f4a460'}];

    sectorConf.forEach(function(sc) {
        // Stem lines
        traces.push({
            y: items.map(function(it) { return it.label; }),
            x: items.map(function(it) { return it[sc.key]; }),
            type: 'scatter', mode: 'lines', line: {color: sc.color, width: 0}, showlegend: false,
            hoverinfo: 'skip',
        });
        // Dots
        traces.push({
            y: items.map(function(it) { return it.label; }),
            x: items.map(function(it) { return it[sc.key]; }),
            type: 'scatter', mode: 'markers+text', name: sc.name,
            marker: {size: 14, color: sc.color, line: {color: '#0a1628', width: 2}},
            text: items.map(function(it) { return it[sc.key].toFixed(0); }),
            textposition: 'top center', textfont: {size: 9, color: sc.color},
            hovertemplate: '<b>' + sc.name + '</b><br>%{y}: %{x:.1f}%<extra></extra>',
        });
    });

    Plotly.newPlot('lollipop-norms', traces, Object.assign({}, darkLayout, {
        height: 450,
        margin: {l: 140, r: 30, t: 20, b: 40},
        xaxis: {title: '%', showgrid: true, gridcolor: '#1a3050', zeroline: false, range: [0, 50]},
        yaxis: {tickfont: {size: 10}},
        legend: {orientation: 'h', x: 0.5, xanchor: 'center', y: -0.08, font: {size: 10}},
    }), plotlyConfig);
})();

// ═══════════════════════════════════════════════════════════════
// TAB 4: NORMS OVERVIEW - Radial Bars
// ═══════════════════════════════════════════════════════════════
(function() {
    var theta = ['Desc: Present', 'Desc: Absent', 'Desc: Unclear',
                 'Inj: Present', 'Inj: Unclear', 'Inj: Absent',
                 '2nd: Strong', '2nd: Weak', '2nd: None'];
    var food = [28.4, 26.6, 45.0, 8.0, 5.0, 87.0, 11.6, 20.8, 67.6];
    var transport = [15.4, 70.0, 14.6, 5.4, 1.4, 93.2, 10.8, 15.8, 73.4];
    var housing = [14.0, 65.6, 20.4, 9.4, 2.2, 88.4, 16.6, 16.6, 66.8];

    Plotly.newPlot('radial-norms', [
        {type: 'scatterpolar', r: food, theta: theta, name: 'Food', fill: 'toself',
         fillcolor: 'rgba(90,180,172,0.15)', line: {color: '#5ab4ac', width: 2},
         marker: {size: 6, color: '#5ab4ac'}},
        {type: 'scatterpolar', r: transport, theta: theta, name: 'Transport', fill: 'toself',
         fillcolor: 'rgba(175,141,195,0.15)', line: {color: '#af8dc3', width: 2},
         marker: {size: 6, color: '#af8dc3'}},
        {type: 'scatterpolar', r: housing, theta: theta, name: 'Housing', fill: 'toself',
         fillcolor: 'rgba(244,164,96,0.15)', line: {color: '#f4a460', width: 2},
         marker: {size: 6, color: '#f4a460'}},
    ], Object.assign({}, darkLayout, {
        height: 450,
        margin: {t: 40, b: 40, l: 60, r: 60},
        polar: {
            bgcolor: 'rgba(0,0,0,0)',
            radialaxis: {color: '#4a6a8a', gridcolor: '#1a3050', range: [0, 100], ticksuffix: '%', tickfont: {size: 8}},
            angularaxis: {color: '#b0c4d8', gridcolor: '#1a3050', tickfont: {size: 9}},
        },
        legend: {orientation: 'h', x: 0.5, xanchor: 'center', y: -0.05, font: {size: 10}},
    }), plotlyConfig);
})();

// Init on load
window.addEventListener('load', function() { setTimeout(initBubbles, 200); });
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════
# Generate waffle charts for Tab 1
# ═══════════════════════════════════════════════════════════════════

def make_waffle(sector, yes_count, total, yes_color, no_color):
    """Generate HTML for a waffle chart (100 cells = 5 rows x 20 cols)."""
    pct = round(yes_count / total * 100)
    cells_yes = round(pct)  # number of colored cells out of 100

    html_parts = [f'<div style="text-align:center;"><span style="color:#b0c4d8;font-size:0.9em;">{sector}</span>'
                  f' <span style="color:{yes_color};font-weight:700;font-size:1.1em;">{pct}%</span></div>']
    html_parts.append('<div class="waffle-grid">')

    for i in range(100):
        color = yes_color if i < cells_yes else no_color
        html_parts.append(f'<div class="waffle-cell" style="background:{color};" title="{sector}: {"Yes" if i < cells_yes else "No"}"></div>')

    html_parts.append('</div>')
    return '\n'.join(html_parts)

waffle_html = make_waffle("Food", 201, 500, "#5ab4ac", "#1a3050")
waffle_html += make_waffle("Transport", 58, 500, "#af8dc3", "#1a3050")
waffle_html += make_waffle("Housing", 81, 500, "#f4a460", "#1a3050")

html = html.replace("WAFFLE_PLACEHOLDER", waffle_html)

# Write output
with open("temp.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Generated temp.html with 4 tabs:")
print("  Tab 1: Norm Signal (gauge, treemap, waffle)")
print("  Tab 2: Author Stance (sankey, butterfly, treemap)")
print("  Tab 3: Reference Group (animated bubbles, treemap)")
print("  Tab 4: Norms Overview (heatmap, lollipop, radar)")
print("\nOpen temp.html in browser to view!")
