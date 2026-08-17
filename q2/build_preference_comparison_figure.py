#!/usr/bin/env python3
"""Build an interactive Qwen/Gemini reasoning-depth comparison as HTML."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median


Q2 = Path("q2")
RESULTS = Q2 / "results"
MODELS = {
    "Qwen 3.6 27B": RESULTS / "qwen-preference-reasoning-depth",
    "Gemini 2.5 Flash": RESULTS / "gemini25-preference-reasoning-depth",
}
CONDITIONS = ["none", "short", "long"]
DOMAINS = ["aesthetic", "utility"]
COLORS = {"Qwen 3.6 27B": "#6C5CE7", "Gemini 2.5 Flash": "#00A896"}


def load() -> dict:
    return {
        model: {
            "summary": json.loads((root / "summary.json").read_text()),
            "raw": json.loads((root / "raw-results.json").read_text()),
        }
        for model, root in MODELS.items()
    }


def generated_tokens(model: str, row: dict) -> int:
    completion = row.get("completion_tokens") or 0
    reasoning = row.get("reasoning_tokens") or 0
    # OpenRouter's Qwen completion count already contains reasoning tokens;
    # Vertex reports candidate and thought tokens separately.
    return completion if model.startswith("Qwen") else completion + reasoning


def build_payload(data: dict) -> dict:
    rates = []
    transitions = []
    tokens = []
    effects = []
    for model, bundle in data.items():
        summary = bundle["summary"]
        rows = [row for row in bundle["raw"]["records"] if "canonical_choice" in row]
        for domain in DOMAINS:
            for condition in CONDITIONS:
                cell = next(
                    item for item in summary["domain_conditions"]
                    if item["domain"] == domain and item["condition"] == condition
                )
                subset = [row for row in rows if row["domain"] == domain and row["condition"] == condition]
                rates.append({
                    "model": model,
                    "domain": domain,
                    "condition": condition,
                    "rate": cell["option_2"]["rate"],
                })
                tokens.append({
                    "model": model,
                    "domain": domain,
                    "condition": condition,
                    "generated": median(generated_tokens(model, row) for row in subset),
                    "reasoning": median((row.get("reasoning_tokens") or 0) for row in subset),
                })
            transition = next(
                item for item in summary["transitions"]
                if item["domain"] == domain and item["from"] == "none" and item["to"] == "long"
            )
            transitions.append({"model": model, **transition})
        for effect in summary["preference_effects"]:
            effects.append({"model": model, **effect})
    return {"rates": rates, "transitions": transitions, "tokens": tokens, "effects": effects, "colors": COLORS}


def main() -> None:
    payload = json.dumps(build_payload(load()))
    output = RESULTS / "qwen-gemini-reasoning-depth-comparison.html"
    output.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reasoning depth and model preferences</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ --ink:#18212f; --muted:#667085; --paper:#f5f7fb; --card:#fff; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }}
    main {{ max-width:1320px; margin:auto; padding:42px 24px 64px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,46px); letter-spacing:-.035em; }}
    .lede {{ max-width:850px; color:var(--muted); font-size:17px; line-height:1.55; margin:0 0 28px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .card {{ background:var(--card); border:1px solid #e4e7ec; border-radius:16px; padding:16px; box-shadow:0 3px 14px rgba(16,24,40,.04); }}
    .wide {{ grid-column:1/-1; }}
    .plot {{ width:100%; height:440px; }}
    .note {{ color:var(--muted); font-size:13px; line-height:1.45; padding:2px 10px 8px; }}
    @media(max-width:850px) {{ .grid {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} }}
  </style>
</head>
<body><main>
  <h1>Reasoning depth changes models differently</h1>
  <p class="lede">Matched comparison across 20 preference pairs, three deliberation conditions, 10 repetitions, and balanced A/B versus B/A presentation. Hover for exact values; click legends to isolate a model.</p>
  <section class="grid">
    <article class="card"><div id="rates" class="plot"></div><p class="note">Canonical option 2 was constructed as the more reflective alternative in many scenarios, so this is not a general desirability score.</p></article>
    <article class="card"><div id="switches" class="plot"></div><p class="note">Directions are canonical option changes within matched seed/order pairs.</p></article>
    <article class="card"><div id="tokens" class="plot"></div><p class="note">Generated tokens are provider completion tokens for Qwen and candidate + thought tokens for Gemini. Hover also shows median reasoning tokens.</p></article>
    <article class="card"><div id="effects" class="plot"></div><p class="note">Positive values mean long reasoning increased selection of canonical option 2.</p></article>
  </section>
</main>
<script>
const D = {payload};
const order = ['none','short','long'];
const labels = {{none:'Answer only',short:'Brief rationale',long:'Long reasoning'}};
const base = {{paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{{family:'Inter, system-ui, sans-serif',color:'#18212f'}},margin:{{l:62,r:20,t:62,b:70}},hoverlabel:{{bgcolor:'#18212f',font:{{color:'#fff'}}}}}};

const rateTraces = [];
for (const model of Object.keys(D.colors)) for (const domain of ['aesthetic','utility']) {{
  const rows = D.rates.filter(d=>d.model===model && d.domain===domain);
  rateTraces.push({{type:'scatter',mode:'lines+markers',name:`${{model}} · ${{domain}}`,x:order.map(labels),y:order.map(c=>rows.find(r=>r.condition===c).rate*100),line:{{color:D.colors[model],dash:domain==='utility'?'solid':'dot',width:3}},marker:{{size:9}},hovertemplate:'%{{fullData.name}}<br>%{{x}}: %{{y:.0f}}%<extra></extra>'}});
}}
Plotly.newPlot('rates',rateTraces,{{...base,title:{{text:'Preference rate by reasoning condition',x:.02}},yaxis:{{title:'Chose canonical option 2 (%)',range:[0,105],gridcolor:'#edf0f5'}},xaxis:{{title:''}},legend:{{orientation:'h',y:-.23}}}},{{responsive:true,displaylogo:false}});

const switchTraces=[];
for (const model of Object.keys(D.colors)) {{
  const rows=D.transitions.filter(d=>d.model===model);
  switchTraces.push({{type:'bar',name:`${{model}}: 1→2`,x:rows.map(r=>r.domain),y:rows.map(r=>r.option_1_to_option_2),marker:{{color:D.colors[model]}},offsetgroup:model,legendgroup:model,hovertemplate:'%{{fullData.name}}<br>%{{x}}: %{{y}} cases<extra></extra>'}});
  switchTraces.push({{type:'bar',name:`${{model}}: 2→1`,x:rows.map(r=>r.domain),y:rows.map(r=>-r.option_2_to_option_1),marker:{{color:D.colors[model],opacity:.42}},offsetgroup:model,legendgroup:model,hovertemplate:'%{{fullData.name}}<br>%{{x}}: %{{customdata}} cases<extra></extra>',customdata:rows.map(r=>r.option_2_to_option_1)}});
}}
Plotly.newPlot('switches',switchTraces,{{...base,barmode:'relative',title:{{text:'Matched answer-only → long reversals',x:.02}},yaxis:{{title:'Cases (direction shown by sign)',zeroline:true,zerolinecolor:'#98a2b3',gridcolor:'#edf0f5'}},xaxis:{{title:''}},legend:{{orientation:'h',y:-.23}}}},{{responsive:true,displaylogo:false}});

const tokenTraces=[];
for (const model of Object.keys(D.colors)) for (const domain of ['aesthetic','utility']) {{
  const rows=D.tokens.filter(d=>d.model===model && d.domain===domain);
  tokenTraces.push({{type:'bar',name:`${{model}} · ${{domain}}`,x:order.map(labels),y:order.map(c=>rows.find(r=>r.condition===c).generated),customdata:order.map(c=>rows.find(r=>r.condition===c).reasoning),marker:{{color:D.colors[model],opacity:domain==='utility'?1:.48}},offsetgroup:`${{model}}-${{domain}}`,hovertemplate:'%{{fullData.name}}<br>%{{x}}<br>Median generated: %{{y:.0f}}<br>Median reasoning: %{{customdata:.0f}}<extra></extra>'}});
}}
Plotly.newPlot('tokens',tokenTraces,{{...base,barmode:'group',title:{{text:'Median generated tokens',x:.02}},yaxis:{{title:'Median tokens',gridcolor:'#edf0f5'}},xaxis:{{title:''}},legend:{{orientation:'h',y:-.23}}}},{{responsive:true,displaylogo:false}});

const prefs=[...new Set(D.effects.map(d=>d.preference_id))];
const effectTraces=Object.keys(D.colors).map(model=>{{const rows=prefs.map(p=>D.effects.find(d=>d.model===model&&d.preference_id===p));return {{type:'bar',orientation:'h',name:model,y:prefs,x:rows.map(r=>r.long_minus_none*100),marker:{{color:D.colors[model]}},hovertemplate:'%{{fullData.name}}<br>%{{y}}: %{{x:+.0f}} pp<extra></extra>'}};}});
Plotly.newPlot('effects',effectTraces,{{...base,barmode:'group',title:{{text:'Per-preference long − answer-only change',x:.02}},xaxis:{{title:'Percentage-point change',zeroline:true,zerolinecolor:'#98a2b3',gridcolor:'#edf0f5'}},yaxis:{{automargin:true,tickfont:{{size:10}}}},legend:{{orientation:'h',y:-.17}},margin:{{l:145,r:20,t:62,b:70}}}},{{responsive:true,displaylogo:false}});
</script></body></html>
""")
    print(output)


if __name__ == "__main__":
    main()
