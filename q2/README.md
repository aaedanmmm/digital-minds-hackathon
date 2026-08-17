# Question 2: Reasoning depth and preferences

Question 2 asks whether longer deliberation changes a model's preferences and,
when it does, whether the model calculates a utility or derives its choice from
other beliefs and values.

## Qwen 3.6 27B

[`results/qwen-preference-reasoning-depth/`](results/qwen-preference-reasoning-depth/)
contains the complete 600-response study, quantitative summaries, and a
case-level qualitative audit of every answer-only to long-reasoning reversal.

- [`trace-cause-analysis.md`](results/qwen-preference-reasoning-depth/trace-cause-analysis.md): interpretation of why choices changed.
- [`none-to-long-classifications.csv`](results/qwen-preference-reasoning-depth/none-to-long-classifications.csv): all 26 changed cases and their mechanism labels.
- [`raw-results.json`](results/qwen-preference-reasoning-depth/raw-results.json): responses and available reasoning traces.
- [`quantitative-report.md`](results/qwen-preference-reasoning-depth/quantitative-report.md): preference-rate and transition results.

Use `run_qwen_preference_reasoning_study.py` to collect/resume the experiment
and `summarize_preference_reasoning_study.py` to regenerate its summaries.

## Gemini 2.5 Flash

[`results/gemini25-preference-reasoning-depth/`](results/gemini25-preference-reasoning-depth/)
contains the 600-response Vertex AI replication, quantitative report, raw
responses and thought summaries, and case-level qualitative classifications.

- [`quantitative-report.md`](results/gemini25-preference-reasoning-depth/quantitative-report.md): matched preference shifts and manipulation checks.
- [`trace-cause-analysis.md`](results/gemini25-preference-reasoning-depth/trace-cause-analysis.md): calculation-versus-belief audit.
- [`study-design.md`](results/gemini25-preference-reasoning-depth/study-design.md): exact Vertex configuration and parser provenance.

Use `run_gemini25_preference_reasoning_study.py` to collect/resume the Vertex
replication. The generic `summarize_preference_reasoning_study.py` accepts the Gemini result directory via
`--root`.

## Cross-model comparison

- [`results/cross-model-analysis.md`](results/cross-model-analysis.md): Qwen/Gemini agreement, mechanism, and methodological comparison.
- [`results/qwen-gemini-reasoning-depth-comparison.html`](results/qwen-gemini-reasoning-depth-comparison.html): interactive preference-rate, reversal, token, and item-effect plots.
- `build_preference_comparison_figure.py`: regenerate the HTML figure.
