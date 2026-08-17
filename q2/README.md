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
and `summarize_qwen_preference_reasoning_study.py` to regenerate its summaries.
