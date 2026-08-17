# Qwen Q1 and Q2 studies

OpenRouter-based studies of Qwen 3.6 27B for research questions 1 and 2. Q3
(persona elicitation) lives in [`q3/`](../q3/) and runs on Vertex AI instead.

## Layout

```
scripts/   runners and summarizers, stdlib only, resumable, never write the API key
prompts/   prompt templates and scenario sets
results/   raw responses, summaries, and written reports per study
```

## Run from this directory

Every script defaults to paths relative to this folder — `results/aesthetic-reasoning/raw-results.json`
and similar — so run them from here, not from the repository root:

```bash
cd qwen-q1-q2
python scripts/run_aesthetic_reasoning_study.py
```

`OPENROUTER_API_KEY` is read from `.env` or the environment and is never
written to disk or into any results file.

## Studies

**`results/aesthetic-reasoning/`** — deterministic pilot of aesthetic choices
under three reasoning settings (no reasoning, low effort, high effort) across
ten paired scenarios, at temperature 0 with seed 42. Single seed, so it is a
pilot rather than an estimate of a stable preference. Retains
`raw-results-before-short-budget-fix.json` for comparison.

**`results/preference-reasoning-10rep/`** — 600 records: 20 preference pairs
(10 aesthetic, 10 utility) x 3 deliberation conditions x 10 repetitions at
temperature 0.7. Each preference/condition uses five AB and five BA
presentations, with displayed choices mapped back to canonical option IDs, so
presentation order is balanced rather than confounded with preference.

## Related documents

- Q1 plan: [`docs/superpowers/plans/2026-08-16-preference-trace-pipeline.md`](../docs/superpowers/plans/2026-08-16-preference-trace-pipeline.md)
- Research questions and methods: [root README](../README.md)
