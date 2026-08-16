# Aesthetic Reasoning-Budget Study

This directory contains one deterministic pilot of Qwen 3.6 27B's aesthetic
choices under three reasoning settings, using ten paired scenarios.

## Files

- `raw-results.json` — prompt-level responses, returned reasoning traces, token
  usage, and parsed choices. It contains no API key.
- `condition-comparison.csv` — one row per scenario, with choices and token
  counts aligned across `none`, `short`, and `long` conditions.
- `summary.json` — condition totals, pairwise choice-change rates, and the
  scenario-level data used in the figure.

## Conditions

All calls used temperature `0` and seed `42`.

- `none`: reasoning disabled; response constrained to an answer tag.
- `short`: reasoning disabled, one visible reason capped at 30 words, with a
  96-token completion cap.
- `long`: high reasoning effort, with a 4,200-token completion cap.

The short condition uses a visible brief rationale because OpenRouter's
effort-based reasoning allocation has a 1,024-token minimum. Calling that
minimum allocation “short” produced roughly 900 reasoning tokens in the
original pilot, so those superseded records are preserved in
`raw-results-before-short-budget-fix.json`.

Use [`scripts/run_aesthetic_reasoning_study.py`](../../scripts/run_aesthetic_reasoning_study.py)
to resume the raw collection and
[`scripts/summarize_aesthetic_reasoning_study.py`](../../scripts/summarize_aesthetic_reasoning_study.py)
to regenerate the summary files.

This is a single-seed pilot, not an estimate of a stable model preference.
