# Replicated Preference Reasoning Study

This study compares Qwen 3.6 27B choices under answer-only, brief-rationale,
and long-reasoning conditions.

## Design

- 10 aesthetic and 10 utility preference pairs.
- 10 repetitions per preference and condition.
- Exactly five canonical (`option 1` then `option 2`) and five reversed
  (`option 2` then `option 1`) presentations in every cell.
- Displayed `A`/`B` answers are mapped back to canonical `option_1`/`option_2`
  choices before analysis.
- Temperature 0.7 with a distinct, paired seed for each preference/repetition.
- 600 successful calls in total.

## Conditions

- `none`: reasoning disabled; answer tag only.
- `short`: reasoning disabled; one visible rationale targeted at 30 words, with
  a 96-token hard output cap. Observed rationales were 13–31 words.
- `long`: high reasoning effort with a 4,200-token completion cap.

The brief rationale is used because OpenRouter effort-based reasoning has a
minimum allocation around 1,024 tokens, which is not genuinely short.

## Outputs

- `raw-results.json`: responses, reasoning traces, canonical and displayed
  choices, order, seeds, token usage, and cost. It contains no API key.
- `summary.json`: aggregated preference rates, Wilson intervals, matched
  condition transitions, and order-bias metrics.
- `preference-rates.csv`: one row per preference and condition.
- `condition-transitions.csv`: matched switch rates by domain and condition.

From the repository root, run
[`../../run_qwen_preference_reasoning_study.py`](../../run_qwen_preference_reasoning_study.py)
to resume collection and
[`../../summarize_preference_reasoning_study.py`](../../summarize_preference_reasoning_study.py)
to regenerate summaries.
