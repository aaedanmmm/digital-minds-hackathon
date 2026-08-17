# Gemini 2.5 preference-reasoning replication

This is a direct Vertex AI replication of the Qwen Q2 study using
`gemini-2.5-flash` in the `global` location.

## Design

- 10 aesthetic and 10 utility preference pairs.
- Three conditions and 10 repetitions per preference: 600 records.
- Five canonical A/B and five reversed B/A presentations in every cell.
- Displayed answers mapped back to canonical `option_1` / `option_2`.
- Temperature 0.7 and the same paired seed and order schedule as the Qwen study.

Gemini 2.5 Flash was selected instead of Pro because Flash supports a thinking
budget of zero. Gemini 2.5 Pro cannot disable thinking, so it cannot reproduce
the answer-only condition.

## Conditions

| Condition | Thinking budget | Prompt/output |
|---|---:|---|
| `none` | 0 | Answer tag only |
| `short` | 0 | Visible rationale of at most 30 words, then answer tag |
| `long` | 4,096 | Explicit deliberation prompt; returned thought summaries requested |

The long run initially used a 4,200-token shared thought-plus-final ceiling.
Twelve records completed without the requested XML answer tag and were retried
with an 8,192-token ceiling. The thinking budget remained fixed at 4,096. The
exact retry keys are recorded in `raw-results.json`.

## Trace and parser provenance

All 200 long calls report positive thought-token usage. Separate reasoning text
is present in 199 records; one record places its complete analysis in final
`content` despite reporting 2,918 thought tokens.

Final choices were parsed as:

- 596 strict `<answer>A|B</answer>` tags;
- 1 explicit declaration such as “My choice is Option B”;
- 3 explicit LaTeX boxed answers.

Every record stores `parser_format`, full Vertex response payload, usage
metadata, and canonical/displayed choices.

## Reproduction

From the repository root:

```bash
python q2/run_gemini25_preference_reasoning_study.py --project PROJECT_ID --dry-run
python q2/run_gemini25_preference_reasoning_study.py --project PROJECT_ID --smoke
python q2/run_gemini25_preference_reasoning_study.py --project PROJECT_ID --batch-size 600
python q2/summarize_preference_reasoning_study.py --root q2/results/gemini25-preference-reasoning-depth
```

The runner uses the active `gcloud` access token, writes atomically after every
record, and resumes from successful keys.
