# reasoning_depth

Does reasoning depth change *what* a model prefers?

The sibling project ([thunderingluck/digital-minds](https://github.com/thunderingluck/digital-minds))
asked whether LLM preferences are **transitive** and answered decisively: yes.
Analysis of its 2,430 trials showed cycle rate pinned at **0–0.8%** in all nine
cells, with ranking correlation across reasoning budgets at **Spearman 0.976–1.000**.
Transitivity has no headroom left to measure.

So this asks the different question: not whether preferences are *coherent*, but
whether they **change** — across reasoning depth, and (later) across persona.

## Design

**Baseline run: neutral persona only** (no system instruction at all — "you are a
helpful assistant" is itself a persona).

```
540 prompts x 3 reasoning levels x 5 replicates = 8,100 trials per arm
```

`prompts.json` (built by `build_prompts.py`, 540 rows):

| topic | complexity levels | ordered pairs |
|---|---|---|
| quantitative | 3 / 5 / 10 attributes | 90 x 3 = 270 |
| aesthetic | haiku / sonnet / villanelle | 90 x 3 = 270 |

90 ordered pairs = all 45 unordered pairs x both presentation orders, so position
bias is measurable inside every cell.

### Reasoning levels

| label | gemini (`thinking_level`) | openai (scratchpad words) |
|---|---|---|
| `none` | `minimal` | 0 — answer immediately |
| `low` | `low` | ~50 |
| `high` | `high` | ~200 |

The **`none`** level is the sharpest test of an "innate" preference: no
deliberation at all.

**Why `gemini-3.6-flash`.** Gemini 3.x replaced the numeric `thinking_budget`
with a `thinking_level` enum; only `gemini-3.5-flash` still accepts the numeric
form, and it costs 2.3x more. 3.6-flash keeps the `minimal` level that the
`none` condition needs — **`gemini-3.7-flash` is excluded because it dropped
`minimal`**, which would eliminate the zero-reasoning condition entirely (it is
also same-price and measurably more hallucination-prone, 64.5% vs 55.6%).

`thoughts_tokens` is still returned under the enum, so the measured-vs-requested
manipulation check survives. `minimal` is not guaranteed to be literally zero
thinking tokens the way `thinking_budget=0` was — **check the smoke run.**

For exact replication of the sibling's numeric axis:

```bash
python run_experiment.py --arm gemini --numeric-budget --model gemini-3.5-flash
```

That writes to `results/gemini_numeric.jsonl`, kept separate because `trial_id`s
collide across the two schemes and resume would otherwise conflate them.

### Two arms

**`gemini`** — native thinking, one call per trial. Constrained decoding
(`response_schema` with `Literal[first, second]`) makes the answer physically one
letter; the model cannot hedge or explain. Actual spend recorded as
`thoughts_tokens`.

**`openai`** — `gpt-4.1-mini`, written scratchpad **plus logprobs**, two calls:

1. scratchpad, with an **instructed** word budget the model can plan against
2. scratchpad appended, choice requested with `max_tokens=1` and `logprobs=True`

`max_tokens` is only a safety net here, never the binding constraint. A hard cap
truncates silently and the model has no idea it exists — it would generate as if
unconstrained and get cut mid-sentence, sometimes before answering at all. An
instructed budget is a manipulation the model can actually plan within, and
measured-vs-instructed length becomes a free compliance check.

Logprobs matter because in the sibling data **35–43 of 45 pairs were unanimous**.
Where `p` is pinned at 1.0, the answer-token margin is the only place a small
reasoning effect can still show. No OpenAI *reasoning* model exposes logprobs
(nor raw CoT), which is precisely why this arm uses a non-reasoning model with an
explicit scratchpad.

### Two design fixes over the sibling

**Letter randomization.** The sibling labelled houses by fixed letters `A`–`J`,
and house A won **0 of 2,430 trials** — an alphabetical prior inseparable from
preference. Here each prompt draws its own letter→house permutation, recorded in
`letter_map`. A build-time assertion fails if any house is ever frozen to one
letter.

**No majority-flip counting.** Simulation shows that at 3 replicates, two
conditions with *identical* true probabilities produce **~4.1 spurious
majority-flips per 45 pairs** — and the sibling's headline effect was 5 flips. At
R=3 a per-pair probability can only be {0, ⅓, ⅔, 1}, so their reported |Δp| of
0.011 is one flipped trial in 135. Hence **R=5** (resolution 0.200, vs their
0.333) and a paired test on continuous probabilities, never a flip count.

## Cost

**Thinking tokens bill as output.** The sibling's metered usage was 841k thought
tokens against 15k answer tokens — reasoning is ~98% of the output bill. The
6-token answer is free; the thinking is the entire cost. Their run cost **$8.69
for 2,430 trials** on `gemini-3.5-flash` ($1.50/$9.00 per 1M).

| arm | model | trials | calls | cost |
|---|---|---|---|---|
| gemini | `gemini-3.6-flash` ($0.75/$3.75) | 8,100 | 8,100 | **$7.69** |
| openai | `gpt-4.1-mini` ($0.40/$1.60) | 8,100 | 13,500 | **$3.08** |
| | | | | **~$11** |

Alternatives for the Gemini arm: `gemini-3-flash-preview` $5.96,
`gemini-3.1-flash-lite` $2.98, `gemini-3.5-flash` $17.89 (numeric axis).
Note 3.6-flash's price is promotional through Dec 31 2026 and doubles after.

Our per-trial cost runs below the sibling's because one third of trials use
`thinking_budget=0`, where their cheapest level was 512. The `high` level alone
is ~61% of the Gemini bill — if calibration shows 4096 doesn't differ from 256 in
actual spend (the sibling measured 4096→404 and 8192→410, i.e. saturation),
dropping it roughly halves the run.

### Why more trials than the sibling's 2,430

They ran **houses only at R=3**. Their repo contains a poems file, but no code
references it and 0 of 2,430 result rows are poems — it was staged, never run
(and is malformed: 110 villanelle rows where 90 is the complete set). We run both
topics (2x) at R=5 (1.7x) = 3.3x their trial count.

## Run

```bash
pip install google-genai openai pydantic python-dotenv
cat > .env <<'EOF'
GEMINI_API_KEY=...
OPENAI_API_KEY=...
EOF

python build_prompts.py                       # -> prompts.json (540 rows)
python build_prompts.py --check               # re-validate without rewriting

python run_experiment.py --arm gemini --dry-run
python run_experiment.py --arm gemini --smoke
python run_experiment.py --arm gemini --reps 5 --resume
python run_experiment.py --arm openai --reps 5 --resume
```

**Nothing is ever recomputed.** Each trial is appended to `results/<arm>.jsonl`
and flushed as it completes, and `--resume` skips any `trial_id` already recorded
with an answer. Killing the run loses only in-flight requests; a torn final line
is tolerated on reload. Failed rows retry on the next resume, so re-running
`--resume` until it reports `nothing to do` is always safe.

## Output

One JSONL row per trial. Conditions (`topic`, `complexity`, `reasoning_level`,
`persona`, `replicate`), the pair (`first`, `second`, `pair_id`, `letter_map`),
the full verbatim `raw_response`, and:

| field | |
|---|---|
| `choice_letter` | the answer as displayed |
| `choice_item` | mapped back to the internal item id — **analyse on this, never on letters** |
| `thoughts_tokens` | gemini: actual thinking spend vs the requested budget |
| `logprobs` | openai: answer-token distribution, `top_logprobs=20` |
| `scratchpad`, `scratchpad_words` | openai: the reasoning text and its measured length |

`pair_id` is order-invariant, so both presentation orders of a pair share one id.
