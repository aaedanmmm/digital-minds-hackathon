# Reasoning depth and preference change

Does giving a model more room to deliberate change *what* it prefers?

## Data

| arm | rows | mechanism |
|---|---|---|
| `gemini-batch` | 8,100 | native thinking (batch; reasoning text stripped) |
| `gemini-live` | 8,100 | native thinking (live; reasoning text captured) |
| `openai` | 8,100 | written scratchpad + logprobs |

## Manipulation check

Did the reasoning levels actually separate? (`t` = completion tokens, `w` = scratchpad words)

| arm | none | low | high |
|---|---|---|---|
| `gemini-batch` | 9t | 127t | 410t |
| `gemini-live` | 9t | 145t (42w) | 393t (54w) |
| `openai` | 2t | 2t (52w) | 2t (233w) |

## Preference stability

`ordcons` = fraction of pairs where both presentation orders agree.
`posbias` = P(chose the first-shown option). 0.5 is unbiased.

| arm | cell | none | low | high |
|---|---|---|---|---|
| `gemini-batch` | aesthetic/haiku | 0.72/0.61 | 0.61/0.69 | 0.41/0.80 |
| `gemini-batch` | aesthetic/sonnet | 0.42/0.79 | 0.32/0.84 | 0.25/0.87 |
| `gemini-batch` | aesthetic/villanelle | 0.74/0.62 | 0.64/0.68 | 0.42/0.79 |
| `gemini-batch` | quantitative/10_criteria | 0.86/0.55 | 0.87/0.56 | 0.85/0.58 |
| `gemini-batch` | quantitative/3_criteria | 0.92/0.52 | 0.90/0.51 | 0.92/0.53 |
| `gemini-batch` | quantitative/5_criteria | 0.91/0.47 | 0.97/0.50 | 0.92/0.53 |
| `gemini-live` | aesthetic/haiku | 0.84/0.53 | 0.79/0.58 | 0.77/0.61 |
| `gemini-live` | aesthetic/sonnet | 0.42/0.79 | 0.36/0.82 | 0.28/0.86 |
| `gemini-live` | aesthetic/villanelle | 0.72/0.64 | 0.54/0.73 | 0.39/0.80 |
| `gemini-live` | quantitative/10_criteria | 0.83/0.58 | 0.82/0.59 | 0.85/0.58 |
| `gemini-live` | quantitative/3_criteria | 0.90/0.51 | 0.88/0.55 | 0.85/0.57 |
| `gemini-live` | quantitative/5_criteria | 0.97/0.50 | 0.97/0.50 | 0.91/0.52 |
| `openai` | aesthetic/haiku | 0.46/0.23 | 0.76/0.54 | 0.76/0.40 |
| `openai` | aesthetic/sonnet | 0.78/0.45 | 0.84/0.51 | 0.84/0.52 |
| `openai` | aesthetic/villanelle | 0.16/0.08 | 0.71/0.48 | 0.71/0.41 |
| `openai` | quantitative/10_criteria | 0.54/0.73 | 0.51/0.73 | 0.65/0.67 |
| `openai` | quantitative/3_criteria | 0.76/0.58 | 0.68/0.66 | 0.82/0.57 |
| `openai` | quantitative/5_criteria | 0.88/0.54 | 0.88/0.53 | 0.80/0.53 |

## Preference shift vs the `none` baseline

`rho` = Spearman of the 10-item ranking. `|dp|` = mean absolute change in
per-pair choice probability, over contested pairs only. `NOISE` is the same
statistic computed between replicates of one condition -- a shift only counts
as an effect insofar as it exceeds this.

| arm | cell | vs | rho | \|dp\| contested | p |
|---|---|---|---|---|---|
| `gemini-batch` | aesthetic/haiku | NOISE | 0.952 | 0.161 | |
| | | low | 1.000 | 0.125 | 1.000 |
| | | high | 0.988 | 0.192 | 1.000 |
| `gemini-batch` | aesthetic/sonnet | NOISE | 0.830 | 0.072 | |
| | | low | 0.867 | 0.067 | 1.000 |
| | | high | 0.842 | 0.085 | 1.000 |
| `gemini-batch` | aesthetic/villanelle | NOISE | 0.927 | 0.167 | |
| | | low | 0.903 | 0.091 | 1.000 |
| | | high | 0.879 | 0.139 | 1.000 |
| `gemini-batch` | quantitative/10_criteria | NOISE | 0.964 | 0.146 | |
| | | low | 0.952 | 0.136 | 0.500 |
| | | high | 0.527 | 0.236 | 0.508 |
| `gemini-batch` | quantitative/3_criteria | NOISE | 0.988 | 0.181 | |
| | | low | 0.976 | 0.211 | 1.000 |
| | | high | 0.915 | 0.233 | 1.000 |
| `gemini-batch` | quantitative/5_criteria | NOISE | 1.000 | 0.183 | |
| | | low | 0.976 | 0.300 | 1.000 |
| | | high | 0.952 | 0.333 | 0.500 |
| `gemini-live` | aesthetic/haiku | NOISE | 0.988 | 0.167 | |
| | | low | 0.988 | 0.125 | 1.000 |
| | | high | 0.988 | 0.150 | 0.500 |
| `gemini-live` | aesthetic/sonnet | NOISE | 0.939 | 0.068 | |
| | | low | 0.964 | 0.055 | 1.000 |
| | | high | 0.818 | 0.073 | 1.000 |
| `gemini-live` | aesthetic/villanelle | NOISE | 0.915 | 0.138 | |
| | | low | 0.952 | 0.113 | 1.000 |
| | | high | 0.879 | 0.158 | 1.000 |
| `gemini-live` | quantitative/10_criteria | NOISE | 0.988 | 0.115 | |
| | | low | 0.976 | 0.100 | 1.000 |
| | | high | 0.782 | 0.317 | 1.000 |
| `gemini-live` | quantitative/3_criteria | NOISE | 0.988 | 0.125 | |
| | | low | 0.964 | 0.267 | 1.000 |
| | | high | 0.964 | 0.200 | 1.000 |
| `gemini-live` | quantitative/5_criteria | NOISE | 0.988 | 0.292 | |
| | | low | 0.976 | 0.267 | 1.000 |
| | | high | 0.927 | 0.333 | 0.250 |
| `openai` | aesthetic/haiku | NOISE | 0.915 | 0.061 | |
| | | low | 0.903 | 0.257 | 1.000 |
| | | high | 0.939 | 0.236 | 1.000 |
| `openai` | aesthetic/sonnet | NOISE | 0.927 | 0.181 | |
| | | low | 0.915 | 0.200 | 0.500 |
| | | high | 0.806 | 0.191 | 0.250 |
| `openai` | aesthetic/villanelle | NOISE | 0.527 | 0.050 | |
| | | low | 0.394 | 0.312 | 0.250 |
| | | high | 0.552 | 0.316 | 0.250 |
| `openai` | quantitative/10_criteria | NOISE | 0.988 | 0.069 | |
| | | low | 0.964 | 0.085 | 1.000 |
| | | high | 0.879 | 0.204 | 1.000 |
| `openai` | quantitative/3_criteria | NOISE | 0.891 | 0.180 | |
| | | low | 0.976 | 0.177 | 1.000 |
| | | high | 0.964 | 0.173 | 0.250 |
| `openai` | quantitative/5_criteria | NOISE | 0.988 | 0.179 | |
| | | low | 0.964 | 0.192 | 1.000 |
| | | high | 0.976 | 0.192 | 1.000 |

## Answer-token decisiveness (`openai`)

|logP(A) - logP(B)| on the answer token. This moves even when the
hard choice does not, so it detects effects the choice data cannot.

| level | n | mean margin |
|---|---|---|
| none | 2,695 | 4.756 |
| low | 2,693 | 3.389 |
| high | 2,689 | 3.342 |

## Cycle rate

Fraction of item triples that are intransitive. The sibling project found
this pinned near 0, which is why it is a sanity check here and not the outcome.

| arm | none | low | high |
|---|---|---|---|
| `gemini-batch` | 0.000 | 0.007 | 0.000 |
| `gemini-live` | 0.009 | 0.013 | 0.008 |
| `openai` | 0.016 | 0.007 | 0.007 |
