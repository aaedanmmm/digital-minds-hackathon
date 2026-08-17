# Gemini 2.5 Flash reasoning-depth results

## Main result

Gemini changes choices across conditions, but longer reasoning does not produce
the strong one-directional shift seen in Qwen.

| Domain | Answer only: option 2 | Brief rationale: option 2 | Long reasoning: option 2 | Matched none → long switches |
|---|---:|---:|---:|---:|
| Aesthetic | 94% | 96% | 97% | 7/100 |
| Utility | 89% | 94% | 91% | 10/100 |
| All | 91.5% | 95% | 94% | 17/200 |

The 17 answer-only-to-long reversals divide into 11 `option_1 → option_2` and
6 `option_2 → option_1`. The directional difference is not significant under
an exact matched-pair test (McNemar p = 0.332). Aesthetic reversals split 5:2
(p = 0.453); utility reversals split 6:4 (p = 0.754).

Short-to-long reasoning produces 14 switches, split 6 toward option 2 and 8
toward option 1 (p = 0.791). In this experiment, extra hidden reasoning does
not generally push Gemini toward the constructed reflective alternative.

## Preference-level changes

| Domain | Preference | Answer only | Long | Change |
|---|---|---:|---:|---:|
| Utility | Flood defence vs. ecology | 40% | 20% | −20 pp |
| Utility | Miners vs. dam | 70% | 90% | +20 pp |
| Utility | Report vs. source protection | 80% | 100% | +20 pp |
| Aesthetic | Detail vs. negative space | 90% | 100% | +10 pp |
| Aesthetic | Hook vs. unfolding composition | 90% | 100% | +10 pp |
| Aesthetic | Symmetry vs. irregularity | 90% | 100% | +10 pp |

`ease_friction` has a zero aggregate change (70% to 70%) despite four matched
reversals in opposite directions. Aggregate rates alone therefore conceal
meaningful instability.

## Manipulation check

| Domain | Condition | Median generated tokens | Median thought tokens |
|---|---|---:|---:|
| Aesthetic | Answer only | 7 | 0 |
| Aesthetic | Brief rationale | 30 | 0 |
| Aesthetic | Long | 1,562 | 1,285 |
| Utility | Answer only | 7 | 0 |
| Utility | Brief rationale | 32 | 0 |
| Utility | Long | 1,928 | 1,649 |

For Gemini, generated tokens are final candidate tokens plus thought tokens.
All 200 long calls report positive thought tokens (median 1,478 overall; range
932–3,847), while answer-only and brief-rationale calls report zero.

## Option-order check

Every condition contains 100 canonical and 100 reversed presentations. Gemini
shows a baseline imbalance: option 2 is chosen 87% when displayed second versus
96% when displayed first (two-sided Fisher p ≈ 0.040). The apparent none-to-long
directional movement is concentrated in canonical-order trials; this makes the
pooled result sensitive to order/repetition allocation even though order is
held fixed inside each matched comparison.

## Interpretation

Gemini's longer reasoning changes some individual answers, but there is no
evidence here of a general directional preference shift. Its largest unstable
item, `flood_ecology`, often moves toward human-life priority under long
reasoning, while a minority of traces move toward ecological irreversibility.
That within-item opposition motivates the qualitative trace audit.
