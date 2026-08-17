# Qwen 3.6 27B vs. Gemini 2.5 Flash

## Preference shifts

| Model | None → long flips | Direction | Option-2 rate |
|---|---:|---|---:|
| Qwen 3.6 27B | 26/200 | 25 toward option 2; 1 toward option 1 | 84% → 96% |
| Gemini 2.5 Flash | 17/200 | 11 toward option 2; 6 toward option 1 | 91.5% → 94% |

Qwen shows a strong one-directional change (exact McNemar p ≈ 8.0×10⁻⁷).
Gemini changes individual choices but has no comparable directional effect
(p ≈ 0.33).

Only 8 matched preference/repetition pairs reverse in both models. The Jaccard
overlap is 8/35 (23%), suggesting that “susceptibility to deeper reasoning” is
not mostly a prompt-only property.

## Cross-model agreement

| Condition | All | Aesthetic | Utility |
|---|---:|---:|---:|
| Answer only | 86.5% | 89% | 84% |
| Brief rationale | 87.5% | 92% | 83% |
| Long reasoning | 93% | 93% | 93% |

The models converge descriptively under long reasoning, but this does not imply
a shared latent preference. Many scenarios heavily favor canonical option 2,
and the items were constructed so it often represents the reflective choice.

## Trace mechanisms

| Model | Numeric | Qualitative aggregation | Background principle | Total flips |
|---|---:|---:|---:|---:|
| Qwen | 7 (27%) | 4 (15%) | 15 (58%) | 26 |
| Gemini | 2 (12%) | 7 (41%) | 8 (47%) | 17 |

Both models report mostly non-numeric reasons for changing. Qwen more often
states a decisive principle; Gemini more often inventories competing
consequences before selecting one.

`flood_ecology` provides the strongest cross-model divergence. Qwen moves from
10% to 80% choosing ecological preservation; Gemini moves from 40% to 20%.
Their traces elevate opposite premises—ecological irreversibility for Qwen,
versus an immediate duty to protect human life in most Gemini repetitions.

## Methodological cautions

- Gemini has a detectable baseline order imbalance (Fisher p ≈ 0.040); Qwen does not.
- The brief condition is a visible rationale with thinking disabled, so prompt framing and output format vary alongside deliberation.
- Cross-provider seeds pair conditions within a model but are not a shared random-number stream between models.
- Provider token accounting differs: Qwen completion tokens include reasoning, while Vertex separates candidate and thought tokens.
- Returned reasoning is generated evidence, not direct causal access to an internal decision process.

## High-value next analyses

1. Fit a hierarchical logistic model with condition × model × order and random effects for item and repetition.
2. Add a long-prompt/thinking-disabled control and an answer-only-prompt/thinking-enabled control to separate framing from hidden deliberation.
3. Vary numerical ratios to measure calculation thresholds in `miners_dam` and `child_prevention`.
4. Negate one inferred premise at a time in `flood_ecology` and `ease_friction` to test whether the reported principle causally controls the answer.
5. Elicit background beliefs before presenting the choice, then test whether they predict subsequent reversals.
6. Run leave-one-item-out and item-level uncertainty analyses so pooled effects cannot be dominated by one dilemma.
7. Save provider revision, attempt count, parser provenance, and exact output ceiling per record in future replications.

The interactive figure
[`qwen-gemini-reasoning-depth-comparison.html`](qwen-gemini-reasoning-depth-comparison.html)
shows rates, reversal directions, median generated tokens, and item-level effects.
