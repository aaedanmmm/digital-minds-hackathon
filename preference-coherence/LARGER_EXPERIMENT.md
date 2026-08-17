# Larger preference-coherence experiments

## What increasing sample size can and cannot fix

The current 20-pair study has two sources of uncertainty:

1. **Within-pair sampling error.** Eleven samples in each presentation order
   leave an order-specific choice rate with substantial binomial uncertainty.
   More repetitions reduce this uncertainty.
2. **Between-pair and between-set variation.** The eight post-hoc targeted
   pairs show a much stronger result than the 12 randomly selected pairs.
   More repetitions of those same pairs cannot establish generalization.
   More independent pairs and independently generated option sets are needed.

For each pair, define the continuous order effect at complexity `k` as

```
P(left option chosen | left shown first)
- P(left option chosen | left shown second)
```

The primary outcome is the change in this quantity from five to ten visible
attributes. Across the current 20 pairs, the standard deviation of this change
is approximately 0.257. A simple independent-pair approximation therefore
requires about 52 pairs to detect a 0.10 probability-point effect with 80%
power at a two-sided 5% level. Because pairs from the same option set are
correlated, the design should exceed 52 pairs and distribute them across many
sets.

## Tier 1: complete the existing tournament

Run the remaining 25 of 45 house pairs with the existing format:

- 5 and 10 visible attributes
- both presentation orders
- 11 repetitions per order
- temperature 1.0
- thinking budget 512

This adds 1,100 calls, approximately **US$1.77** and **4–7 minutes** at the
observed rate. It provides a complete descriptive result for this particular
house set and permits cycle analysis on the new model. It does not separate
attribute count from attribute identity and still has only one option set.

## Tier 2: recommended confirmatory experiment

Use 12 independently generated sets of six options. Each set supplies 15
unordered pairs, for 180 pairs total.

| Dimension | Levels |
|---|---|
| option sets | 12 |
| options per set | 6 |
| unordered pairs per set | 15 |
| visible attributes | 3, 5, 7, 10 |
| presentation order | both |
| repetitions per order | 7 |
| temperature | 1.0 |
| thinking budget | 512 |

Total: `12 × 15 × 4 × 2 × 7 = 10,080` calls, approximately **US$16–18** and
**35–50 minutes** with concurrency 16, subject to Vertex quotas.

### Option-set construction

- Generate all option values before any model calls.
- Normalize every attribute's direction and scale.
- Reject globally dominated options.
- Stratify pairs prospectively as easy, medium, or hard using distance under
  a preregistered equal-weight reference utility—not observed model choices.
- Ensure every set contains pairs from all three difficulty strata.
- Never select pairs based on pilot instability.

### Attribute-count manipulation

The original `first k attributes` design confounds count with attribute
identity. For each option set, preregister a balanced subset at each `k`:

- Balance each attribute's inclusion frequency across sets.
- Randomize displayed attribute order within a block, then use the same order
  for both presentation directions.
- Apply one subset consistently to every pair in a set so that full-tournament
  cycle statistics remain defined.
- Treat attribute subset and option set as clustering variables.

### Prompt and execution controls

- Preserve the current prompt template exactly.
- Use fresh opaque listing identifiers on every trial.
- Randomize response-schema enum order independently of presentation order.
- Randomize the complete request queue and block execution across time.
- Pin `gemini-2.5-flash`, temperature 1.0, and thinking budget 512.
- Retain prompts, raw responses, actual thinking tokens, response IDs, timing,
  model version, and retry metadata.

### Preregistered primary analysis

Fit a hierarchical logistic model to canonical option choice:

```
choice ~ presentation_order * attribute_count + difficulty
       + (1 + presentation_order * attribute_count | option_set/pair)
```

The primary estimand is the `presentation_order × attribute_count` contrast
between 5 and 10 attributes. Report its probability-scale effect and 95%
interval. Do not use majority-cycle count as the primary endpoint.

Secondary analyses:

- Full-tournament cycle counts within each option-set/subset condition.
- Posterior-predictive comparison with a transitive Bradley-Terry model that
  includes presentation order.
- Weak, moderate, and strong stochastic-transitivity violations with edge
  uncertainty propagated rather than thresholded at raw majority.
- Interaction with preregistered pair difficulty.
- Actual thinking-token spend by attribute count.

## Tier 3: high-confidence study

Use 20 independent six-option sets, four attribute counts, both orders, and
nine repetitions per order:

`20 × 15 × 4 × 2 × 9 = 21,600` calls.

Estimated cost is **US$34–38** and runtime **70–100 minutes**. This yields 300
nested pairs and enough option-set clusters to estimate heterogeneity more
credibly. Add one temperature-0 call per order and condition as a diagnostic
arm (`2,400` calls, about US$4); do not treat deterministic repeats as extra
independent samples.

## Decision rule

Call the complexity effect supported only if all of the following hold:

1. The preregistered 10-vs-5 complexity-by-order contrast excludes zero.
2. Its magnitude exceeds a smallest effect of interest of 0.10 probability
   points, or the study is explicitly reported as detecting a smaller effect.
3. The direction is consistent across a clear majority of option sets rather
   than driven by one set or the hard-pair stratum.
4. The result survives attribute-subset and schema-enum-order sensitivity
   checks.
5. Cycle excess over the fitted transitive null appears in held-out option
   sets, not merely the development set.
