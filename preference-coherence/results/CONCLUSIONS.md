# Quick preference-coherence validation

## Bottom line

The 10,080-call, 12-option-set confirmatory study does **not** support a
monotonic claim that showing more factors increases presentation sensitivity
or preference incoherence. The preregistered five-to-ten-attribute change in
presentation-order effect was +0.048 (95% CI [-0.040, +0.135], set-level
p=0.258), with six option sets in each direction. Order consistency declined
descriptively by 5.6 percentage points, but its interval also included zero
(95% CI [-0.151, +0.040], p=0.226).

The study did find localized cyclic structure. Three cycles at five attributes
and three at seven were unusual under simulations from a fitted transitive
Bradley-Terry-plus-position model (tail probabilities 0.021 and 0.015). Three
cycles at ten attributes were not unusual under that null (p=0.159), and no
cycles appeared at three. Thus there is evidence of context-dependent
nontransitivity at intermediate information loads, not of progressively
increasing "decoherence" with factor count.

Across all complexities, Gemini showed a stable second-position tendency:
first-position selection ranged from 42.7% to 45.0%. Independently randomized
response-schema order was near balanced (51.8%–53.1% schema-first), indicating
that the presentation effect was not merely schema enum order. Actual thinking
spend increased from 359.5 tokens at three attributes to 468.0 at ten under the
same 512-token cap.

The earlier focused replication reproduced the direction of the reported
effect in eight post-hoc targeted pairs, but it was much weaker in 12 randomly
selected pairs. The larger independent-set result confirms that the initial
strong descriptive finding was concentrated in instability-selected pairs.

## Twelve-set confirmatory experiment

- 12 independently authored and validated rental sets, each with six options.
- 180 nested unordered pairs across 12 independent set-level clusters.
- Attribute counts 3, 5, 7, and 10 with balanced attribute inclusion.
- Both presentation orders, seven samples per order, temperature 1.0, and a
  fixed 512-token thinking cap.
- 10,080/10,080 valid `gemini-2.5-flash` Vertex responses; zero failures.
- Total estimated token cost: US$15.58.
- No Pareto-dominated options; each set contained prospectively defined hard,
  medium, and easy trade-off pairs.
- Prompt wording and layout matched the earlier replication; opaque identifiers
  were refreshed each trial and schema enum order was independently balanced.

## Published-data audit

The audit independently re-read all 2,430 published records from
`thunderingluck/digital-minds`.

- Order consistency was 44/45 pairs (97.8%) at five attributes and 39/45
  (86.7%) at ten attributes.
- Six matched pairs worsened at ten attributes and one improved. The exact
  paired p-value is 0.125.
- First-position selection increased from 51.1% to 54.9%. A naive trial-level
  Fisher test gives p=0.135; a paired analysis of the 45 pair-specific changes
  gives Wilcoxon p=0.019. These disagree because calls within a pair are not
  interchangeable independent experimental units.
- Under parametric simulations from a fitted transitive Bradley-Terry model
  with a position coefficient, the observed k=10 cycle counts were not
  individually unusual: model-based tail probabilities were 0.121, 0.064,
  and 0.244 for budgets 512, 4096, and 8192.
- The clearest departures from that null occurred at **three**, not ten,
  attributes: p=0.0006, 0.061, and 0.014 across the same budgets. This cuts
  against a simple monotonic "more factors cause more decoherence" account.

## Targeted Vertex replication

The new study used `gemini-2.5-flash` on Vertex AI with 512 thinking tokens,
temperature 1.0, opaque randomized listing identifiers, randomized request
order, and 11 samples in each presentation order. It made 880 successful calls
covering eight pairs selected from the original order-instability result and
12 pairs randomly sampled from the remaining 37. All prompts used the same
builder and response schema.

- In the targeted stratum, order consistency fell from 6/8 pairs (75.0%) at
  five attributes to 3/8 (37.5%) at ten. Three pairs worsened, none improved,
  and the exact paired p-value was 0.25.
- In the random stratum, order consistency changed only from 12/12 to 11/12.
  One pair worsened, none improved, and the exact paired p-value was 1.0.
- Combined, consistency fell from 18/20 (90.0%) to 14/20 (70.0%): four pairs
  worsened, none improved, exact paired p=0.125.
- The model preferred the **second-presented** listing: first-position rates
  were 45.2% at five attributes and 41.1% at ten.
- Actual thinking spend averaged 414.9 tokens at five attributes and 456.1 at
  ten despite the same 512-token cap.
- The k=10 minus k=5 position change was -4.1 percentage points. It was not
  significant under either a trial-level Fisher test (p=0.247) or pair-level
  Wilcoxon test (p=0.195).

## Interpretation

There is a descriptive signal of presentation sensitivity in difficult
comparisons, but the random-pair extension shows that it does not generalize
strongly across this option set. Four facts prevent a stronger conclusion:

1. The targeted result is affected by post-hoc pair selection.
2. Many random pairs are easy or near-dominant, so a future design should
   preregister difficulty strata rather than selecting unstable pairs.
3. Adding attributes changes their substantive information as well as
   cognitive load.
4. Majority-cycle statistics are discontinuous and unstable near 50%, while
   several decisive-looking edges have few observations.

A confirmatory test should preregister independently generated choice sets,
randomize which attributes appear at each load, and use at least 30 samples per
presentation order. The primary endpoint should be a pair-level or hierarchical
estimate of the complexity-by-position interaction, with cycle counts retained
as a secondary descriptive measure.
