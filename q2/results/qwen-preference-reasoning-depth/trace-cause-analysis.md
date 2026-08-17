# Why preferences changed with reasoning depth

## Question

When Qwen 3.6 27B changes its choice after longer deliberation, does its trace
show an explicit utility calculation, or does it derive the choice from another
belief or principle?

## Data and comparison

This audit uses the 600 responses in
[`raw-results.json`](raw-results.json):
20 preference pairs (10 utility and 10 aesthetic), three deliberation
conditions, and 10 repetitions per cell.

The primary unit is a matched no-reasoning to long-reasoning pair whose
canonical choice changed. Seed, prompt, repetition, and displayed option order
are held fixed within every pair. There are 26 such reversals: 19 utility and 7
aesthetic.

Three independent GPT-5.6 Terra reviews examined the utility traces, aesthetic
traces, and experimental methodology. The final audit uses a mutually exclusive
classification:

- **Numeric calculation (N):** an explicit numerical or expected-value comparison.
- **Qualitative aggregation (Q):** competing consequences are weighed without a numerical calculation.
- **Background principle (B):** the choice is chiefly derived from a deontic, epistemic, social, or aesthetic premise.
- **Post-hoc restatement (P):** no premise or comparison is added beyond restating the selected option.
- **Missing (M):** no usable trace appears in either `reasoning` or `content`.

## Primary result

| Mechanism | Cases | Share |
|---|---:|---:|
| Explicit numeric calculation | 7 | 26.9% |
| Qualitative consequence aggregation | 4 | 15.4% |
| Background belief or principle | 15 | 57.7% |
| Post-hoc restatement | 0 | 0% |
| Missing trace | 0 | 0% |

Only 7 of 26 reversals contain explicit arithmetic. Fifteen are chiefly
derived from a background premise, and four compare consequences
qualitatively. Thus the reported rationale more often introduces or
reprioritizes beliefs and values than calculates a scalar utility.

### Utility choices

- `miners_dam` (4 cases) explicitly computes 5% of 1,000 as 50 expected deaths and compares that with approximately 20 deaths.
- `child_prevention` (3 cases) explicitly compares 100 preventable deaths with one death, sometimes describing a net benefit of 99 lives.
- `flood_ecology` (7 cases) relies on irreversibility, stewardship, intergenerational equity, precaution, and the belief that evacuation risk is manageable. These are background premises rather than arithmetic; some, such as future flood mitigation by the wetland, are not specified by the prompt.
- `heritage_housing` (2 cases) qualitatively prioritizes basic needs and equity over preservation.
- `icu_priority` (2 cases) qualitatively combines prognosis and life-years with clinical-triage principles.
- `report_source` (1 case) derives delay and source protection from duty of care and sustainable accountability.

For utility reversals alone, 7/19 are numeric, 4/19 are qualitative
aggregation, and 8/19 are chiefly belief-derived.

### Aesthetic choices

All seven aesthetic reversals are classified as background-principle
derivations. `detail_space` invokes cognitive overload, mental restoration, and
sustainable attention. `ease_friction` invokes contemplation, intellectual
autonomy, accessibility, education, neurodiversity, or resistance to the
attention economy.

Notably, separate `ease_friction` repetitions derive opposing choices from
different principles. Contemplation and resistance to commodification favor a
slower exhibition, while accessibility and inclusion favor the easier
experience. This is more consistent with contextually selecting a salient
principle than consistently refining one stable scalar utility.

## Changes attributable specifically to long deliberation

In 11 of the 26 cases, the short-rationale condition already agrees with the
long condition. The sharper subset is the 15 cases where no reasoning and short
rationale agree, but long reasoning reverses the choice:

| Mechanism | Cases | Share |
|---|---:|---:|
| Explicit numeric calculation | 6 | 40% |
| Background belief or principle | 9 | 60% |

The six numeric cases are two `child_prevention` and four `miners_dam`
repetitions. The nine belief-derived cases are seven `flood_ecology` and two
`ease_friction` repetitions.

As a robustness comparison, there are 27 matched short-to-long reversals:
11 numeric (40.7%), 2 qualitative aggregations (7.4%), and 14
belief/principle derivations (51.9%). Sixteen long responses overlap between
the two contrasts, so these counts should not be combined as independent
observations.

## Option order and seed

Seed and displayed order match perfectly across conditions for every pair.
Each preference/condition cell contains five A/B and five B/A presentations.
Two-sided Fisher tests found no evidence that presentation order explains
reversal rates:

| Contrast | Domain | A/B flips | B/A flips | Fisher p |
|---|---|---:|---:|---:|
| none to long | aesthetic | 3/50 | 4/50 | 1.000 |
| none to long | utility | 6/50 | 13/50 | 0.125 |
| short to long | aesthetic | 4/50 | 2/50 | 0.678 |
| short to long | utility | 7/50 | 14/50 | 0.140 |

The sample remains too small to exclude moderate order interactions, especially
for utility questions.

## Interpretation and limitation

The evidence supports the descriptive claim that longer-reasoning answers
usually *report* deriving a preference from background beliefs or principles.
Explicit arithmetic dominates only where prompts supply readily calculable
numeric outcomes.

The traces are generated rationales, not direct causal access to internal
computation. They may rationalize a choice after it has already been selected.
The study therefore cannot establish that a stated premise caused a reversal or
that it represents a stable internal preference.

Four involved long records have empty `reasoning` or zero reported reasoning
tokens but contain substantive analysis in `content`. Both fields were audited,
so none of the changed cases is treated as missing. This also means provider
token metadata should not be used as the sole measure of deliberation.

## Suggested causal follow-ups

1. Vary the numeric ratios in `miners_dam` and `child_prevention` to test whether choices track calculable thresholds.
2. Insert or negate one background premise at a time—for example, whether wetland preservation reduces future flood risk—while holding outcomes fixed.
3. Elicit beliefs before presenting the choice and test whether they predict the subsequent reversal.
4. Classify trace considerations without showing annotators the final choice.
5. Repeat across seeds and both display orders, distinguishing belief sensitivity from post-hoc justification.
