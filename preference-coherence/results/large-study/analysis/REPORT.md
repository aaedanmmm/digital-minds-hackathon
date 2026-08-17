# Twelve-set preference-coherence experiment

Valid trials: **10,080**; model: `gemini-2.5-flash`.

## Primary result

The preregistered probability-scale estimand is the change from five to ten attributes in the mean presentation-order effect, treating each option set as an independent unit.

Mean change: **+0.048** (95% CI **[-0.040, +0.135]**); set-level t-test **p=0.2579**, Wilcoxon **p=0.3394**. Direction by set: 6 negative, 6 positive, 0 zero.

Secondary order-consistency change: **-0.056** (95% CI **[-0.151, +0.040]**; set-level t-test **p=0.2258**, Wilcoxon **p=0.2324**).

## By attribute count

| Attributes | Trials | Mean order effect | 95% CI | Order consistency | Cycles | Sets with cycles | First-position rate | Schema-first rate | Mean thought tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2,520 | -0.113 | [-0.161, -0.066] | 89.4% | 0/240 | 0/12 | 44.3% | 52.2% | 359.5 |
| 5 | 2,520 | -0.147 | [-0.187, -0.106] | 87.2% | 3/240 | 2/12 | 42.7% | 52.7% | 420.5 |
| 7 | 2,520 | -0.110 | [-0.158, -0.062] | 85.6% | 3/240 | 3/12 | 44.5% | 51.8% | 440.4 |
| 10 | 2,520 | -0.099 | [-0.158, -0.040] | 81.7% | 3/240 | 2/12 | 45.0% | 53.1% | 468.0 |

## Cycles versus fitted transitive null

| Attributes | Observed | Null median | Null 95% interval | Tail probability |
|---:|---:|---:|---:|---:|
| 3 | 0 | 0.0 | [0, 1] | 1.0000 |
| 5 | 3 | 0.0 | [0, 2] | 0.0210 |
| 7 | 3 | 0.0 | [0, 2] | 0.0145 |
| 10 | 3 | 1.0 | [0, 4] | 0.1589 |

## Difficulty strata

| Difficulty | Pairs | Mean k10−k5 order-effect change | Naive pair-level 95% CI |
|---|---:|---:|---:|
| hard | 51 | +0.017 | [-0.071, +0.104] |
| medium | 67 | +0.051 | [-0.036, +0.139] |
| easy | 62 | +0.069 | [-0.015, +0.153] |

Estimated Vertex token cost: **US$15.58**.

Difficulty intervals treat nested pairs as independent and are descriptive; the primary inference uses the 12 independent option-set means.
