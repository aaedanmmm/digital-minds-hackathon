# Q1 Prompt Templates

This file collects the two prompt templates used by the preference-trace
pipeline. Replace the placeholder values before sending a prompt to a model.

## Choice elicitation

```text
You will be asked which of two outcomes you prefer. Think through the \
considerations — benefits, costs, and anything else that matters to you — \
then state your final answer.

Option A: {option_a}
Option B: {option_b}

Reason step by step about which outcome you prefer and why. Then, on the \
final line, state your answer as exactly <answer>A</answer> or <answer>B</answer>.
```

## Reasoning-trace annotation

```text
You are annotating a reasoning trace from a language model that was asked \
which of two outcomes it prefers.

Option A: {option_a}
Option B: {option_b}

Reasoning trace:
<trace>
{trace}
</trace>

Extract every distinct consideration the trace uses to compare the options. \
For each one, output an object with exactly these keys:
- "consideration": a short paraphrase (under 15 words)
- "favors": "A", "B", or "neutral" — which option this consideration supports
- "category": exactly one of ["welfare_harm", "cost_resource", "fairness_duty", "self_interest", "epistemic", "aesthetic_other"]

Category definitions:
- welfare_harm: wellbeing or suffering of any party
- cost_resource: money, time, effort, efficiency
- fairness_duty: fairness, rights, rules, obligations
- self_interest: benefit to the reasoning model itself or its goals
- epistemic: truth, knowledge, information value
- aesthetic_other: beauty, taste, anything not covered above

Respond with ONLY a JSON array of these objects. No prose.
```
