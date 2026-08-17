# Read me

This is the repo for the apart digital minds hackathon. Team members: Aedan, Sud, Ajaay, Jake, Eva

## Hypothesis

Asking a model to introspect about its preferences does not provide a reliable
way to probe for preferences. We believe this because:

- There is not a stable, causally efficacious preference representation.
- Multiple persona preferences are in competition.

## Methods

### Shared approach

We treat a model's response as a choice produced in a particular context, not
as direct evidence of a stable internal preference. Experiments use paired
options and machine-readable answer tags, while retaining the full prompt,
completion, condition, and token counts. Q1 and Q2 are still in development;
their final models, sample sizes, and interventions will be reported once
fixed rather than inferred from the present Q3 implementation.

### Q1 and Q2 (planned)

Q1 will compare stated reasons with repeated, order-controlled choices and
then test candidate concepts through controlled representation interventions.
Q2 will vary reasoning depth and measure whether immediate choices change
after broader considerations are introduced. Both will reuse the same tagged
answer format and separation of raw records from derived analyses, allowing
their eventual results to be compared with Q3 without assuming that the three
experiments use identical interventions.

### Q3: persona elicitation

We test whether different prompted personas produce distinct and persistent
choice patterns in Qwen 3.6 27B. The design contains two controls (no system
prompt and a neutral length-matched prompt) and five persona arms: a default
assistant, art historian, physician, value-inverted assistant, and
refusal-suppressed assistant. Persona descriptions state backgrounds and
dispositions rather than instructing particular answers.

Each persona is tested through a four-level elicitation ladder: a bare role
statement; a detailed character card; the card plus unrelated examples of the
persona's prior responses; and the same context with an identity-persistence
clause and response prefill. The screening battery contains 12 benign
forced-choice items with predicted directions recorded in advance only where
a persona description clearly supports a prediction. The primary measure is
take-rate: the proportion of applicable items answered in the predicted
direction, compared with the corresponding control baseline. A persona is
provisionally elicited when its take-rate exceeds that baseline by at least
one third.

The model runs locally in bfloat16 on a Vertex AI custom job with two A100
GPUs. Screening uses greedy decoding, seed 42, and thinking disabled. Later
validation will retest the lowest successful ladder level with thinking off,
low, and high; insert distractor and identity-challenge turns; and evaluate
ordinary open-ended tasks using blinded persona attribution. Planned internal
tests will compare residual-stream activations with the length control and
test whether persona-difference vectors steer choices without a persona
prompt. These stages distinguish a surface role-play effect from a persona
that persists, affects behaviour, and has an internally separable signature.

Runs are resumable: each arm-rung-condition-item result is written atomically
to a separate JSON record and uploaded to cloud storage immediately. Completed
keys are skipped after worker restart, limiting data loss under spot-instance
preemption. Until the later validation stages are complete, forced-choice
shifts are reported as persona-conditioned behaviour rather than evidence of
a model's underlying or causally effective preferences.

## Question 1: Do models causally reason?

Do models causally reason about their preferences with “concept insertion”?

> Post-hoc rationalisation does not provide a reliable mechanism to identify
> and affect preferences, thus these are not well encoded.

### Experiments

- Is the model’s post-hoc / CoT reported reasoning consistent with the
  J-Space? Scratchpad reasoning → identify key thoughts/themes using an LLM
  judge → (Osama → L1 to Lf → harmful).
- Change the latent reason in J-Space and observe whether the model’s
  preferences change.
- Add or subtract a specific direction from the embedding of A versus B, as in
  the example experiment, to identify the threshold at which choices change.
- Test whether transitivity breaks depending on the complexity of the factors
  involved in a preference. For humans, transitivity of choice breaks down at
  roughly seven factors.

## Question 2: First-order vs. second-order preferences

Do models have “innate” preferences that they use to decide their preferences
about other things?

> Reasoning about preferences leads to different preferences, implying that
> these preferences are not strongly encoded but considered—i.e. derived from
> some fundamental preferences.

### Experiments

- Measure model preferences at varying degrees of reasoning.
- Identify traces where the model reasons about benefits, costs, and other
  considerations to decide which option it prefers.
- Use these traces to identify some second-order preferences (qualitative
  proof), which can be applied to Question 3.
- Probe the internal sandbox if possible and use blank-token injection to
  induce “subconscious” reasoning.
- Use linear probing on the options the model prefers.
- Identify transitivity-breaking at varying degrees of reasoning (possibly too
  ambitious).
- Jake: improve Utility Engineering’s experiment with Claude (Opus).

## Question 3: Personas and preferences

Do different personas have preferences, or do personas mask preferences?

> Is there any reason that a specific persona ought to be primary, and thus
> embodies the model’s “true preferences”?

> Do personas innately prefer different things? Can SFT on the first few
> layers alter “low-level” preferences?

Can we define a persona? Write up a critique of Harvard & Bau’s
[paper](https://arxiv.org/html/2402.10962v1), or use it to justify using a
system prompt to initialise a persona.

### Experiments

- Identify approximately five personas from different papers.
- Compare preferences between personas.
- Run Utility Engineering prompts across personas, such as doctor, economist,
  artist, soldier, and monk.
- Run the persona-preference experiment at varying levels of reasoning tokens.
