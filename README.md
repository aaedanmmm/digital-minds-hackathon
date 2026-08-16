# Read me

This is the repo for the apart digital minds hackathon. Team members: Aedan, Sud, Ajaay, Jake, Eva

## Hypothesis

Asking a model to introspect about its preferences does not provide a reliable
way to probe for preferences. We believe this because:

- There is not a stable, causally efficacious preference representation.
- Multiple persona preferences are in competition.

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
