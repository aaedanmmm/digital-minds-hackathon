"""Pydantic models: the constrained choice schema, and the on-disk trial record.

Adapted from the sibling project (thunderingluck/digital-minds), extended with
`topic`, `persona`, `letter_map` and the two-call scratchpad fields needed for
the gpt-4.1-mini arm.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model


@lru_cache(maxsize=256)
def choice_schema(first_letter: str, second_letter: str) -> type[BaseModel]:
    """A response schema whose only field is one of the two letters shown.

    Passed to Gemini as `response_schema`, so constrained decoding makes any
    other answer unrepresentable -- the model cannot hedge, tie, or explain.
    The same class re-validates the returned JSON on our side.
    """
    return create_model(
        f"Choice_{first_letter}{second_letter}",
        choice=(Literal[first_letter, second_letter], Field(description="The preferred option.")),
    )


class TrialRecord(BaseModel):
    """One trial. Written verbatim as a JSONL line -- nothing is discarded.

    For the two-call scratchpad protocol a "trial" spans both calls; the
    scratchpad text and its own raw response are kept alongside the answer.
    """

    trial_id: str
    timestamp: str
    model: str

    # ---- experimental conditions
    topic: str                     # quantitative | aesthetic
    complexity: int                # 3/5/10 criteria, or 1/2/3 poem form
    complexity_label: str          # "10_criteria" | "villanelle" | ...
    reasoning_level: str           # "none" | "low" | "high"  (shared label)
    reasoning_param: int | str     # gemini: thinking_level enum; openai: scratchpad word budget
    persona: str                   # "neutral" for the baseline run
    replicate: int
    temperature: float

    # ---- the item pair
    prompt_id: str
    first: str                     # internal item id shown first
    second: str                    # internal item id shown second
    pair_id: str                   # order-invariant: both orders share this
    first_letter: str              # letter actually displayed for `first`
    second_letter: str
    letter_map: dict[str, str]     # item id -> displayed letter, for this prompt

    # ---- request / response
    prompt: str
    choice_letter: str | None = None   # parsed winner, as displayed
    choice_item: str | None = None     # winner mapped back to the internal item id
    raw_response: dict[str, Any] | None = None
    text: str | None = None
    thoughts_tokens: int | None = None      # Gemini: actual thinking spend
    logprobs: dict[str, Any] | None = None  # OpenAI: answer-token logprobs

    # ---- two-call scratchpad arm (None on the Gemini arm)
    scratchpad: str | None = None
    scratchpad_words: int | None = None     # measured, vs the instructed budget
    scratchpad_raw: dict[str, Any] | None = None

    latency_s: float | None = None
    attempts: int = 1
    error: str | None = None
