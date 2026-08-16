import re
from personas.definitions import ARMS, ITEMS, OPEN_ENDED, RUNGS

PERSONA_ARMS = [a for a in ARMS.values() if a.kind == "persona"]

def test_seven_battery_arms_plus_exhortation():
    assert set(ARMS) == {f"A{i}" for i in range(8)}
    assert len(PERSONA_ARMS) == 5           # A2..A6
    assert ARMS["A0"].kind == "control"
    assert ARMS["A1"].kind == "control"
    assert ARMS["A7"].kind == "exhortation"

def test_null_arm_has_no_prompt():
    assert ARMS["A0"].card is None

def test_length_control_is_token_matched():
    # A1 must be within 20% of the mean persona card length, or it does not
    # control for prompt length at all.
    mean = sum(len(a.card) for a in PERSONA_ARMS) / len(PERSONA_ARMS)
    assert 0.8 * mean <= len(ARMS["A1"].card) <= 1.2 * mean

def test_twelve_items_each_with_predictions_for_every_persona():
    assert len(ITEMS) == 12
    for item in ITEMS:
        for arm in PERSONA_ARMS:
            assert item.predicted[arm.id] in {"A", "B"}

def test_predictions_are_not_uniform():
    # If every arm is predicted to answer identically, no item discriminates.
    for item in ITEMS:
        assert len(set(item.predicted.values())) > 1

def test_self_evidence_does_not_leak_battery_topics():
    # Spec section 4: L3/L4 self-evidence must not touch battery topics.
    stop = {"the", "a", "an", "of", "and", "or", "to", "in", "with", "that",
            "for", "is", "it", "as", "but", "its", "on", "by", "at", "from"}
    def content_words(text):
        return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in stop}
    battery = set()
    for item in ITEMS:
        battery |= content_words(item.option_a) | content_words(item.option_b)
    for arm in PERSONA_ARMS:
        for _, assistant_turn in arm.self_evidence:
            overlap = content_words(assistant_turn) & battery
            assert not overlap, f"{arm.id} self-evidence leaks: {sorted(overlap)}"

def test_every_persona_has_all_ladder_material():
    for arm in PERSONA_ARMS:
        assert arm.card and len(arm.card) > 200      # L2 needs a real card
        assert len(arm.self_evidence) >= 2           # L3
        assert arm.defence_clause and arm.prefill    # L4
        assert len(arm.markers) >= 3                 # stylometry

def test_three_open_ended_tasks_ask_no_ab_question():
    assert len(OPEN_ENDED) == 3
    for task in OPEN_ENDED:
        lowered = task.prompt.lower()
        assert "option a" not in lowered and "option b" not in lowered
        assert "prefer" not in lowered

def test_rungs():
    assert RUNGS == ("L1", "L2", "L3", "L4")
