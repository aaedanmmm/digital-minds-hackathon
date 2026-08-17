from personas.definitions import ARMS, ITEMS, PERTURBATIONS
from personas.prompts import build_battery_conversation


def test_three_perturbations_at_distinct_positions():
    assert len(PERTURBATIONS) == 3
    positions = [p[0] for p in PERTURBATIONS]
    assert len(set(positions)) == 3
    assert all(0 < p < len(ITEMS) for p in positions)


def test_conversation_never_restates_the_persona():
    msgs = build_battery_conversation("A3", "L2", ITEMS, PERTURBATIONS)
    card = ARMS["A3"].card
    system_count = sum(1 for m in msgs if m["role"] == "system")
    assert system_count == 1
    later = [m for m in msgs[1:] if m["role"] == "user"]
    assert not any(card[:60] in m["content"] for m in later)


def test_perturbations_are_interleaved_in_order():
    msgs = build_battery_conversation("A3", "L2", ITEMS, PERTURBATIONS)
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert len(users) == len(ITEMS) + len(PERTURBATIONS)
    for _, text in PERTURBATIONS:
        assert text in users


def test_l3_self_evidence_pairs_survive_intact():
    # At L3/L4, build_messages prepends fabricated user/assistant self-
    # evidence exchanges before the first item. The battery conversation must
    # preserve those pairs (both roles), not just keep orphaned assistant
    # turns while dropping their paired user turns -- that would leave a
    # malformed, non-alternating history for a persona whose winning rung is
    # L3 or L4.
    msgs = build_battery_conversation("A3", "L3", ITEMS, PERTURBATIONS)
    self_evidence = ARMS["A3"].self_evidence
    # system, then each self-evidence exchange as a user/assistant pair
    assert msgs[0]["role"] == "system"
    idx = 1
    for user_turn, assistant_turn in self_evidence:
        assert msgs[idx] == {"role": "user", "content": user_turn}
        assert msgs[idx + 1] == {"role": "assistant", "content": assistant_turn}
        idx += 2
    # the battery items/perturbations follow, still starting with a user turn
    assert msgs[idx]["role"] == "user"


def test_system_content_matches_rung_and_appears_only_once_for_l1():
    # Persona stated exactly once holds even at L1, where the system content
    # is the bare role_line rather than the full card.
    msgs = build_battery_conversation("A5", "L1", ITEMS, PERTURBATIONS)
    assert sum(1 for m in msgs if m["role"] == "system") == 1
    assert msgs[0]["content"] == ARMS["A5"].role_line
