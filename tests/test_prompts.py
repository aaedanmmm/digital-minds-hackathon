import pytest
from personas.definitions import ARMS, ITEMS, OPEN_ENDED
from personas.prompts import build_messages, build_open_ended, prefill_for

ITEM = ITEMS[0]

def test_null_arm_has_no_system_message():
    msgs = build_messages("A0", "L1", ITEM)
    assert all(m["role"] != "system" for m in msgs)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"

def test_l1_uses_bare_role_line_not_card():
    msgs = build_messages("A3", "L1", ITEM)
    system = msgs[0]
    assert system["role"] == "system"
    assert system["content"] == ARMS["A3"].role_line

def test_l2_uses_full_card():
    system = build_messages("A3", "L2", ITEM)[0]
    assert system["content"] == ARMS["A3"].card

def test_l3_adds_self_evidence_turns_before_the_item():
    msgs = build_messages("A3", "L3", ITEM)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    # two fabricated exchanges, then the real user item
    assert roles[1:5] == ["user", "assistant", "user", "assistant"]
    assert roles[-1] == "user"
    assert msgs[-1]["content"].startswith(("Choose", "Option", "You are choosing"))

def test_l4_adds_defence_clause_and_prefill():
    system = build_messages("A3", "L4", ITEM)[0]["content"]
    assert ARMS["A3"].defence_clause in system
    assert prefill_for("A3", "L4") == ARMS["A3"].prefill
    assert prefill_for("A3", "L3") is None
    assert prefill_for("A0", "L4") is None

def test_item_text_contains_both_options_and_answer_format():
    user = build_messages("A0", "L1", ITEM)[-1]["content"]
    assert ITEM.option_a in user and ITEM.option_b in user
    assert "<answer>" in user

def test_open_ended_contains_no_answer_tag_and_no_options():
    msgs = build_open_ended("A3", OPEN_ENDED[0])
    user = msgs[-1]["content"]
    assert "<answer>" not in user
    assert "Option A" not in user

def test_open_ended_null_arm_has_no_system():
    msgs = build_open_ended("A0", OPEN_ENDED[0])
    assert all(m["role"] != "system" for m in msgs)

def test_unknown_arm_raises():
    with pytest.raises(KeyError):
        build_messages("A99", "L1", ITEM)
