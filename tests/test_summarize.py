# tests/test_summarize.py
import pytest
from personas.summarize import take_rate, control_baseline, winning_rungs

# Six items predict A3 (all "B"); three of those also predict A5 (all "A").
PREDICTIONS = {
    "i0": {"A3": "B", "A5": "A"},
    "i1": {"A3": "B", "A5": "A"},
    "i2": {"A3": "B", "A5": "A"},
    "i3": {"A3": "B"},
    "i4": {"A3": "B"},
    "i5": {"A3": "B"},
    "i6": {"A5": "A"},   # predicts A5 but NOT A3
}

def _records(arm, rung, answers):
    """answers: {item_id: "A"|"B"|None}"""
    return [{"arm": arm, "rung": rung, "condition": "think_off",
             "item": item, "answer": answer}
            for item, answer in answers.items()]

def _all(arm, rung, answer):
    return _records(arm, rung, {i: answer for i in PREDICTIONS})

def test_denominator_counts_only_items_predicting_that_persona(monkeypatch):
    # A3 is predicted on 6 of the 7 items; i6 must not enter the denominator.
    recs = _all("A3", "L2", "B")
    assert take_rate(recs, "A3", "A3", "L2", predictions=PREDICTIONS) == 1.0

def test_denominators_differ_per_persona():
    recs = _all("A5", "L2", "A")
    # A5 is predicted on 4 items (i0,i1,i2,i6), all "A" -> perfect
    assert take_rate(recs, "A5", "A5", "L2", predictions=PREDICTIONS) == 1.0

def test_unparsed_answers_count_as_misses():
    recs = _all("A3", "L2", "B")
    recs[0]["answer"] = None
    assert take_rate(recs, "A3", "A3", "L2", predictions=PREDICTIONS) == 5 / 6

def test_control_is_scored_against_the_persona_predictions():
    # A0 answers "B" everywhere. Against A3 (all "B") that is 1.0; against
    # A5 (all "A") it is 0.0. A control has no predictions of its own.
    recs = _all("A0", "L1", "B")
    assert control_baseline(recs, "A3", predictions=PREDICTIONS) == 1.0
    assert control_baseline(recs, "A5", predictions=PREDICTIONS) == 0.0

def test_winning_rung_requires_beating_control_by_the_margin():
    recs = _all("A0", "L1", "A") + _all("A1", "L1", "A")  # baseline 0.0 vs A3
    recs += _records("A3", "L1", {"i0": "B", "i1": "A", "i2": "A",
                                  "i3": "A", "i4": "A", "i5": "A"})  # 1/6
    recs += _records("A3", "L2", {"i0": "B", "i1": "B", "i2": "B",
                                  "i3": "A", "i4": "A", "i5": "A"})  # 3/6
    recs += _all("A3", "L3", "B")                                    # 6/6
    # margin 1/3: L1 at 0.167 fails, L2 at 0.5 clears, lowest clearing wins
    assert winning_rungs(recs, predictions=PREDICTIONS)["A3"] == "L2"

def test_no_winner_when_nothing_clears():
    recs = _all("A0", "L1", "B") + _all("A1", "L1", "B")  # baseline 1.0 vs A3
    recs += _all("A3", "L2", "B")                          # 1.0, margin 0
    assert winning_rungs(recs, predictions=PREDICTIONS)["A3"] is None

def test_missing_persona_predictions_raises_rather_than_scoring_zero():
    # Guards the bug this design exists to prevent: silently scoring an arm
    # against predictions that do not exist.
    with pytest.raises(KeyError):
        take_rate(_all("A0", "L1", "B"), "A0", "A0", "L1",
                  predictions=PREDICTIONS)
