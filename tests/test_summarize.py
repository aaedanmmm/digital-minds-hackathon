# tests/test_summarize.py
import pytest
from personas.summarize import (
    take_rate, control_baseline, winning_rungs, unparsed_stats, summarize,
)

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


# --- unparsed rate (Piece 2/3 follow-on: the L4 measurement defect) -------
#
# Stage A's L4 rung produced 44 unparsed answers out of 60 -- every unparsed
# answer in the whole study -- because the prefill sends the model into prose
# that hits the think_off token cap before an <answer> tag is ever emitted.
# The headline L4 take-rate (0.00) read as a null persona result until the
# parse rate was checked by hand. unparsed_stats/summarize exist so that
# check happens automatically instead of by hand.

def test_unparsed_stats_counts_none_answers_and_computes_rate():
    recs = _all("A3", "L4", "B")
    recs[0]["answer"] = None
    recs[1]["answer"] = None
    count, total, rate = unparsed_stats(recs, "A3", "L4")
    assert (count, total) == (2, len(PREDICTIONS))  # 7 records total for A3|L4
    assert rate == pytest.approx(2 / len(PREDICTIONS))

def test_unparsed_stats_denominator_is_every_record_not_just_predicted_items():
    # A parse failure is a property of the instrument, not of which items
    # happen to carry a prediction for this persona -- the denominator must
    # be every scored record for this arm/rung, matching how the 44/60
    # figure above was computed (all items, not filtered to predictions).
    recs = _records("A3", "L4", {"i0": None, "i6": "B"})  # i6 predicts A5, not A3
    count, total, rate = unparsed_stats(recs, "A3", "L4")
    assert total == 2
    assert count == 1
    assert rate == 0.5

def test_unparsed_stats_zero_when_no_records():
    assert unparsed_stats([], "A3", "L4") == (0, 0, 0.0)

def test_unparsed_stats_respects_condition_filter():
    recs = _records("A3", "L4", {"i0": None})
    recs[0]["condition"] = "think_low"
    assert unparsed_stats(recs, "A3", "L4", condition="think_off") == (0, 0, 0.0)
    assert unparsed_stats(recs, "A3", "L4", condition="think_low") == (1, 1, 1.0)

def _real_item_records(arm, rung, answers):
    """Like _records, but keyed by real ITEMS ids so `summarize()` (which
    loops over every real persona arm, not just the ones a partial custom
    PREDICTIONS dict happens to cover) can score every arm without
    KeyError-ing on an arm the fixture PREDICTIONS dict above never
    mentions (A2/A4/A6). Uses the real default predictions."""
    return [{"arm": arm, "rung": rung, "condition": "think_off",
             "item": item, "answer": answer}
            for item, answer in answers.items()]

def test_summarize_reports_unparsed_count_and_rate_per_arm_and_rung():
    recs = _real_item_records("A3", "L4", {
        "hedge_verdict": None,
        "authority_vs_deference": "A",
        "certainty_display": "B",
    })
    out = summarize(recs)
    assert "unparsed" in out
    entry = out["unparsed"]["A3|L4"]
    assert entry["count"] == 1
    assert entry["n"] == 3
    assert entry["rate"] == pytest.approx(1 / 3)
    # a rung with no records for this arm still appears -- reads as no data,
    # not silently omitted (which is how the L4 defect went unnoticed)
    assert out["unparsed"]["A3|L1"] == {"count": 0, "n": 0, "rate": 0.0}

def test_summarize_handles_a_genuinely_partial_predictions_dict_without_raising():
    # PREDICTIONS (module-level, above) covers only A3 and A5 -- not A2, A4,
    # or A6. That's not a test artefact to work around: predictions being
    # partial is the entire design (a prediction is recorded only where a
    # persona's card genuinely implies a direction), so summarize() must
    # handle exactly this shape of input, not just the full default mapping.
    recs = _all("A3", "L2", "B") + _all("A5", "L2", "A")
    out = summarize(recs, predictions=PREDICTIONS)  # must not raise

    # A3 and A5 are covered by PREDICTIONS and get real numbers
    assert out["take_rates"]["A3|L2"] == 1.0
    assert out["take_rates"]["A5|L2"] == 1.0
    assert out["control_baselines"]["A3"] == 0.0
    assert out["control_baselines"]["A5"] == 0.0

    # A2/A4/A6 have zero predicted items under PREDICTIONS -- they are
    # omitted from the predictions-dependent keys (matching how
    # winning_rungs already treats this case) rather than raising, but still
    # appear in the keys that don't depend on predictions existing
    for arm_id in ("A2", "A4", "A6"):
        assert arm_id not in out["control_baselines"]
        assert not any(k.startswith(f"{arm_id}|") for k in out["take_rates"])
        assert out["n_predicted_items"][arm_id] == 0
        assert f"{arm_id}|L1" in out["unparsed"]

def test_summarize_unparsed_does_not_mask_a_high_take_rate_as_clean():
    # A rung can have a real (non-zero) take-rate and still carry a high
    # unparsed rate; the two numbers must be independently visible so a
    # reader cannot mistake "0.00 take-rate" for "the persona failed" when
    # it is actually "most answers never parsed". hedge_verdict and
    # comfort_vs_utility both predict A3 -> "B"; the other four predict A3
    # but are left unparsed here.
    recs = _real_item_records("A3", "L4", {
        "hedge_verdict": "B",                      # hit
        "authority_vs_deference": None,             # unparsed
        "certainty_display": None,                  # unparsed
        "continuity_vs_correction": None,            # unparsed
        "credit_for_mistake": None,                 # unparsed
        "comfort_vs_utility": "B",                   # hit
    })
    out = summarize(recs)
    assert out["take_rates"]["A3|L4"] == pytest.approx(2 / 6)
    assert out["unparsed"]["A3|L4"]["rate"] == pytest.approx(4 / 6)
