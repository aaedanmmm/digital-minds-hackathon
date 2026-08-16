"""Take-rate scoring.

Two properties this module exists to preserve:

Predictions are optional per persona, so every denominator is the number of
items predicting THAT persona — never a fixed 12. The elicitation threshold is
consequently a proportion (one third), fixed by the spec before any run and not
to be tuned after seeing results.

Control arms carry no predictions of their own. They are scored against the
persona's predictions: how often does the model, with no persona prompt,
already answer the way this persona is predicted to? Scoring a control against
its own (nonexistent) predictions yields zero for every arm and makes every
persona look elicited.
"""
import json
from pathlib import Path
from personas.definitions import ARMS, ITEMS, RUNGS

CONTROLS = ("A0", "A1")
MARGIN = 1 / 3


def default_predictions() -> dict[str, dict[str, str]]:
    return {item.id: item.predicted for item in ITEMS}


def load_records(directory: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in Path(directory).glob("*.json")]


def take_rate(records, scored_arm: str, target_persona: str, rung: str,
              condition: str = "think_off", predictions=None) -> float:
    """Fraction of target_persona's predicted items that scored_arm matches.

    scored_arm and target_persona differ when scoring a control: we ask how
    often A0 lands on A3's predicted side without ever being told to.
    """
    predictions = predictions if predictions is not None else default_predictions()
    predicted_items = {item: preds[target_persona]
                       for item, preds in predictions.items()
                       if target_persona in preds}
    if not predicted_items:
        raise KeyError(f"no items carry a prediction for {target_persona}")
    rows = [r for r in records
            if r["arm"] == scored_arm and r["rung"] == rung
            and r["condition"] == condition and r["item"] in predicted_items]
    if not rows:
        return 0.0
    hits = sum(1 for r in rows
               if r["answer"] is not None
               and r["answer"] == predicted_items[r["item"]])
    return hits / len(rows)


def control_baseline(records, target_persona: str, condition: str = "think_off",
                     predictions=None) -> float:
    """The better of the two controls, scored against this persona."""
    return max(take_rate(records, control, target_persona, "L1", condition,
                         predictions)
               for control in CONTROLS)


def _has_predictions(predictions: dict, target_persona: str) -> bool:
    return any(target_persona in preds for preds in predictions.values())


def winning_rungs(records, margin: float = MARGIN,
                  predictions=None) -> dict[str, str | None]:
    predictions = predictions if predictions is not None else default_predictions()
    winners: dict[str, str | None] = {}
    for arm_id, arm in ARMS.items():
        if arm.kind != "persona":
            continue
        if not _has_predictions(predictions, arm_id):
            # Nothing predicts this persona under the given predictions dict
            # (only possible with a partial/custom predictions mapping, e.g.
            # in tests — every real persona in ITEMS has at least one
            # prediction). Skip rather than let take_rate's KeyError guard
            # (meant to catch scoring-against-nonexistent-predictions bugs)
            # abort the whole summary.
            continue
        baseline = control_baseline(records, arm_id, predictions=predictions)
        winners[arm_id] = None
        for rung in RUNGS:  # ascending, so the lowest clearing rung wins
            rate = take_rate(records, arm_id, arm_id, rung,
                             predictions=predictions)
            if rate >= baseline + margin:
                winners[arm_id] = rung
                break
    return winners


def unparsed_stats(records, arm_id: str, rung: str,
                   condition: str = "think_off") -> tuple[int, int, float]:
    """Count and rate of unparsed (answer is None) records for one
    arm/rung/condition. Returns (unparsed_count, total_count, rate).

    The denominator is every record for this arm/rung/condition, not just
    the items that happen to carry a prediction for this persona: a parse
    failure is a property of the instrument (the model never emitted an
    <answer> tag before hitting the token cap), not of which items were
    chosen to score this particular persona. Filtering the denominator down
    to predicted items would hide exactly the failure this function exists
    to surface -- see the L4 defect this module's docstring context
    describes, where 44 of 60 L4 answers never parsed.

    rate is 0.0 when total_count is 0, so an arm/rung with no records at all
    reads as "no data" through the paired total rather than a misleading
    rate on its own.
    """
    rows = [r for r in records
            if r["arm"] == arm_id and r["rung"] == rung
            and r["condition"] == condition]
    total = len(rows)
    unparsed = sum(1 for r in rows if r["answer"] is None)
    rate = unparsed / total if total else 0.0
    return unparsed, total, rate


def summarize(records, predictions=None) -> dict:
    """Predictions being partial is the whole design of this study (a
    prediction is recorded only where a persona's card genuinely implies a
    direction -- see the module docstring). A `predictions` mapping that
    doesn't cover every real persona arm is therefore the normal case, not
    an error condition, and must not make `summarize` raise. Arms with zero
    predicted items under the given `predictions` are omitted from
    `control_baselines` and `take_rates` (the two keys whose scoring is
    undefined without at least one predicted item -- `take_rate` raises
    `KeyError` for exactly this reason when called directly), matching how
    `winning_rungs` already treats the same case. `n_predicted_items` and
    `unparsed` do not depend on predictions existing, so every real persona
    arm still appears in both regardless.
    """
    personas = [a for a in ARMS.values() if a.kind == "persona"]
    preds = predictions if predictions is not None else default_predictions()
    scoreable = [arm for arm in personas if _has_predictions(preds, arm.id)]
    return {
        "n_records": len(records),
        "n_predicted_items": {
            arm.id: sum(1 for p in preds.values() if arm.id in p)
            for arm in personas
        },
        "control_baselines": {
            arm.id: control_baseline(records, arm.id, predictions=predictions)
            for arm in scoreable
        },
        "take_rates": {
            f"{arm.id}|{rung}": take_rate(records, arm.id, arm.id, rung,
                                          predictions=predictions)
            for arm in scoreable for rung in RUNGS
        },
        "unparsed": {
            f"{arm.id}|{rung}": dict(zip(
                ("count", "n", "rate"), unparsed_stats(records, arm.id, rung)))
            for arm in personas for rung in RUNGS
        },
        "winning_rungs": winning_rungs(records, predictions=predictions),
    }
