#!/usr/bin/env python3
"""Summarize replicated preference choices, reasoning effects, and order bias."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path("q2/results/qwen-preference-reasoning-depth")
RAW = ROOT / "raw-results.json"
SUMMARY = ROOT / "summary.json"
RATES = ROOT / "preference-rates.csv"
TRANSITIONS = ROOT / "condition-transitions.csv"
CONDITIONS = ["none", "short", "long"]
DOMAINS = ["aesthetic", "utility"]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    proportion = successes / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return max(0, centre - margin), min(1, centre + margin)


def rate_summary(rows: list[dict], key: str, value: str) -> dict:
    successes = sum(row[key] == value for row in rows)
    low, high = wilson(successes, len(rows))
    return {"count": successes, "n": len(rows), "rate": successes / len(rows) if rows else None, "ci_low": low, "ci_high": high}


def exact_mcnemar(option_1_to_option_2: int, option_2_to_option_1: int) -> float:
    """Two-sided exact McNemar test over discordant matched pairs."""
    discordant = option_1_to_option_2 + option_2_to_option_1
    if discordant == 0:
        return 1.0
    tail = min(option_1_to_option_2, option_2_to_option_1)
    probability = 2 * sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, probability)


def main() -> None:
    study = json.loads(RAW.read_text())
    rows = [row for row in study["records"] if "canonical_choice" in row]
    preference_lookup = {(item["domain"], item["id"]): item for item in study["preferences"]}
    rates = []
    for (domain, preference_id), preference in preference_lookup.items():
        for condition in CONDITIONS:
            subset = [row for row in rows if row["domain"] == domain and row["preference_id"] == preference_id and row["condition"] == condition]
            option_2 = rate_summary(subset, "canonical_choice", "option_2")
            first = rate_summary(subset, "selected_position", "first")
            option_2_first = [row for row in subset if row["order"] == "21"]
            option_2_second = [row for row in subset if row["order"] == "12"]
            rates.append({
                "domain": domain,
                "preference_id": preference_id,
                "preference_title": preference["title"],
                "condition": condition,
                "n": len(subset),
                "option_2_count": option_2["count"],
                "option_2_rate": option_2["rate"],
                "option_2_ci_low": option_2["ci_low"],
                "option_2_ci_high": option_2["ci_high"],
                "first_position_count": first["count"],
                "first_position_rate": first["rate"],
                "option_2_rate_when_first": sum(row["canonical_choice"] == "option_2" for row in option_2_first) / len(option_2_first),
                "option_2_rate_when_second": sum(row["canonical_choice"] == "option_2" for row in option_2_second) / len(option_2_second),
                "median_completion_tokens": median(row["completion_tokens"] for row in subset),
                "median_reasoning_tokens": median(row["reasoning_tokens"] for row in subset),
            })

    transitions = []
    indexed = {(row["domain"], row["preference_id"], row["repetition"], row["condition"]): row for row in rows}
    for domain in DOMAINS + ["all"]:
        preference_keys = [key for key in preference_lookup if domain == "all" or key[0] == domain]
        for left, right in (("none", "short"), ("short", "long"), ("none", "long")):
            pairs = []
            for pref_domain, preference_id in preference_keys:
                for repetition in range(study["repetitions"]):
                    pairs.append((
                        indexed[(pref_domain, preference_id, repetition, left)],
                        indexed[(pref_domain, preference_id, repetition, right)],
                    ))
            switched = [(a, b) for a, b in pairs if a["canonical_choice"] != b["canonical_choice"]]
            option_1_to_option_2 = sum(a["canonical_choice"] == "option_1" and b["canonical_choice"] == "option_2" for a, b in switched)
            option_2_to_option_1 = sum(a["canonical_choice"] == "option_2" and b["canonical_choice"] == "option_1" for a, b in switched)
            transitions.append({
                "domain": domain,
                "from": left,
                "to": right,
                "n": len(pairs),
                "switches": len(switched),
                "switch_rate": len(switched) / len(pairs),
                "option_1_to_option_2": option_1_to_option_2,
                "option_2_to_option_1": option_2_to_option_1,
                "exact_mcnemar_p": exact_mcnemar(option_1_to_option_2, option_2_to_option_1),
            })

    domain_conditions = []
    for domain in DOMAINS + ["all"]:
        for condition in CONDITIONS:
            subset = [row for row in rows if row["condition"] == condition and (domain == "all" or row["domain"] == domain)]
            option_2 = rate_summary(subset, "canonical_choice", "option_2")
            first = rate_summary(subset, "selected_position", "first")
            reversed_rows = [row for row in subset if row["order"] == "21"]
            canonical_rows = [row for row in subset if row["order"] == "12"]
            per_preference = [rate for rate in rates if rate["condition"] == condition and (domain == "all" or rate["domain"] == domain)]
            domain_conditions.append({
                "domain": domain,
                "condition": condition,
                "n": len(subset),
                "option_2": option_2,
                "first_position": first,
                "option_2_rate_when_first": sum(row["canonical_choice"] == "option_2" for row in reversed_rows) / len(reversed_rows),
                "option_2_rate_when_second": sum(row["canonical_choice"] == "option_2" for row in canonical_rows) / len(canonical_rows),
                "order_effect_option_2_rate": (
                    sum(row["canonical_choice"] == "option_2" for row in reversed_rows) / len(reversed_rows)
                    - sum(row["canonical_choice"] == "option_2" for row in canonical_rows) / len(canonical_rows)
                ),
                "mixed_preferences": sum(0 < rate["option_2_count"] < rate["n"] for rate in per_preference),
                "median_completion_tokens": median(row["completion_tokens"] for row in subset),
                "median_reasoning_tokens": median(row["reasoning_tokens"] for row in subset),
                "mean_cost": mean(row["cost"] for row in subset if row["cost"] is not None),
            })

    effects = []
    for domain, preference_id in preference_lookup:
        none = next(rate for rate in rates if rate["domain"] == domain and rate["preference_id"] == preference_id and rate["condition"] == "none")
        short = next(rate for rate in rates if rate["domain"] == domain and rate["preference_id"] == preference_id and rate["condition"] == "short")
        long = next(rate for rate in rates if rate["domain"] == domain and rate["preference_id"] == preference_id and rate["condition"] == "long")
        effects.append({
            "domain": domain,
            "preference_id": preference_id,
            "preference_title": preference_lookup[(domain, preference_id)]["title"],
            "none_option_2_rate": none["option_2_rate"],
            "short_option_2_rate": short["option_2_rate"],
            "long_option_2_rate": long["option_2_rate"],
            "long_minus_none": long["option_2_rate"] - none["option_2_rate"],
        })

    summary = {
        "study": study["study"],
        "model": study["model"],
        "temperature": study["temperature"],
        "repetitions": study["repetitions"],
        "total_records": len(rows),
        "total_cost": sum(row["cost"] or 0 for row in rows),
        "domain_conditions": domain_conditions,
        "transitions": transitions,
        "preference_effects": effects,
        "preference_rates": rates,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    with RATES.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rates[0]))
        writer.writeheader()
        writer.writerows(rates)
    with TRANSITIONS.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)


if __name__ == "__main__":
    main()
