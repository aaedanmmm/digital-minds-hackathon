#!/usr/bin/env python3
"""Analyze the 12-set confirmatory preference-coherence experiment."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from validate_prompt_sets import normalized_utilities


COMPLEXITIES = (3, 5, 7, 10)
OPTIONS = "ABCDEF"


def difficulty_lookup(option_sets: list[dict]) -> dict[tuple[str, str], str]:
    output = {}
    for option_set in option_sets:
        scores = normalized_utilities(option_set["options"])
        for left, right in itertools.combinations(OPTIONS, 2):
            gap = abs(scores[left] - scores[right])
            output[(option_set["set_id"], left + right)] = (
                "hard" if gap <= 0.05 else "medium" if gap <= 0.15 else "easy"
            )
    return output


def pair_metrics(rows: list[dict], difficulties: dict[tuple[str, str], str]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["set_id"], row["pair"], row["n_criteria"])].append(row)
    output = []
    for (set_id, pair, complexity), subset in sorted(grouped.items()):
        left = pair[0]
        left_first = [row for row in subset if row["first"] == left]
        left_second = [row for row in subset if row["second"] == left]
        first_rate = sum(row["winner"] == left for row in left_first) / len(left_first)
        second_rate = sum(row["winner"] == left for row in left_second) / len(left_second)
        first_majority = first_rate > 0.5
        second_majority = second_rate > 0.5
        output.append({
            "set_id": set_id,
            "pair": pair,
            "n_criteria": complexity,
            "difficulty": difficulties[(set_id, pair)],
            "left_win_rate_when_first": first_rate,
            "left_win_rate_when_second": second_rate,
            "order_effect": first_rate - second_rate,
            "order_consistent": first_majority == second_majority,
        })
    return output


def cycle_count(rows: list[dict]) -> int:
    majority = {}
    for left, right in itertools.combinations(OPTIONS, 2):
        subset = [row for row in rows if row["pair"] == left + right]
        counts = Counter(row["winner"] for row in subset)
        majority[(left, right)] = left if counts[left] > counts[right] else right if counts[right] > counts[left] else None

    def beats(left: str, right: str) -> bool:
        pair = tuple(sorted((left, right)))
        return majority[pair] == left

    cycles = 0
    for a, b, c in itertools.combinations(OPTIONS, 3):
        cycles += int(
            (beats(a, b) and beats(b, c) and beats(c, a))
            or (beats(a, c) and beats(c, b) and beats(b, a))
        )
    return cycles


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    average = float(np.mean(values))
    if len(values) < 2:
        return average, float("nan"), float("nan")
    margin = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return average, float(average - margin), float(average + margin)


def fit_transitive_model(rows: list[dict]) -> np.ndarray:
    design, outcomes = [], []
    for row in rows:
        first, second = OPTIONS.index(row["first"]), OPTIONS.index(row["second"])
        utilities = np.zeros(6)
        utilities[first] = 1
        utilities[second] = -1
        design.append(np.r_[utilities[:5], 1.0])
        outcomes.append(row["winner"] == row["first"])
    x = np.asarray(design)
    y = np.asarray(outcomes, dtype=float)

    def objective(parameters: np.ndarray) -> float:
        logits = x @ parameters
        return float(np.logaddexp(0, logits).sum() - y @ logits + 0.05 * (parameters @ parameters))

    fitted = minimize(objective, np.zeros(6), method="BFGS").x
    return 1 / (1 + np.exp(-(x @ fitted)))


def simulated_cycle_counts(rows: list[dict], simulations: int, rng: np.random.Generator) -> np.ndarray:
    probabilities = fit_transitive_model(rows)
    first_wins = rng.random((simulations, len(rows))) < probabilities
    pair_names = ["".join(pair) for pair in itertools.combinations(OPTIONS, 2)]
    majority = np.zeros((simulations, len(pair_names)), dtype=np.int8)
    for pair_index, pair in enumerate(pair_names):
        indices = [index for index, row in enumerate(rows) if row["pair"] == pair]
        left_wins = np.zeros(simulations, dtype=int)
        for index in indices:
            row = rows[index]
            left_wins += first_wins[:, index] if row["first"] == pair[0] else ~first_wins[:, index]
        majority[:, pair_index] = np.where(left_wins > len(indices) / 2, 1, np.where(left_wins < len(indices) / 2, -1, 0))
    pair_index = {pair: index for index, pair in enumerate(pair_names)}

    def beats(left: str, right: str) -> np.ndarray:
        pair = "".join(sorted((left, right)))
        direction = 1 if left < right else -1
        return majority[:, pair_index[pair]] == direction

    cycles = np.zeros(simulations, dtype=int)
    for a, b, c in itertools.combinations(OPTIONS, 3):
        cycles += (
            (beats(a, b) & beats(b, c) & beats(c, a))
            | (beats(a, c) & beats(c, b) & beats(b, a))
        )
    return cycles


def analyze(rows: list[dict], option_sets: list[dict]) -> dict:
    difficulties = difficulty_lookup(option_sets)
    pairs = pair_metrics(rows, difficulties)
    per_set_complexity = []
    for set_id in sorted({row["set_id"] for row in rows}):
        for complexity in COMPLEXITIES:
            subset = [
                record for record in pairs
                if record["set_id"] == set_id and record["n_criteria"] == complexity
            ]
            trial_subset = [
                row for row in rows
                if row["set_id"] == set_id and row["n_criteria"] == complexity
            ]
            per_set_complexity.append({
                "set_id": set_id,
                "n_criteria": complexity,
                "mean_order_effect": float(np.mean([record["order_effect"] for record in subset])),
                "order_consistency": sum(record["order_consistent"] for record in subset) / len(subset),
                "cycles": cycle_count(trial_subset),
            })

    complexity_summary = {}
    for complexity in COMPLEXITIES:
        trial_subset = [row for row in rows if row["n_criteria"] == complexity]
        pair_subset = [record for record in pairs if record["n_criteria"] == complexity]
        set_subset = [record for record in per_set_complexity if record["n_criteria"] == complexity]
        set_effects = [record["mean_order_effect"] for record in set_subset]
        effect_mean, effect_low, effect_high = mean_ci(set_effects)
        complexity_summary[str(complexity)] = {
            "trials": len(trial_subset),
            "pairs": len(pair_subset),
            "mean_order_effect": effect_mean,
            "mean_order_effect_ci95": [effect_low, effect_high],
            "mean_order_consistency": float(np.mean([record["order_consistency"] for record in set_subset])),
            "cycles": sum(record["cycles"] for record in set_subset),
            "possible_triples": len(set_subset) * math.comb(6, 3),
            "sets_with_cycles": sum(record["cycles"] > 0 for record in set_subset),
            "first_position_rate": sum(row["selected_position"] == "first" for row in trial_subset) / len(trial_subset),
            "schema_first_rate": sum(row["selected_id"] == row["schema_ids"][0] for row in trial_subset) / len(trial_subset),
            "mean_thought_tokens": float(np.mean([
                row.get("usage_metadata", {}).get("thoughtsTokenCount", 0) for row in trial_subset
            ])),
        }

    set_deltas = []
    for set_id in sorted({record["set_id"] for record in per_set_complexity}):
        by_complexity = {
            record["n_criteria"]: record
            for record in per_set_complexity if record["set_id"] == set_id
        }
        set_deltas.append({
            "set_id": set_id,
            "order_effect_5": by_complexity[5]["mean_order_effect"],
            "order_effect_10": by_complexity[10]["mean_order_effect"],
            "delta_10_minus_5": by_complexity[10]["mean_order_effect"] - by_complexity[5]["mean_order_effect"],
        })
    delta_values = [record["delta_10_minus_5"] for record in set_deltas]
    delta_mean, delta_low, delta_high = mean_ci(delta_values)
    primary = {
        "estimand": "set-level mean order-effect change, k10 minus k5",
        "n_sets": len(delta_values),
        "mean": delta_mean,
        "ci95": [delta_low, delta_high],
        "one_sample_t_p": float(stats.ttest_1samp(delta_values, 0).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(delta_values).pvalue) if any(delta_values) else 1.0,
        "sets_negative": sum(value < 0 for value in delta_values),
        "sets_positive": sum(value > 0 for value in delta_values),
        "sets_zero": sum(value == 0 for value in delta_values),
        "by_set": set_deltas,
    }

    consistency_deltas = []
    for set_id in sorted({record["set_id"] for record in per_set_complexity}):
        by_complexity = {
            record["n_criteria"]: record
            for record in per_set_complexity if record["set_id"] == set_id
        }
        consistency_deltas.append(
            by_complexity[10]["order_consistency"] - by_complexity[5]["order_consistency"]
        )
    consistency_mean, consistency_low, consistency_high = mean_ci(consistency_deltas)
    consistency_contrast = {
        "mean": consistency_mean,
        "ci95": [consistency_low, consistency_high],
        "one_sample_t_p": float(stats.ttest_1samp(consistency_deltas, 0).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(consistency_deltas).pvalue) if any(consistency_deltas) else 1.0,
        "by_set": consistency_deltas,
    }

    rng = np.random.default_rng(2_026_081_700)
    simulations = 2000
    transitive_null = {}
    for complexity in COMPLEXITIES:
        aggregate_null = np.zeros(simulations, dtype=int)
        observed = 0
        for set_id in sorted({row["set_id"] for row in rows}):
            subset = [
                row for row in rows
                if row["set_id"] == set_id and row["n_criteria"] == complexity
            ]
            observed += cycle_count(subset)
            aggregate_null += simulated_cycle_counts(subset, simulations, rng)
        transitive_null[str(complexity)] = {
            "observed_cycles": observed,
            "null_median": float(np.median(aggregate_null)),
            "null_interval_95": [float(value) for value in np.quantile(aggregate_null, (0.025, 0.975))],
            "tail_probability": float((1 + np.sum(aggregate_null >= observed)) / (simulations + 1)),
            "simulations": simulations,
        }

    difficulty_summary = {}
    for label in ("hard", "medium", "easy"):
        difficulty_pairs = sorted({
            (record["set_id"], record["pair"]) for record in pairs if record["difficulty"] == label
        })
        changes = []
        for set_id, pair in difficulty_pairs:
            values = {
                record["n_criteria"]: record["order_effect"]
                for record in pairs if record["set_id"] == set_id and record["pair"] == pair
            }
            changes.append(values[10] - values[5])
        average, low, high = mean_ci(changes)
        difficulty_summary[label] = {
            "pairs": len(changes),
            "mean_delta_10_minus_5": average,
            "naive_pair_ci95": [low, high],
        }

    prompt_tokens = sum(row.get("usage_metadata", {}).get("promptTokenCount", 0) for row in rows)
    output_tokens = sum(
        row.get("usage_metadata", {}).get("candidatesTokenCount", 0)
        + row.get("usage_metadata", {}).get("thoughtsTokenCount", 0)
        for row in rows
    )
    return {
        "records": len(rows),
        "model_versions": sorted({row.get("model_version") for row in rows}),
        "complexity": complexity_summary,
        "primary_10_vs_5": primary,
        "order_consistency_10_vs_5": consistency_contrast,
        "transitive_cycle_null": transitive_null,
        "difficulty": difficulty_summary,
        "per_set_complexity": per_set_complexity,
        "pair_metrics": pairs,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "output_and_thought_tokens": output_tokens,
            "estimated_usd": prompt_tokens / 1_000_000 * 0.15 + output_tokens / 1_000_000 * 3.50,
        },
    }


def markdown(summary: dict) -> str:
    primary = summary["primary_10_vs_5"]
    consistency = summary["order_consistency_10_vs_5"]
    lines = [
        "# Twelve-set preference-coherence experiment",
        "",
        f"Valid trials: **{summary['records']:,}**; model: `{', '.join(summary['model_versions'])}`.",
        "",
        "## Primary result",
        "",
        "The preregistered probability-scale estimand is the change from five to ten attributes in the mean presentation-order effect, treating each option set as an independent unit.",
        "",
        f"Mean change: **{primary['mean']:+.3f}** (95% CI **[{primary['ci95'][0]:+.3f}, {primary['ci95'][1]:+.3f}]**); "
        f"set-level t-test **p={primary['one_sample_t_p']:.4f}**, Wilcoxon **p={primary['wilcoxon_p']:.4f}**. "
        f"Direction by set: {primary['sets_negative']} negative, {primary['sets_positive']} positive, {primary['sets_zero']} zero.",
        "",
        f"Secondary order-consistency change: **{consistency['mean']:+.3f}** "
        f"(95% CI **[{consistency['ci95'][0]:+.3f}, {consistency['ci95'][1]:+.3f}]**; "
        f"set-level t-test **p={consistency['one_sample_t_p']:.4f}**, Wilcoxon **p={consistency['wilcoxon_p']:.4f}**).",
        "",
        "## By attribute count",
        "",
        "| Attributes | Trials | Mean order effect | 95% CI | Order consistency | Cycles | Sets with cycles | First-position rate | Schema-first rate | Mean thought tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for complexity in COMPLEXITIES:
        record = summary["complexity"][str(complexity)]
        lines.append(
            f"| {complexity} | {record['trials']:,} | {record['mean_order_effect']:+.3f} "
            f"| [{record['mean_order_effect_ci95'][0]:+.3f}, {record['mean_order_effect_ci95'][1]:+.3f}] "
            f"| {record['mean_order_consistency']:.1%} | {record['cycles']}/{record['possible_triples']} "
            f"| {record['sets_with_cycles']}/12 | {record['first_position_rate']:.1%} "
            f"| {record['schema_first_rate']:.1%} | {record['mean_thought_tokens']:.1f} |"
        )
    lines.extend(["", "## Cycles versus fitted transitive null", "", "| Attributes | Observed | Null median | Null 95% interval | Tail probability |", "|---:|---:|---:|---:|---:|"])
    for complexity in COMPLEXITIES:
        record = summary["transitive_cycle_null"][str(complexity)]
        lines.append(
            f"| {complexity} | {record['observed_cycles']} | {record['null_median']:.1f} "
            f"| [{record['null_interval_95'][0]:.0f}, {record['null_interval_95'][1]:.0f}] "
            f"| {record['tail_probability']:.4f} |"
        )
    lines.extend(["", "## Difficulty strata", "", "| Difficulty | Pairs | Mean k10−k5 order-effect change | Naive pair-level 95% CI |", "|---|---:|---:|---:|"])
    for label in ("hard", "medium", "easy"):
        record = summary["difficulty"][label]
        lines.append(
            f"| {label} | {record['pairs']} | {record['mean_delta_10_minus_5']:+.3f} "
            f"| [{record['naive_pair_ci95'][0]:+.3f}, {record['naive_pair_ci95'][1]:+.3f}] |"
        )
    lines.extend([
        "",
        f"Estimated Vertex token cost: **US${summary['usage']['estimated_usd']:.2f}**.",
        "",
        "Difficulty intervals treat nested pairs as independent and are descriptive; the primary inference uses the 12 independent option-set means.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/large-study/raw"))
    parser.add_argument("--sets", type=Path, default=Path("prompt_sets/compiled_sets.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/large-study/analysis"))
    args = parser.parse_args()
    rows = []
    failures = []
    for path in sorted(args.input_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("winner"):
            rows.append(record)
        else:
            failures.append({"trial_id": record.get("trial_id"), "error": record.get("error")})
    if len(rows) != 10_080:
        raise SystemExit(f"expected 10,080 valid rows; found {len(rows)}; failures={len(failures)}")
    option_sets = json.loads(args.sets.read_text(encoding="utf-8"))["sets"]
    summary = analyze(rows, option_sets)
    summary["failures"] = failures
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "REPORT.md").write_text(markdown(summary), encoding="utf-8")
    consolidated = args.input_dir.parent / "raw_responses.jsonl"
    consolidated.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in sorted(rows, key=lambda row: row["trial_id"])
        ),
        encoding="utf-8",
    )
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
