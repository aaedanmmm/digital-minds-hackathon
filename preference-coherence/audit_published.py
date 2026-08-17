#!/usr/bin/env python3
"""Audit the published 2,430-response experiment against a transitive null."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, binomtest, fisher_exact, wilcoxon


LETTERS = "ABCDEFGHIJ"
INDEX = {letter: index for index, letter in enumerate(LETTERS)}


def cycle_count(wins: np.ndarray, totals: np.ndarray) -> int:
    probabilities = np.divide(
        wins,
        totals,
        out=np.full_like(wins, np.nan, dtype=float),
        where=totals > 0,
    )
    beats = probabilities > 0.5
    count = 0
    for i, j, k in combinations(range(10), 3):
        count += int(
            (beats[i, j] and beats[j, k] and beats[k, i])
            or (beats[i, k] and beats[k, j] and beats[j, i])
        )
    return count


def statistics(rows: list[dict], simulated_first_wins: np.ndarray | None = None) -> tuple[int, float]:
    wins = np.zeros((10, 10))
    totals = np.zeros((10, 10))
    per_order: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        first, second = INDEX[row["first"]], INDEX[row["second"]]
        if simulated_first_wins is None:
            winner = INDEX[row["choice"]]
        else:
            winner = first if simulated_first_wins[index] else second
        totals[first, second] += 1
        totals[second, first] += 1
        loser = second if winner == first else first
        wins[winner, loser] += 1
        per_order[(first, second)].append(winner)
    agreements = 0
    for first, second in combinations(range(10), 2):
        forward = Counter(per_order[(first, second)]).most_common(1)[0][0]
        reverse = Counter(per_order[(second, first)]).most_common(1)[0][0]
        agreements += forward == reverse
    return cycle_count(wins, totals), agreements / 45


def fit_transitive_position_model(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    design, outcomes = [], []
    for row in rows:
        first, second = INDEX[row["first"]], INDEX[row["second"]]
        attributes = np.zeros(10)
        attributes[first] = 1
        attributes[second] = -1
        design.append(np.r_[attributes[:9], 1.0])
        outcomes.append(row["choice"] == row["first"])
    x = np.asarray(design)
    y = np.asarray(outcomes, dtype=float)

    def objective(parameters: np.ndarray) -> float:
        logits = x @ parameters
        # Weak ridge stabilization matters for near-perfectly separated pairs.
        return float(np.logaddexp(0, logits).sum() - y @ logits + 0.05 * (parameters @ parameters))

    fit = minimize(objective, np.zeros(10), method="BFGS")
    probabilities = 1 / (1 + np.exp(-(x @ fit.x)))
    return probabilities, fit.x


def null_simulation(rows: list[dict], simulations: int, rng: np.random.Generator) -> dict:
    observed_cycles, observed_order = statistics(rows)
    probabilities, parameters = fit_transitive_position_model(rows)
    cycles = np.empty(simulations, dtype=int)
    order = np.empty(simulations)
    for index in range(simulations):
        cycles[index], order[index] = statistics(rows, rng.random(len(rows)) < probabilities)
    return {
        "observed_cycles": observed_cycles,
        "observed_order_consistency": observed_order,
        "position_log_odds": float(parameters[-1]),
        "null_cycles_median": float(np.median(cycles)),
        "null_cycles_interval_95": [float(x) for x in np.quantile(cycles, (0.025, 0.975))],
        "cycle_tail_probability": float((1 + np.sum(cycles >= observed_cycles)) / (simulations + 1)),
        "null_order_median": float(np.median(order)),
        "null_order_interval_95": [float(x) for x in np.quantile(order, (0.025, 0.975))],
        "order_tail_probability": float((1 + np.sum(order <= observed_order)) / (simulations + 1)),
    }


def order_flags(rows: list[dict], complexity: int) -> dict[str, bool]:
    per_order: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        if row["n_criteria"] == complexity:
            per_order[(row["first"], row["second"])][row["choice"]] += 1
    flags = {}
    for first, second in combinations(LETTERS, 2):
        flags[first + second] = (
            per_order[(first, second)].most_common(1)[0][0]
            == per_order[(second, first)].most_common(1)[0][0]
        )
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/published-audit/summary.json"))
    parser.add_argument("--simulations", type=int, default=5000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("choice")]
    rng = np.random.default_rng(2_026_081_700)

    flags5, flags10 = order_flags(rows, 5), order_flags(rows, 10)
    five_worse = sum(not flags5[pair] and flags10[pair] for pair in flags5)
    ten_worse = sum(flags5[pair] and not flags10[pair] for pair in flags5)
    position = {}
    for complexity in (5, 10):
        subset = [row for row in rows if row["n_criteria"] == complexity]
        first = sum(row["choice"] == row["first"] for row in subset)
        position[complexity] = {"first": first, "n": len(subset), "rate": first / len(subset)}
    paired_differences = []
    for pair in combinations(LETTERS, 2):
        pair_name = "".join(pair)
        rates = []
        for complexity in (5, 10):
            subset = [
                row for row in rows
                if row["n_criteria"] == complexity and set((row["first"], row["second"])) == set(pair)
            ]
            rates.append(sum(row["choice"] == row["first"] for row in subset) / len(subset))
        paired_differences.append(rates[1] - rates[0])

    cells = {}
    for complexity in (3, 5, 10):
        for budget in (512, 4096, 8192):
            subset = [
                row for row in rows
                if row["n_criteria"] == complexity and row["thinking_budget"] == budget
            ]
            cells[f"k{complexity}_tb{budget}"] = null_simulation(subset, args.simulations, rng)

    first5, first10 = position[5]["first"], position[10]["first"]
    report = {
        "records": len(rows),
        "order_consistency": {
            "k5": sum(flags5.values()) / 45,
            "k10": sum(flags10.values()) / 45,
            "k5_worse_k10_consistent": five_worse,
            "k10_worse_k5_consistent": ten_worse,
            "paired_exact_p": float(binomtest(min(five_worse, ten_worse), five_worse + ten_worse, 0.5).pvalue),
        },
        "position_bias": {
            "k5": position[5],
            "k10": position[10],
            "naive_fisher_p": float(
                fisher_exact(
                    [[first5, position[5]["n"] - first5], [first10, position[10]["n"] - first10]]
                ).pvalue
            ),
            "pair_level_mean_change": float(np.mean(paired_differences)),
            "pair_level_wilcoxon_p": float(wilcoxon(paired_differences).pvalue),
        },
        "transitive_null_by_cell": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
