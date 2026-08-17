#!/usr/bin/env python3
"""Analyze the focused k=5 versus k=10 Vertex replication."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import beta, binomtest, fisher_exact, wilcoxon

from houses import CONFIRMATORY_PAIRS, STUDY_PAIRS, TARGET_PAIRS


def beta_edge(wins: int, total: int) -> dict:
    posterior = beta(wins + 1, total - wins + 1)
    low, high = posterior.ppf((0.025, 0.975))
    return {
        "wins": wins,
        "n": total,
        "rate": wins / total,
        "posterior_p_gt_half": float(1 - posterior.cdf(0.5)),
        "credible_interval_95": [float(low), float(high)],
    }


def exact_mcnemar(left_worse: int, right_worse: int) -> float:
    discordant = left_worse + right_worse
    if not discordant:
        return 1.0
    return float(binomtest(min(left_worse, right_worse), discordant, 0.5).pvalue)


def summarize(rows: list[dict]) -> dict:
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["pair"], row["n_criteria"])].append(row)

    pair_results = []
    consistency: dict[int, dict[str, bool | None]] = {5: {}, 10: {}}
    for pair in STUDY_PAIRS:
        canonical = pair[0]
        for complexity in (5, 10):
            subset = by_cell[(pair, complexity)]
            orders: dict[str, list[dict]] = defaultdict(list)
            for row in subset:
                orders[row["first"] + row["second"]].append(row)
            order_winners = {}
            for order, order_rows in orders.items():
                counts = Counter(row["winner"] for row in order_rows)
                top = counts.most_common()
                order_winners[order] = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None
            values = list(order_winners.values())
            consistent = len(values) == 2 and None not in values and values[0] == values[1]
            consistency[complexity][pair] = consistent if None not in values else None
            wins = sum(row["winner"] == canonical for row in subset)
            pair_results.append(
                {
                    "pair": pair,
                    "selection": "targeted" if pair in TARGET_PAIRS else "confirmatory_random",
                    "n_criteria": complexity,
                    "canonical_edge": f"{canonical}>{pair[1]}",
                    "edge": beta_edge(wins, len(subset)),
                    "order_winners": order_winners,
                    "order_consistent": consistency[complexity][pair],
                    "first_position_rate": sum(row["selected_position"] == "first" for row in subset) / len(subset),
                }
            )

    valid_pairs = [
        pair for pair in STUDY_PAIRS
        if consistency[5][pair] is not None and consistency[10][pair] is not None
    ]
    five_worse = sum(not consistency[5][pair] and consistency[10][pair] for pair in valid_pairs)
    ten_worse = sum(consistency[5][pair] and not consistency[10][pair] for pair in valid_pairs)
    aggregate = {}
    for complexity in (5, 10):
        subset = [row for row in rows if row["n_criteria"] == complexity]
        flags = [consistency[complexity][pair] for pair in valid_pairs]
        thought_counts = [
            row.get("usage_metadata", {}).get("thoughtsTokenCount", 0)
            for row in subset
        ]
        aggregate[str(complexity)] = {
            "trials": len(subset),
            "order_consistent_pairs": sum(flags),
            "pairs": len(flags),
            "order_consistency": sum(flags) / len(flags),
            "first_position_rate": sum(row["selected_position"] == "first" for row in subset) / len(subset),
            "mean_thought_tokens": sum(thought_counts) / len(thought_counts),
            "first_position_binomial_p": float(
                binomtest(
                    sum(row["selected_position"] == "first" for row in subset),
                    len(subset),
                    0.5,
                ).pvalue
            ),
        }
    first5 = round(aggregate["5"]["first_position_rate"] * aggregate["5"]["trials"])
    first10 = round(aggregate["10"]["first_position_rate"] * aggregate["10"]["trials"])
    pair_position_changes = []
    for pair in STUDY_PAIRS:
        rates = []
        for complexity in (5, 10):
            subset = by_cell[(pair, complexity)]
            rates.append(sum(row["selected_position"] == "first" for row in subset) / len(subset))
        pair_position_changes.append(rates[1] - rates[0])
    aggregate["first_position_k5_vs_k10_fisher_p"] = float(
        fisher_exact(
            [
                [first5, aggregate["5"]["trials"] - first5],
                [first10, aggregate["10"]["trials"] - first10],
            ]
        ).pvalue
    )
    aggregate["first_position_pair_level_mean_change"] = sum(pair_position_changes) / len(pair_position_changes)
    aggregate["first_position_pair_level_wilcoxon_p"] = float(wilcoxon(pair_position_changes).pvalue)
    group_results = {}
    for group_name, group_pairs in (
        ("targeted", TARGET_PAIRS),
        ("confirmatory_random", CONFIRMATORY_PAIRS),
        ("combined", STUDY_PAIRS),
    ):
        group_valid = [
            pair for pair in group_pairs
            if consistency[5][pair] is not None and consistency[10][pair] is not None
        ]
        group_five_worse = sum(
            not consistency[5][pair] and consistency[10][pair] for pair in group_valid
        )
        group_ten_worse = sum(
            consistency[5][pair] and not consistency[10][pair] for pair in group_valid
        )
        group_results[group_name] = {
            "pairs": len(group_valid),
            "k5_order_consistent": sum(consistency[5][pair] for pair in group_valid),
            "k10_order_consistent": sum(consistency[10][pair] for pair in group_valid),
            "k5_worse_k10_consistent": group_five_worse,
            "k10_worse_k5_consistent": group_ten_worse,
            "exact_mcnemar_p": exact_mcnemar(group_five_worse, group_ten_worse),
        }
    return {
        "records": len(rows),
        "target_pairs": list(TARGET_PAIRS),
        "confirmatory_pairs": list(CONFIRMATORY_PAIRS),
        "order_consistency_by_selection": group_results,
        "aggregate": aggregate,
        "paired_order_consistency": {
            "k5_worse_k10_consistent": five_worse,
            "k10_worse_k5_consistent": ten_worse,
            "discordant_pairs": five_worse + ten_worse,
            "exact_mcnemar_p": exact_mcnemar(five_worse, ten_worse),
        },
        "pair_results": pair_results,
    }


def markdown(summary: dict) -> str:
    a5, a10 = summary["aggregate"]["5"], summary["aggregate"]["10"]
    paired = summary["paired_order_consistency"]
    lines = [
        "# Focused Gemini preference-coherence replication",
        "",
        f"Valid records: **{summary['records']}**.",
        "",
        "| Pair selection | Pairs | k=5 consistent | k=10 consistent | k=10 worse / improved | Exact p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("targeted", "confirmatory_random", "combined"):
        group = summary["order_consistency_by_selection"][name]
        lines.append(
            f"| {name} | {group['pairs']} | {group['k5_order_consistent']}/{group['pairs']} "
            f"| {group['k10_order_consistent']}/{group['pairs']} "
            f"| {group['k10_worse_k5_consistent']} / {group['k5_worse_k10_consistent']} "
            f"| {group['exact_mcnemar_p']:.4f} |"
        )
    lines.extend([
        "",
        "| Attributes | Order-consistent pairs | First-position rate |",
        "|---|---:|---:|",
        f"| 5 | {a5['order_consistent_pairs']}/{a5['pairs']} ({a5['order_consistency']:.1%}) | {a5['first_position_rate']:.1%} |",
        f"| 10 | {a10['order_consistent_pairs']}/{a10['pairs']} ({a10['order_consistency']:.1%}) | {a10['first_position_rate']:.1%} |",
        "",
        f"Paired exact McNemar p-value: **{paired['exact_mcnemar_p']:.4f}** "
        f"({paired['k10_worse_k5_consistent']} pairs worse at k=10; "
        f"{paired['k5_worse_k10_consistent']} worse at k=5).",
        "",
        f"The first-position rate changes by **{summary['aggregate']['first_position_pair_level_mean_change']:+.1%}** "
        f"from k=5 to k=10 across the {len(STUDY_PAIRS)} pairs (pair-level Wilcoxon "
        f"**p={summary['aggregate']['first_position_pair_level_wilcoxon_p']:.4f}**; "
        f"trial-level Fisher **p={summary['aggregate']['first_position_k5_vs_k10_fisher_p']:.4f}**).",
        "",
        f"Mean actual thinking spend was **{a5['mean_thought_tokens']:.1f}** tokens at k=5 and "
        f"**{a10['mean_thought_tokens']:.1f}** at k=10 despite the same 512-token cap.",
        "",
        "## Pair-level results",
        "",
        "| Pair | k | First-house win rate | P(edge > 0.5) | Order consistent | First-position rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["pair_results"]:
        edge = row["edge"]
        lines.append(
            f"| {row['pair']} | {row['n_criteria']} | {edge['wins']}/{edge['n']} ({edge['rate']:.1%}) "
            f"| {edge['posterior_p_gt_half']:.1%} | {row['order_consistent']} "
            f"| {row['first_position_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The targeted stratum is post-hoc; the 12-pair random stratum was selected before its calls were made. Neither covers all 45 pairs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    records = []
    failures = []
    for path in sorted(args.input_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("winner"):
            records.append(row)
        else:
            failures.append({"trial_id": row.get("trial_id"), "error": row.get("error")})
    expected = len(STUDY_PAIRS) * 2 * 2 * 11
    if len(records) != expected:
        print(f"warning: found {len(records)}/{expected} valid records; failures={len(failures)}")
    if not records:
        return 1
    summary = summarize(records)
    summary["failures"] = failures
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consolidated = args.input_dir.parent / "raw_responses.jsonl"
    consolidated.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in sorted(records, key=lambda row: row["trial_id"])
        ),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "REPORT.md").write_text(markdown(summary))
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
