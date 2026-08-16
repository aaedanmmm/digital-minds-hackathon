#!/usr/bin/env python3
"""Create compact, analysis-ready summaries for the aesthetic reasoning study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path("results/aesthetic-reasoning")
RAW = ROOT / "raw-results.json"
SUMMARY = ROOT / "summary.json"
TABLE = ROOT / "condition-comparison.csv"
ORDER = ["none", "short", "long"]


def pairwise(rows: list[dict], left: str, right: str) -> dict:
    comparable = [row for row in rows if row[left] is not None and row[right] is not None]
    switches = [row for row in comparable if row[left] != row[right]]
    return {
        "from": left,
        "to": right,
        "comparable_scenarios": len(comparable),
        "switches": len(switches),
        "switch_rate": len(switches) / len(comparable) if comparable else None,
        "switched_scenarios": [row["scenario_id"] for row in switches],
    }


def main() -> None:
    study = json.loads(RAW.read_text())
    records = {(row["scenario_id"], row["condition"]): row for row in study["records"]}
    rows = []
    for scenario in study["scenarios"]:
        row = {"scenario_id": scenario["id"], "scenario_title": scenario["title"]}
        for condition in ORDER:
            record = records.get((scenario["id"], condition), {})
            row[condition] = record.get("answer")
            row[f"{condition}_reasoning_tokens"] = record.get("reasoning_tokens")
            row[f"{condition}_completion_tokens"] = record.get("completion_tokens")
        rows.append(row)

    by_condition = {}
    for condition in ORDER:
        condition_records = [record for record in study["records"] if record["condition"] == condition]
        token_values = [record["reasoning_tokens"] for record in condition_records if record["reasoning_tokens"] is not None]
        completion_values = [record["completion_tokens"] for record in condition_records if record["completion_tokens"] is not None]
        by_condition[condition] = {
            "n": len(condition_records),
            "parsed_answers": sum(record["answer"] is not None for record in condition_records),
            "answers": {answer: sum(record["answer"] == answer for record in condition_records) for answer in ["A", "B", None]},
            "mean_reasoning_tokens": mean(token_values) if token_values else None,
            "median_reasoning_tokens": median(token_values) if token_values else None,
            "mean_completion_tokens": mean(completion_values) if completion_values else None,
            "median_completion_tokens": median(completion_values) if completion_values else None,
        }

    summary = {
        "study": study["study"],
        "model": study["model"],
        "seed": study["seed"],
        "conditions": by_condition,
        "pairwise_choice_changes": [
            pairwise(rows, "none", "short"),
            pairwise(rows, "short", "long"),
            pairwise(rows, "none", "long"),
        ],
        "scenario_rows": rows,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    with TABLE.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
