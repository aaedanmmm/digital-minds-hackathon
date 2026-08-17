#!/usr/bin/env python3
"""Validate and compile the twelve independently authored rental sets."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


RANGES = (
    (900, 2800), (5, 60), (25, 145), (2, 9), (30, 70),
    (3, 24), (2, 25), (2, 10), (0, 65), (2, 10),
)
MINIMIZE = {0, 1, 4, 5, 6}


def dominates(left: list[int], right: list[int]) -> bool:
    oriented_left = [-value if i in MINIMIZE else value for i, value in enumerate(left)]
    oriented_right = [-value if i in MINIMIZE else value for i, value in enumerate(right)]
    return all(a >= b for a, b in zip(oriented_left, oriented_right, strict=True)) and any(
        a > b for a, b in zip(oriented_left, oriented_right, strict=True)
    )


def normalized_utilities(options: list[dict]) -> dict[str, float]:
    columns = list(zip(*(option["values"] for option in options), strict=True))
    scores = {option["id"]: 0.0 for option in options}
    for index, column in enumerate(columns):
        low, high = min(column), max(column)
        for option in options:
            value = option["values"][index]
            normalized = 0.5 if high == low else (value - low) / (high - low)
            if index in MINIMIZE:
                normalized = 1 - normalized
            scores[option["id"]] += normalized / len(columns)
    return scores


def difficulty_counts(options: list[dict]) -> dict[str, int]:
    scores = normalized_utilities(options)
    counts = {"hard": 0, "medium": 0, "easy": 0}
    for left, right in itertools.combinations(scores, 2):
        gap = abs(scores[left] - scores[right])
        if gap <= 0.05:
            counts["hard"] += 1
        elif gap <= 0.15:
            counts["medium"] += 1
        else:
            counts["easy"] += 1
    return counts


def validate_set(record: dict) -> dict:
    errors = []
    options = record.get("options", [])
    if [option.get("id") for option in options] != list("ABCDEF"):
        errors.append("option IDs must be A-F in order")
    rows = [option.get("values", []) for option in options]
    if len({tuple(row) for row in rows}) != len(rows):
        errors.append("duplicate option rows")
    for option in options:
        values = option.get("values", [])
        if len(values) != 10 or any(type(value) is not int for value in values):
            errors.append(f"{option.get('id')}: values must be ten integers")
            continue
        for index, (value, bounds) in enumerate(zip(values, RANGES, strict=True)):
            if not bounds[0] <= value <= bounds[1]:
                errors.append(f"{option['id']}: attribute {index + 1} out of range")
    if not errors:
        for left, right in itertools.permutations(options, 2):
            if dominates(left["values"], right["values"]):
                errors.append(f"{left['id']} Pareto-dominates {right['id']}")
        difficulty = difficulty_counts(options)
        for label in ("hard", "medium", "easy"):
            if difficulty[label] == 0:
                errors.append(f"no {label} pairs under normalized equal-weight utility")
    else:
        difficulty = {"hard": 0, "medium": 0, "easy": 0}
    return {"set_id": record.get("set_id"), "errors": errors, "difficulty": difficulty}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("prompt_sets"))
    parser.add_argument("--output", type=Path, default=Path("prompt_sets/compiled_sets.json"))
    args = parser.parse_args()
    sets = []
    for path in sorted(args.input_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sets.extend(payload.get("sets", []))
    expected = [f"S{index:02d}" for index in range(1, 13)]
    found = [record.get("set_id") for record in sets]
    reports = [validate_set(record) for record in sets]
    global_errors = []
    if found != expected:
        global_errors.append(f"set IDs must be exactly {expected}; found {found}")
    all_errors = global_errors + [
        f"{report['set_id']}: {error}" for report in reports for error in report["errors"]
    ]
    print(json.dumps({"sets": reports, "errors": all_errors}, indent=2))
    if all_errors:
        return 1
    args.output.write_text(json.dumps({"sets": sets}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
