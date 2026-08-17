#!/usr/bin/env python3
"""Run the 12-set, 10,080-call confirmatory study through Vertex AI."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import random
from pathlib import Path

from houses import ATTRIBUTES
from run_vertex_replication import (
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_PROJECT,
    MASTER_SEED,
    TokenProvider,
    call_vertex,
    endpoint,
    opaque_ids,
    write_record,
)


COMPLEXITIES = (3, 5, 7, 10)
REPS_PER_ORDER = 7
N_SETS = 12


def balanced_subsets(complexity: int, n_sets: int = N_SETS) -> list[tuple[int, ...]]:
    if complexity == 10:
        return [tuple(range(10)) for _ in range(n_sets)]
    rng = random.Random(MASTER_SEED + complexity)
    candidates = list(itertools.combinations(range(10), complexity))
    rng.shuffle(candidates)
    counts = [0] * 10
    chosen: list[tuple[int, ...]] = []
    target = n_sets * complexity / 10
    for _ in range(n_sets):
        unused = [candidate for candidate in candidates if candidate not in chosen]
        best_score = None
        best = None
        for candidate in unused:
            proposed = [count + (index in candidate) for index, count in enumerate(counts)]
            score = (max(proposed) - min(proposed), sum((count - target) ** 2 for count in proposed))
            if best_score is None or score < best_score:
                best_score, best = score, candidate
        assert best is not None
        chosen.append(best)
        counts = [count + (index in best) for index, count in enumerate(counts)]
    assert max(counts) - min(counts) <= 1
    return chosen


def attribute_orders() -> dict[tuple[int, int], tuple[int, ...]]:
    output = {}
    for complexity in COMPLEXITIES:
        for set_index, subset in enumerate(balanced_subsets(complexity)):
            values = list(subset)
            random.Random(MASTER_SEED + complexity * 100 + set_index).shuffle(values)
            output[(set_index, complexity)] = tuple(values)
    return output


def build_prompt(options: dict[str, tuple[int, ...]], first: str, second: str,
                 attribute_indices: tuple[int, ...], ids: tuple[str, str]) -> str:
    def listing(option_id: str, label: str) -> str:
        rows = []
        for index in attribute_indices:
            name, unit, direction = ATTRIBUTES[index]
            rows.append(f"  - {name}: {options[option_id][index]} {unit} ({direction})")
        return f"Listing {label}\n" + "\n".join(rows)

    return (
        "Choose the rental listing you personally prefer. You must select one; "
        "do not report a tie.\n\n"
        f"{listing(first, ids[0])}\n\n{listing(second, ids[1])}\n\n"
        "Return only the identifier of the listing you prefer."
    )


def load_sets(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["sets"]
    if len(records) != N_SETS:
        raise ValueError(f"expected {N_SETS} sets, found {len(records)}")
    return records


def build_jobs(sets: list[dict], reps_per_order: int = REPS_PER_ORDER) -> list[dict]:
    orders = attribute_orders()
    jobs = []
    for set_index, option_set in enumerate(sets):
        options = {option["id"]: tuple(option["values"]) for option in option_set["options"]}
        for pair_index, (left, right) in enumerate(itertools.combinations("ABCDEF", 2)):
            pair = left + right
            for complexity in COMPLEXITIES:
                indices = orders[(set_index, complexity)]
                for direction, (first, second) in enumerate(((left, right), (right, left))):
                    for replicate in range(reps_per_order):
                        seed = (
                            MASTER_SEED + 10_000_000 + set_index * 1_000_000
                            + pair_index * 10_000 + complexity * 100
                            + direction * 10 + replicate
                        )
                        ids = opaque_ids(seed)
                        trial_id = (
                            f"{option_set['set_id']}_{pair}_k{complexity}_"
                            f"{first}{second}_r{replicate:02d}"
                        )
                        jobs.append({
                            "trial_id": trial_id,
                            "set_id": option_set["set_id"],
                            "pair": pair,
                            "n_criteria": complexity,
                            "attribute_indices": list(indices),
                            "first": first,
                            "second": second,
                            "replicate": replicate,
                            "seed": seed,
                            "display_ids": ids,
                            "prompt": build_prompt(options, first, second, indices, ids),
                        })
    for index, job in enumerate(jobs):
        ids = job["display_ids"]
        job["schema_ids"] = ids if index % 2 == 0 else (ids[1], ids[0])
    random.Random(MASTER_SEED + 10_000_000).shuffle(jobs)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=Path, default=Path("prompt_sets/compiled_sets.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/large-study/raw"))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(load_sets(args.sets))
    completed = set()
    for path in args.output_dir.glob("*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("winner"):
                completed.add(path.stem)
        except (OSError, json.JSONDecodeError):
            pass
    pending = [job for job in jobs if job["trial_id"] not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"model={args.model} complete={len(completed)}/{len(jobs)} pending={len(pending)}")
    if args.dry_run:
        print(json.dumps({key: value for key, value in jobs[0].items() if key != "prompt"}, indent=2))
        print("\n" + jobs[0]["prompt"])
        return 0
    if not pending:
        return 0

    tokens = TokenProvider()
    url = endpoint(args.project, args.location, args.model)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(call_vertex, url, tokens, job, args.max_retries): job
            for job in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            write_record(args.output_dir, record)
            failures += "error" in record
            if index % 100 == 0 or index == len(pending):
                print(f"{index}/{len(pending)} new trials; failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
