#!/usr/bin/env python3
"""Run the focused preference-coherence replication on Vertex Gemini Flash.

The script obtains a short-lived access token from ``gcloud auth
print-access-token`` and calls the Vertex REST endpoint directly. Each trial is
written to its own JSON file, making the run resumable without a shared append
file or corruption risk under concurrency.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from houses import ATTRIBUTES, HOUSES, STUDY_PAIRS


DEFAULT_PROJECT = "secret-loyalty-apart"
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-2.5-flash"
COMPLEXITIES = (5, 10)
# Odd by design: an order-specific majority can never be tied.
REPS_PER_ORDER = 11
THINKING_BUDGET = 512
TEMPERATURE = 1.0
MASTER_SEED = 2_026_081_700


class TokenProvider:
    def __init__(self) -> None:
        self._token = ""
        self._issued_at = 0.0
        self._lock = threading.Lock()

    def get(self, *, refresh: bool = False) -> str:
        with self._lock:
            if refresh or not self._token or time.time() - self._issued_at > 3000:
                proc = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self._token = proc.stdout.strip()
                self._issued_at = time.time()
            return self._token


def opaque_ids(seed: int) -> tuple[str, str]:
    """Return deterministic, non-semantic identifiers with no A/B labels."""
    alphabet = "KLMNPQRSTUVWXYZ23456789"
    rng = random.Random(seed)
    values: list[str] = []
    while len(values) < 2:
        value = "".join(rng.choice(alphabet) for _ in range(4))
        if value not in values:
            values.append(value)
    return values[0], values[1]


def build_prompt(first: str, second: str, n_criteria: int, ids: tuple[str, str]) -> str:
    def listing(house: str, label: str) -> str:
        rows = []
        for (name, unit, direction), value in zip(
            ATTRIBUTES[:n_criteria], HOUSES[house][:n_criteria], strict=True
        ):
            rows.append(f"  - {name}: {value} {unit} ({direction})")
        return f"Listing {label}\n" + "\n".join(rows)

    return (
        "Choose the rental listing you personally prefer. You must select one; "
        "do not report a tie.\n\n"
        f"{listing(first, ids[0])}\n\n{listing(second, ids[1])}\n\n"
        "Return only the identifier of the listing you prefer."
    )


def build_jobs(reps_per_order: int) -> list[dict]:
    jobs = []
    for pair_index, pair in enumerate(STUDY_PAIRS):
        left, right = pair
        for n_criteria in COMPLEXITIES:
            for order_index, (first, second) in enumerate(((left, right), (right, left))):
                for rep in range(reps_per_order):
                    seed = (
                        MASTER_SEED
                        + pair_index * 100_000
                        + n_criteria * 1_000
                        + order_index * 100
                        + rep
                    )
                    ids = opaque_ids(seed)
                    trial_id = f"{pair}_k{n_criteria}_{first}{second}_r{rep:02d}"
                    jobs.append(
                        {
                            "trial_id": trial_id,
                            "pair": pair,
                            "n_criteria": n_criteria,
                            "first": first,
                            "second": second,
                            "replicate": rep,
                            "seed": seed,
                            "display_ids": ids,
                            "prompt": build_prompt(first, second, n_criteria, ids),
                        }
                    )
    random.Random(MASTER_SEED).shuffle(jobs)
    return jobs


def endpoint(project: str, location: str, model: str) -> str:
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}/locations/{location}/"
        f"publishers/google/models/{model}:generateContent"
    )


def request_body(job: dict) -> dict:
    first_id, second_id = job["display_ids"]
    schema_ids = job.get("schema_ids", (first_id, second_id))
    return {
        "contents": [{"role": "user", "parts": [{"text": job["prompt"]}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "seed": job["seed"],
            # Gemini counts hidden thinking tokens against maxOutputTokens.
            # Leave room for the 512-token thinking budget plus the tiny JSON
            # answer so constrained decoding is not cut off mid-response.
            "maxOutputTokens": 768,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "choice": {"type": "STRING", "enum": list(schema_ids)}
                },
                "required": ["choice"],
            },
            "thinkingConfig": {"thinkingBudget": THINKING_BUDGET},
        },
    }


def call_vertex(url: str, tokens: TokenProvider, job: dict, max_retries: int) -> dict:
    body = json.dumps(request_body(job)).encode()
    last_error = "unknown error"
    for attempt in range(1, max_retries + 1):
        token = tokens.get(refresh=attempt > 1 and "401" in last_error)
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            text = payload["candidates"][0]["content"]["parts"][-1]["text"]
            parsed = json.loads(text)
            selected_id = parsed["choice"]
            if selected_id not in job["display_ids"]:
                raise ValueError(f"unexpected choice identifier: {selected_id!r}")
            selected_position = "first" if selected_id == job["display_ids"][0] else "second"
            winner = job[selected_position]
            usage = payload.get("usageMetadata", {})
            return {
                **job,
                "display_ids": list(job["display_ids"]),
                "winner": winner,
                "selected_id": selected_id,
                "selected_position": selected_position,
                "temperature": TEMPERATURE,
                "thinking_budget": THINKING_BUDGET,
                "model_version": payload.get("modelVersion"),
                "usage_metadata": usage,
                "response": payload,
                "attempts": attempt,
                "latency_s": round(time.perf_counter() - started, 3),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")[:1000]
            last_error = f"HTTP {exc.code}: {message}"
            if exc.code not in {401, 408, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            time.sleep(min(20, 2**attempt) * (0.5 + random.random()))
    return {
        **job,
        "display_ids": list(job["display_ids"]),
        "error": last_error,
        "completed_at": datetime.now(UTC).isoformat(),
    }


def write_record(output_dir: Path, record: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record['trial_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reps-per-order", type=int, default=REPS_PER_ORDER)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--limit", type=int, help="run at most this many pending trials")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.reps_per_order)
    completed = set()
    for path in args.output_dir.glob("*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("winner"):
                completed.add(path.stem)
        except (OSError, json.JSONDecodeError):
            pass
    pending = [job for job in jobs if job["trial_id"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"model={args.model} project={args.project} location={args.location} "
        f"complete={len(completed)}/{len(jobs)} pending={len(pending)}"
    )
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
            if index % 20 == 0 or index == len(pending):
                print(f"{index}/{len(pending)} new trials; failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
