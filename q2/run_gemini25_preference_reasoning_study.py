#!/usr/bin/env python3
"""Replicate the Q2 preference-depth study on Gemini 2.5 Flash via Vertex AI.

The prompts, paired seeds, repetitions, and option-order balancing are imported
from the Qwen runner. Gemini thinking is disabled for the answer-only and brief
rationale conditions and given a 4,096-token budget for long reasoning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from run_qwen_preference_reasoning_study import (
    ANSWER_RE,
    CONDITIONS,
    PREFERENCES,
    REPETITIONS,
    atomic_write,
    build_job,
)


MODEL = "gemini-2.5-flash"
LOCATION = "global"
THINKING_BUDGETS = {"none": 0, "short": 0, "long": 4096}
MAX_OUTPUT_TOKENS = {"none": 128, "short": 96, "long": 8192}
EXPLICIT_CHOICE_RE = re.compile(
    r"(?:my (?:final )?choice is|i (?:would )?choose|i (?:would )?prefer|final choice(?: is)?)[ :]*(?:option )?([AB])\b",
    re.I,
)
BOXED_CHOICE_RE = re.compile(r"\\boxed\{([AB])\}", re.I)


def command_value(*args: str) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


class GCloudToken:
    """Thread-safe access-token cache with early refresh."""

    def __init__(self) -> None:
        self._token = ""
        self._issued_at = 0.0
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            if force or not self._token or time.monotonic() - self._issued_at > 2_400:
                self._token = command_value("gcloud", "auth", "print-access-token")
                self._issued_at = time.monotonic()
            return self._token


def response_texts(payload: dict) -> tuple[str, str]:
    candidates = payload.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    thoughts = []
    answers = []
    for part in parts:
        text = part.get("text") or ""
        if not text:
            continue
        (thoughts if part.get("thought") else answers).append(text)
    return "\n".join(answers), "\n".join(thoughts)


def parse_choice(content: str) -> tuple[str | None, str | None]:
    for name, pattern in (
        ("answer_tag", ANSWER_RE),
        ("explicit_declaration", EXPLICIT_CHOICE_RE),
        ("boxed_answer", BOXED_CHOICE_RE),
    ):
        matches = pattern.findall(content)
        if matches:
            return matches[-1].upper(), name
    return None, None


def add_choice_fields(record: dict, displayed_choice: str) -> None:
    record["displayed_choice"] = displayed_choice
    record["canonical_choice"] = (
        ("option_1" if displayed_choice == "A" else "option_2")
        if record["order"] == "12"
        else ("option_2" if displayed_choice == "A" else "option_1")
    )
    record["selected_position"] = "first" if displayed_choice == "A" else "second"


def call_vertex(
    token_cache: GCloudToken,
    endpoint: str,
    job: dict,
    dry_run: bool = False,
) -> dict:
    condition = job["condition"]
    body = {
        "contents": [{"role": "user", "parts": [{"text": job["prompt"]}]}],
        "generationConfig": {
            "temperature": 0.7,
            "seed": job["seed"],
            # Gemini counts returned thoughts and the final response against the
            # same ceiling. Long calls therefore need headroom above the fixed
            # 4,096-token thinking budget to reliably emit the answer tag.
            "maxOutputTokens": MAX_OUTPUT_TOKENS[condition],
            "thinkingConfig": {
                "thinkingBudget": THINKING_BUDGETS[condition],
                "includeThoughts": condition == "long",
            },
        },
    }
    if dry_run:
        return body

    last_error = "unknown error"
    last_payload: dict = {}
    last_content = ""
    last_reasoning = ""
    for attempt in range(6):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token_cache.get(force=attempt > 0 and last_error.startswith('HTTP 401'))}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = json.load(response)
            content, reasoning = response_texts(payload)
            last_payload = payload
            last_content = content
            last_reasoning = reasoning
            displayed_choice, parser_format = parse_choice(content)
            if not displayed_choice:
                last_error = f"missing answer tag: {content[:300]!r}"
                time.sleep(1 + attempt)
                continue
            usage = payload.get("usageMetadata") or {}
            record = {
                **{key: value for key, value in job.items() if key != "prompt"},
                "content": content,
                "reasoning": reasoning,
                "reasoning_tokens": usage.get("thoughtsTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount"),
                "prompt_tokens": usage.get("promptTokenCount"),
                "total_tokens": usage.get("totalTokenCount"),
                "raw_response": payload,
                "requested_at": datetime.now(UTC).isoformat(),
            }
            add_choice_fields(record, displayed_choice)
            record["parser_format"] = parser_format
            return record
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:800]}"
            if exc.code < 500 and exc.code not in (401, 408, 429):
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(min(30, 2**attempt))
    return {
        **{key: value for key, value in job.items() if key != "prompt"},
        "error": last_error,
        "content": last_content,
        "reasoning": last_reasoning,
        "raw_response": last_payload,
        "requested_at": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("q2/results/gemini25-preference-reasoning-depth/raw-results.json"),
    )
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--smoke", action="store_true", help="run one long-reasoning request")
    parser.add_argument("--dry-run", action="store_true", help="print one request without calling Vertex")
    parser.add_argument("--repair-saved", action="store_true", help="reparse saved error responses without API calls")
    args = parser.parse_args()

    project = args.project or command_value("gcloud", "config", "get-value", "project")
    if not project or project == "(unset)":
        raise SystemExit("No GCloud project configured; pass --project")
    endpoint = (
        f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/{args.location}"
        f"/publishers/google/models/{args.model}:generateContent"
    )
    jobs = [
        build_job(preference, preference_index, condition, repetition)
        for preference_index, preference in enumerate(PREFERENCES)
        for condition in CONDITIONS
        for repetition in range(REPETITIONS)
    ]
    if args.smoke or args.dry_run:
        jobs = [next(job for job in jobs if job["condition"] == "long")]
    if args.dry_run:
        print(json.dumps(call_vertex(GCloudToken(), endpoint, jobs[0], dry_run=True), indent=2))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.smoke:
        results = json.loads(args.output.read_text())
    else:
        results = {
            "study": "gemini25-preference-reasoning-10rep-balanced-order",
            "model": args.model,
            "platform": "Vertex AI",
            "project": project,
            "location": args.location,
            "created_at": datetime.now(UTC).isoformat(),
            "temperature": 0.7,
            "repetitions": REPETITIONS,
            "thinking_budgets": THINKING_BUDGETS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "conditions": {
                name: {
                    "label": config["label"],
                    "thinking_budget": THINKING_BUDGETS[name],
                    "max_output_tokens": MAX_OUTPUT_TOKENS[name],
                }
                for name, config in CONDITIONS.items()
            },
            "preferences": PREFERENCES,
            "records": [],
        }
    results["thinking_budgets"] = THINKING_BUDGETS
    results["max_output_tokens"] = MAX_OUTPUT_TOKENS
    if args.repair_saved:
        repaired = 0
        for record in results["records"]:
            usage = record.get("raw_response", {}).get("usageMetadata", {})
            record.setdefault("reasoning_tokens", usage.get("thoughtsTokenCount", 0))
            record.setdefault("completion_tokens", usage.get("candidatesTokenCount"))
            record.setdefault("prompt_tokens", usage.get("promptTokenCount"))
            record.setdefault("total_tokens", usage.get("totalTokenCount"))
            displayed_choice, parser_format = parse_choice(record.get("content", ""))
            if displayed_choice:
                record["parser_format"] = parser_format
            if "canonical_choice" in record:
                continue
            if not displayed_choice:
                continue
            add_choice_fields(record, displayed_choice)
            record.pop("error", None)
            repaired += 1
        atomic_write(args.output, results)
        print(f"repaired: {repaired}; complete: {sum('canonical_choice' in row for row in results['records'])}/600")
        return
    existing = {record["key"] for record in results["records"] if "canonical_choice" in record}
    pending = [job for job in jobs if job["key"] not in existing][: 1 if args.smoke else args.batch_size]
    if not pending:
        print(f"complete: {len(existing)}/{len(jobs)} records")
        return

    token_cache = GCloudToken()
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(call_vertex, token_cache, endpoint, job): job for job in pending}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            with lock:
                results["records"] = [row for row in results["records"] if row.get("key") != record["key"]]
                results["records"].append(record)
                atomic_write(args.output, results)
            status = record.get("canonical_choice", f"ERROR {record.get('error')}")
            print(f"{completed}/{len(pending)} {record['key']} -> {status}", flush=True)
    successful = sum("canonical_choice" in record for record in results["records"])
    print(f"saved: {successful}/{600 if not args.smoke else 1} successful records")


if __name__ == "__main__":
    main()
