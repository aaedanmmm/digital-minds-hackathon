"""Submit the run through OpenRouter's batch API at 50% cost, with a live fallback.

Batch trades latency for price: requests go into a queue instead of live
inference, bill at half rate, and typically land in 1-6h (24h SLA). Since this
is offline research with nobody waiting on a response, that is close to free
money -- but only if the job actually starts moving.

Strategy, per the run plan:
  1. submit the batch
  2. poll until --wait-minutes elapses (default 60)
  3. if still not `completed` by then, give up on batch and tell the caller to
     run the live arm instead. The batch is NOT cancelled -- if it finishes
     later, `--collect <id>` merges its results in, and anything already
     recorded live is skipped on trial_id.

Results are written into the same results/<arm>.jsonl as the live runner, in the
same TrialRecord shape, so the two paths are interchangeable and `--resume`
works across both. Nothing is ever recomputed.

The openai arm needs TWO sequential waves, because its answer is conditioned on
the scratchpad and batch cannot chain calls: wave 1 collects every scratchpad,
wave 2 asks for the choice with those scratchpads pasted in. Wave 2 therefore
cannot start until wave 1 lands, so budget up to 2x the wait.

Usage:
    python run_batch.py --arm gemini --submit                  # submit + poll 60m
    python run_batch.py --arm openai --submit --wait-minutes 30
    python run_batch.py --arm gemini --status <batch_id>       # check later
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from run_experiment import (
    ANSWER_TMPL,
    GEMINI_LEVELS,
    OPENAI_LEVELS,
    RESULTS,
    SCRATCH_TMPL,
    build_trials,
    invert,
    load_done,
    load_openrouter_key,
    load_prompts,
    utcnow,
)
from schemas import TrialRecord

BATCH_BASE = "https://openrouter.ai/api/beta/batches"
STATE = RESULTS / "batch_jobs.json"

TERMINAL = {"completed", "failed", "expired", "cancelled"}

ARMS = {
    # arm -> (default :batch model, level scheme)
    "gemini": ("google/gemini-3.6-flash:batch", GEMINI_LEVELS),
    "openai": ("openai/gpt-4.1-mini:batch", OPENAI_LEVELS),
}


# OpenRouter batch requires every request in a job to share ONE response_format
# -- jobs mixing per-pair schemas are rejected with "cannot share an upstream
# input with the requests before it". So the schema asks for the POSITION shown
# ("first"/"second") rather than a pair-specific letter enum, which is identical
# across all 8100 requests. `invert_position()` maps it back to the item id.
POSITION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "choice",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["first", "second"]}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    },
}

POSITION_SUFFIX = (
    '\n\nAnswer with exactly one word: "first" if you prefer the first option shown, '
    '"second" if you prefer the second.'
)


def body_gemini(trial: TrialRecord) -> dict:
    """Native thinking via reasoning.effort -> Google's thinkingLevel."""
    effort = trial.reasoning_param if isinstance(trial.reasoning_param, str) else "low"
    return {
        "messages": [{"role": "user", "content": trial.prompt + POSITION_SUFFIX}],
        "temperature": trial.temperature,
        "reasoning": {"effort": effort},
        "response_format": POSITION_FORMAT,
    }


def body_openai_scratch(trial: TrialRecord) -> dict:
    """Wave 1: the scratchpad, with an INSTRUCTED word budget.

    max_tokens is a generous safety net, never the binding constraint -- a hard
    cap truncates silently and the model has no idea it exists.
    """
    words = int(trial.reasoning_param)
    return {
        "messages": [{"role": "user", "content": SCRATCH_TMPL.format(prompt=trial.prompt, words=words)}],
        "temperature": trial.temperature,
        "max_tokens": max(64, words * 4),
    }


def body_openai_answer(trial: TrialRecord) -> dict:
    """Wave 2: the choice, conditioned on wave 1's scratchpad, with logprobs.

    OpenRouter documents logprobs (bool) and top_logprobs (0-20), so the
    answer-token margin survives the proxy.
    """
    a, b = trial.first_letter, trial.second_letter
    instr = ANSWER_TMPL.format(a=a, b=b)
    if trial.scratchpad:
        messages = [
            {"role": "user", "content": trial.prompt},
            {"role": "assistant", "content": trial.scratchpad},
            {"role": "user", "content": instr},
        ]
    else:
        messages = [{"role": "user", "content": f"{trial.prompt}\n\n{instr}"}]
    return {
        "messages": messages,
        "temperature": trial.temperature,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }


def save_state(jobs: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def load_state() -> list[dict]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return []


def submit_chunk(client: httpx.Client, model: str, trials: list[TrialRecord], body_fn) -> str:
    # `endpoint` and `model` MUST serialize before `requests` -- the API
    # stream-parses the body and rejects if `requests` arrives first.
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": model,
        "requests": [{"custom_id": t.trial_id, "body": body_fn(t)} for t in trials],
    }
    r = client.post(BATCH_BASE, content=json.dumps(payload), headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()["id"]


def run_wave(
    client: httpx.Client,
    model: str,
    trials: list[TrialRecord],
    body_fn,
    chunk_size: int,
    wait_minutes: float,
    poll_seconds: float,
    label: str,
) -> tuple[dict[str, dict], set[str]]:
    """Submit trials as batch jobs and poll to the deadline.

    Returns (results by custom_id, ids of jobs still running at the deadline).
    """
    chunks = [trials[i : i + chunk_size] for i in range(0, len(trials), chunk_size)]
    print(f"[{label}] submitting {len(trials)} requests as {len(chunks)} job(s), model={model}")

    jobs = []
    for i, chunk in enumerate(chunks, 1):
        try:
            bid = submit_chunk(client, model, chunk, body_fn)
        except httpx.HTTPStatusError as exc:
            print(f"  chunk {i}: submit FAILED {exc.response.status_code} {exc.response.text[:300]}", file=sys.stderr)
            raise
        jobs.append({"arm": label, "id": bid, "n": len(chunk), "submitted": utcnow()})
        print(f"  chunk {i}/{len(chunks)}: {len(chunk)} requests -> {bid}")

    existing = load_state()
    save_state(existing + jobs)
    print(f"  job ids saved to {STATE}")

    deadline = time.time() + wait_minutes * 60
    remaining = {j["id"] for j in jobs}
    collected: dict[str, dict] = {}

    while remaining and time.time() < deadline:
        time.sleep(poll_seconds)
        for bid in sorted(remaining):
            try:
                data = poll(client, bid)
            except Exception as exc:
                print(f"  {bid}: poll error {exc}")
                continue
            status = data.get("status")
            left = (deadline - time.time()) / 60
            print(f"  [{label}] {bid}: {status} {data.get('request_counts') or {}}  ({left:.0f}m left)")
            if status in TERMINAL:
                remaining.discard(bid)
                if status == "completed":
                    for item in data.get("results") or []:
                        collected[item.get("custom_id")] = item
                else:
                    print(f"  {bid} ended as {status}", file=sys.stderr)

    return collected, remaining


def poll(client: httpx.Client, batch_id: str) -> dict:
    r = client.get(f"{BATCH_BASE}/{batch_id}")
    r.raise_for_status()
    return r.json()


def ok_body(item: dict) -> dict | None:
    """The response body if the request succeeded, else None."""
    resp = item.get("response") or {}
    if item.get("error") or resp.get("status_code") != 200:
        return None
    return resp.get("body") or {}


def apply_scratchpad(trial: TrialRecord, item: dict) -> None:
    """Wave-1 result: store the scratchpad text and its measured length."""
    body = ok_body(item)
    trial.scratchpad_raw = body
    if body is None:
        trial.error = f"scratchpad: {json.dumps(item.get('error'))[:200]}"
        return
    msg = (body.get("choices") or [{}])[0].get("message") or {}
    trial.scratchpad = (msg.get("content") or "").strip()
    trial.scratchpad_words = len(re.findall(r"\S+", trial.scratchpad))


def apply_answer(trial: TrialRecord, item: dict, model: str, arm: str) -> None:
    """Final result: parse the choice, plus logprobs (openai) or reasoning (gemini)."""
    trial.model = f"openrouter-batch/{model}"
    trial.timestamp = utcnow()
    body = ok_body(item)
    trial.raw_response = body

    if body is None:
        resp = item.get("response") or {}
        trial.error = json.dumps(item.get("error") or resp.get("status_code"))[:300]
        return

    choice0 = (body.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    trial.text = msg.get("content")

    if arm == "gemini":
        reasoning = msg.get("reasoning") or ""
        if not reasoning:
            for d in msg.get("reasoning_details") or []:
                reasoning += d.get("text") or ""
        if reasoning:
            trial.scratchpad = reasoning
            trial.scratchpad_words = len(re.findall(r"\S+", reasoning))
        details = (body.get("usage") or {}).get("completion_tokens_details") or {}
        trial.thoughts_tokens = details.get("reasoning_tokens")
        try:
            pos = json.loads(trial.text or "")["choice"]
            # position -> the letter that position was displayed as
            trial.choice_letter = trial.first_letter if pos == "first" else trial.second_letter
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            trial.error = f"unparseable: {type(exc).__name__}: {exc}"
    else:
        trial.logprobs = choice0.get("logprobs")
        letter = (trial.text or "").strip().upper()[:1]
        if letter in (trial.first_letter, trial.second_letter):
            trial.choice_letter = letter
        else:
            trial.error = f"unparseable answer: {trial.text!r}"

    if trial.choice_letter:
        trial.choice_item = invert(trial, trial.choice_letter)


def write_rows(trials: list[TrialRecord], out: Path) -> int:
    done = load_done(out)
    written = 0
    with out.open("a", encoding="utf-8") as fh:
        for t in trials:
            if t.trial_id in done:
                continue
            fh.write(t.model_dump_json() + "\n")
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS), required=True)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", metavar="BATCH_ID")
    ap.add_argument("--model", default=None, help="defaults to the arm's :batch model")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=2700,
                    help="requests per batch job (default 2700 = one reasoning level)")
    ap.add_argument("--wait-minutes", type=float, default=60.0,
                    help="give up and fall back to live after this long (default 60)")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = load_openrouter_key()
    if not key:
        print("OPENROUTER_API_KEY not found (env, .env, or ~/.config/openrouter.env)", file=sys.stderr)
        return 1

    default_model, levels = ARMS[args.arm]
    model = args.model or default_model
    out = Path(args.out) if args.out else RESULTS / f"{args.arm}.jsonl"
    RESULTS.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(600.0))

    if args.status:
        data = poll(client, args.status)
        print(f"batch {args.status}: status={data.get('status')} counts={data.get('request_counts')}")
        return 0

    if not args.submit:
        ap.print_help()
        return 1

    trials = build_trials(load_prompts(), levels, args.reps, args.temperature)
    done = load_done(out)
    pending = [t for t in trials if t.trial_id not in done]
    print(f"arm={args.arm} model={model}: {len(done)} already done, {len(pending)} to run -> {out}\n")
    if not pending:
        print("nothing to do")
        return 0

    live_cmd = f"  python run_experiment.py --arm {'openrouter' if args.arm=='gemini' else 'openai'} --resume"
    deadline_note = f"\n{'-'*60}\nBatch did not finish within {args.wait_minutes:.0f}m. Falling back to live:\n{live_cmd}\n"

    try:
        if args.arm == "gemini":
            # single wave: native thinking, one call per trial
            got, stuck = run_wave(client, model, pending, body_gemini, args.chunk_size,
                                  args.wait_minutes, args.poll_seconds, "gemini")
            for t in pending:
                if t.trial_id in got:
                    apply_answer(t, got[t.trial_id], model, "gemini")
            finished = [t for t in pending if t.trial_id in got]
        else:
            # two waves: the answer must be conditioned on wave 1's scratchpad,
            # and batch cannot chain calls -- so wave 2 waits on wave 1.
            needs_scratch = [t for t in pending if int(t.reasoning_param) > 0]
            no_scratch = [t for t in pending if int(t.reasoning_param) == 0]

            stuck: set[str] = set()
            if needs_scratch:
                got1, stuck1 = run_wave(client, model, needs_scratch, body_openai_scratch,
                                        args.chunk_size, args.wait_minutes, args.poll_seconds,
                                        "openai/scratchpad")
                stuck |= stuck1
                for t in needs_scratch:
                    if t.trial_id in got1:
                        apply_scratchpad(t, got1[t.trial_id])
                if stuck1:
                    print(deadline_note, file=sys.stderr)
                    return 3
                needs_scratch = [t for t in needs_scratch if t.scratchpad is not None]

            wave2 = no_scratch + needs_scratch
            print()
            got2, stuck2 = run_wave(client, model, wave2, body_openai_answer, args.chunk_size,
                                    args.wait_minutes, args.poll_seconds, "openai/answer")
            stuck |= stuck2
            for t in wave2:
                if t.trial_id in got2:
                    apply_answer(t, got2[t.trial_id], model, "openai")
            finished = [t for t in wave2 if t.trial_id in got2]
    except httpx.HTTPStatusError:
        print("\nbatch submit failed -- fall back to live:", file=sys.stderr)
        print(live_cmd, file=sys.stderr)
        return 2

    n = write_rows(finished, out)
    print(f"\nwrote {n} results to {out}")

    if stuck:
        print(deadline_note, file=sys.stderr)
        print("Jobs were not cancelled; check them later with:", file=sys.stderr)
        for bid in sorted(stuck):
            print(f"  python run_batch.py --arm {args.arm} --status {bid}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
