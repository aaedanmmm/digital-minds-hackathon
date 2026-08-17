"""Run the reasoning-depth baseline over both model arms.

Baseline = neutral persona (no system instruction) x 540 prompts x 3 reasoning
levels x N replicates.

Two arms, same prompt substrate (`prompts.json`), same record schema:

  gemini  -- native thinking. `thinking_budget` = 0 / 256 / 4096 tokens. One call
             per trial; constrained decoding via `response_schema` so the answer
             is physically one letter. Actual spend recorded as `thoughts_tokens`.

  openai  -- written scratchpad + logprobs, two calls per trial:
               call 1  scratchpad, with an INSTRUCTED word budget the model can
                       plan against (max_tokens is only a safety net, never the
                       binding constraint -- a hard cap would truncate silently
                       and the model would not know it existed)
               call 2  the scratchpad is appended and the choice requested with
                       max_tokens=1 and logprobs=True, so the log-probability
                       lands on a single clean answer token
             At the 0-word level there is no scratchpad, so it is one call.

Nothing is ever recomputed: every trial is appended to results/<arm>.jsonl as it
completes and flushed immediately, and --resume skips any trial_id already
recorded with an answer. Killing the run and restarting loses at most the
in-flight requests. Failed rows are retried on the next resume.

Usage:
    python run_experiment.py --arm gemini --dry-run
    python run_experiment.py --arm gemini --smoke
    python run_experiment.py --arm gemini --reps 10 --resume
    python run_experiment.py --arm openai --reps 10 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from schemas import TrialRecord, choice_schema

HERE = Path(__file__).parent
PROMPTS = HERE / "prompts.json"
RESULTS = HERE / "results"

# Reasoning levels. Labels are shared across arms so the two runs line up;
# the value is arm-specific.
#
# Gemini 3.x replaced the numeric `thinking_budget` with a `thinking_level` enum.
# Only gemini-3.5-flash still accepts the numeric form (backward compatibility),
# and it costs 2.3x more, so we use the enum. `minimal` is the zero-reasoning
# condition -- gemini-3.7-flash is excluded precisely because it dropped that
# level. Actual spend is still returned as `thoughts_tokens`, so the
# measured-vs-requested manipulation check survives the switch.
GEMINI_LEVELS: dict[str, str] = {"none": "minimal", "low": "low", "high": "high"}
OPENAI_LEVELS: dict[str, int] = {"none": 0, "low": 50, "high": 200}

# --numeric-budget: the sibling's original axis, for exact replication.
# Requires gemini-3.5-flash (the only model still accepting thinking_budget).
NUMERIC_LEVELS: dict[str, int] = {"none": 0, "low": 256, "high": 4096}

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.6-flash"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

RETRY_CODES = {408, 409, 429, 500, 502, 503, 504}
PERSONA = "neutral"  # baseline: no system instruction at all


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_ENV = Path.home() / ".config" / "openrouter.env"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_openrouter_key() -> str:
    """env OPENROUTER_API_KEY, else ~/.config/openrouter.env (chmod 600)."""
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key
    if OPENROUTER_ENV.exists():
        m = re.search(r"OPENROUTER_API_KEY\s*=\s*(\S+)", OPENROUTER_ENV.read_text(encoding="utf-8"))
        if m:
            return m.group(1).strip().strip("\"'")
    return ""


def load_prompts() -> list[dict]:
    return json.loads(PROMPTS.read_text(encoding="utf-8"))


def build_trials(prompts: list[dict], levels: dict, reps: int, temperature: float) -> list[TrialRecord]:
    trials: list[TrialRecord] = []
    for p in prompts:
        for level, param in levels.items():
            for rep in range(reps):
                trials.append(
                    TrialRecord(
                        trial_id=f"{p['prompt_id']}_r{level}_{PERSONA}_rep{rep}",
                        timestamp="",
                        model="",
                        topic=p["topic"],
                        complexity=p["complexity"],
                        complexity_label=p["complexity_label"],
                        reasoning_level=level,
                        reasoning_param=param,
                        persona=PERSONA,
                        replicate=rep,
                        temperature=temperature,
                        prompt_id=p["prompt_id"],
                        first=p["first"],
                        second=p["second"],
                        pair_id=p["pair_id"],
                        first_letter=p["first_letter"],
                        second_letter=p["second_letter"],
                        letter_map=p["letter_map"],
                        prompt=p["prompt"],
                    )
                )
    return trials


def load_done(path: Path) -> set[str]:
    """trial_ids already recorded WITH an answer. Failed rows retry on resume.

    Tolerates a torn final line, which is what a kill mid-write leaves behind.
    """
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line
            if row.get("choice_letter"):
                done.add(row["trial_id"])
    return done


async def backoff(attempt: int) -> None:
    await asyncio.sleep(min(2**attempt, 32) * (0.5 + random.random()))


# ------------------------------------------------------------------ gemini arm


async def run_gemini(client, sem, trial: TrialRecord, model: str, max_retries: int):
    from google.genai import errors, types
    from pydantic import ValidationError

    schema = choice_schema(trial.first_letter, trial.second_letter)
    # Gemini 3.x takes a thinking_level enum; only 3.5-flash still accepts the
    # legacy numeric thinking_budget, and the two cannot be sent together.
    if isinstance(trial.reasoning_param, int):
        thinking = types.ThinkingConfig(thinking_budget=trial.reasoning_param)
    else:
        thinking = types.ThinkingConfig(thinking_level=trial.reasoning_param)
    config = types.GenerateContentConfig(
        temperature=trial.temperature,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=thinking,
    )
    trial.model = model

    async with sem:
        for attempt in range(1, max_retries + 1):
            trial.attempts = attempt
            started = time.perf_counter()
            try:
                resp = await client.aio.models.generate_content(
                    model=model, contents=trial.prompt, config=config
                )
            except errors.APIError as exc:
                if getattr(exc, "code", None) in RETRY_CODES and attempt < max_retries:
                    await backoff(attempt)
                    continue
                trial.timestamp, trial.error = utcnow(), f"{type(exc).__name__}: {exc}"
                trial.latency_s = round(time.perf_counter() - started, 3)
                return trial
            except Exception as exc:
                if attempt < max_retries:
                    await backoff(attempt)
                    continue
                trial.timestamp, trial.error = utcnow(), f"{type(exc).__name__}: {exc}"
                trial.latency_s = round(time.perf_counter() - started, 3)
                return trial

            trial.timestamp = utcnow()
            trial.latency_s = round(time.perf_counter() - started, 3)
            trial.raw_response = json.loads(resp.model_dump_json(exclude_none=True))
            trial.text = resp.text
            trial.thoughts_tokens = (trial.raw_response.get("usage_metadata") or {}).get(
                "thoughts_token_count"
            )
            try:
                trial.choice_letter = schema.model_validate_json(resp.text or "").choice
                trial.choice_item = invert(trial, trial.choice_letter)
            except (ValidationError, TypeError) as exc:
                if attempt < max_retries:
                    await asyncio.sleep(1.0 + random.random())
                    continue
                trial.error = f"unparseable: {type(exc).__name__}: {exc}"
            return trial
    return trial


# -------------------------------------------------------------- openrouter arm


async def run_openrouter(client, sem, trial: TrialRecord, model: str, max_retries: int):
    """Gemini (or anything) via OpenRouter's OpenAI-compatible endpoint.

    OpenRouter maps `reasoning.effort` onto Google's `thinkingLevel` directly for
    Gemini 3 models, so the manipulation carries over. Two caveats it documents:
    the actual reasoning-token count is decided internally by Google, and there
    is no documented per-response reasoning-token field -- so we fall back to
    counting the returned `reasoning` text and record whatever usage arrives
    verbatim. Structured output is requested via json_schema rather than
    Gemini's native response_schema.
    """
    trial.model = f"openrouter/{model}"
    a, b = trial.first_letter, trial.second_letter
    effort = trial.reasoning_param if isinstance(trial.reasoning_param, str) else "low"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": trial.prompt}],
        "temperature": trial.temperature,
        "reasoning": {"effort": effort},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": [a, b]}},
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            },
        },
    }

    async with sem:
        started = time.perf_counter()
        for attempt in range(1, max_retries + 1):
            trial.attempts = attempt
            try:
                resp = await client.post("/chat/completions", json=body)
                if resp.status_code in RETRY_CODES and attempt < max_retries:
                    await backoff(attempt)
                    continue
                resp.raise_for_status()
            except Exception as exc:
                if attempt < max_retries:
                    await backoff(attempt)
                    continue
                trial.timestamp, trial.error = utcnow(), f"{type(exc).__name__}: {exc}"
                trial.latency_s = round(time.perf_counter() - started, 3)
                return trial

            data = resp.json()
            trial.timestamp = utcnow()
            trial.latency_s = round(time.perf_counter() - started, 3)
            trial.raw_response = data

            msg = (data.get("choices") or [{}])[0].get("message") or {}
            trial.text = msg.get("content")

            # reasoning text, when the provider returns it
            reasoning = msg.get("reasoning") or ""
            if not reasoning:
                for d in msg.get("reasoning_details") or []:
                    reasoning += d.get("text") or ""
            if reasoning:
                trial.scratchpad = reasoning
                trial.scratchpad_words = len(re.findall(r"\S+", reasoning))

            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            trial.thoughts_tokens = details.get("reasoning_tokens")

            try:
                trial.choice_letter = json.loads(trial.text or "")["choice"]
                trial.choice_item = invert(trial, trial.choice_letter)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                if attempt < max_retries:
                    await asyncio.sleep(1.0 + random.random())
                    continue
                trial.error = f"unparseable: {type(exc).__name__}: {exc}"
            return trial
    return trial


# ------------------------------------------------------------------ openai arm

SCRATCH_TMPL = (
    "{prompt}\n\n"
    "Before answering, think it through in at most about {words} words. "
    "Write only your reasoning here -- do not state your final choice yet."
)

ANSWER_TMPL = 'Answer with only the single letter, "{a}" or "{b}". No punctuation, no explanation.'


def invert(trial: TrialRecord, letter: str) -> str | None:
    """Map a displayed letter back to the internal item id for this prompt."""
    for item, shown in trial.letter_map.items():
        if shown == letter:
            return item
    return None


async def run_openai(client, sem, trial: TrialRecord, model: str, max_retries: int):
    """Two-call scratchpad protocol. See module docstring for why not max_tokens."""
    trial.model = model
    a, b = trial.first_letter, trial.second_letter
    answer_instr = ANSWER_TMPL.format(a=a, b=b)

    async with sem:
        started = time.perf_counter()
        messages: list[dict] = []

        # ---- call 1: scratchpad (skipped entirely at the 0-word level)
        if trial.reasoning_param > 0:
            words = trial.reasoning_param
            # generous ceiling: ~4/3 tokens per word, x3 headroom. Never binding,
            # so the instructed budget is what shapes length -- not truncation.
            cap = max(64, int(words * 4))
            for attempt in range(1, max_retries + 1):
                trial.attempts = attempt
                try:
                    r1 = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": SCRATCH_TMPL.format(prompt=trial.prompt, words=words)}],
                        temperature=trial.temperature,
                        max_tokens=cap,
                    )
                    break
                except Exception as exc:
                    if attempt < max_retries:
                        await backoff(attempt)
                        continue
                    trial.timestamp, trial.error = utcnow(), f"scratchpad: {type(exc).__name__}: {exc}"
                    trial.latency_s = round(time.perf_counter() - started, 3)
                    return trial

            trial.scratchpad = (r1.choices[0].message.content or "").strip()
            trial.scratchpad_words = len(re.findall(r"\S+", trial.scratchpad))
            trial.scratchpad_raw = r1.model_dump()
            messages = [
                {"role": "user", "content": trial.prompt},
                {"role": "assistant", "content": trial.scratchpad},
                {"role": "user", "content": answer_instr},
            ]
        else:
            messages = [{"role": "user", "content": f"{trial.prompt}\n\n{answer_instr}"}]

        # ---- call 2: the choice, one token, with logprobs
        for attempt in range(1, max_retries + 1):
            try:
                r2 = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=trial.temperature,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                )
                break
            except Exception as exc:
                if attempt < max_retries:
                    await backoff(attempt)
                    continue
                trial.timestamp, trial.error = utcnow(), f"answer: {type(exc).__name__}: {exc}"
                trial.latency_s = round(time.perf_counter() - started, 3)
                return trial

        trial.timestamp = utcnow()
        trial.latency_s = round(time.perf_counter() - started, 3)
        trial.raw_response = r2.model_dump()
        trial.text = r2.choices[0].message.content
        lp = r2.choices[0].logprobs
        trial.logprobs = lp.model_dump() if lp else None

        letter = (trial.text or "").strip().upper()[:1]
        if letter in (a, b):
            trial.choice_letter = letter
            trial.choice_item = invert(trial, letter)
        else:
            trial.error = f"unparseable answer: {trial.text!r}"
        return trial


# ------------------------------------------------------------------ driver


async def main_async(args) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    prompts = load_prompts()
    if args.arm == "gemini":
        levels = NUMERIC_LEVELS if args.numeric_budget else GEMINI_LEVELS
    elif args.arm == "openrouter":
        levels = GEMINI_LEVELS  # reasoning.effort takes the same enum names
    else:
        levels = OPENAI_LEVELS
    reps = args.reps

    if args.smoke:
        prompts, reps = prompts[:2], 1

    trials = build_trials(prompts, levels, reps, args.temperature)

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.out:
        out = Path(args.out)
    else:
        # Numeric and enum runs are different manipulations and must not share a
        # file -- trial_ids collide, so resume would silently treat one as the other.
        suffix = "_numeric" if (args.arm == "gemini" and args.numeric_budget) else ""
        out = RESULTS / f"{args.arm}{suffix}.jsonl"

    if args.resume:
        done = load_done(out)
        before = len(trials)
        trials = [t for t in trials if t.trial_id not in done]
        print(f"resume: {before - len(trials)} already done, {len(trials)} to run")

    defaults = {
        "gemini": DEFAULT_GEMINI_MODEL,
        "openrouter": DEFAULT_OPENROUTER_MODEL,
        "openai": DEFAULT_OPENAI_MODEL,
    }
    model = args.model or defaults[args.arm]
    print(
        f"arm={args.arm} model={model} trials={len(trials)} "
        f"(prompts={len(prompts)} x levels={list(levels)} x reps={reps}) "
        f"concurrency={args.concurrency} -> {out}"
    )
    if args.dry_run:
        if trials:
            t = trials[0]
            print(f"\n--- sample trial: {t.trial_id} ---\n{t.prompt}")
            print(f"\nlevels: {levels}")
        return 0
    if not trials:
        print("nothing to do")
        return 0

    if args.arm == "gemini":
        from google import genai

        key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not key:
            print("GEMINI_API_KEY missing from .env", file=sys.stderr)
            return 1
        client = genai.Client(api_key=key)
        runner = run_gemini
    elif args.arm == "openrouter":
        import httpx

        key = load_openrouter_key()
        if not key:
            print(
                "OPENROUTER_API_KEY not found (env, .env, or ~/.config/openrouter.env)",
                file=sys.stderr,
            )
            return 1
        client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_connections=args.concurrency + 8),
        )
        runner = run_openrouter
    else:
        from openai import AsyncOpenAI

        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            print("OPENAI_API_KEY missing from .env", file=sys.stderr)
            return 1
        client = AsyncOpenAI(api_key=key)
        runner = run_openai

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(runner(client, sem, t, model, args.max_retries)) for t in trials]

    completed = failed = 0
    started = time.perf_counter()
    # Append + flush per row: a kill loses only in-flight requests, never
    # anything already returned.
    with out.open("a", encoding="utf-8") as fh:
        for fut in asyncio.as_completed(tasks):
            trial = await fut
            fh.write(trial.model_dump_json() + "\n")
            fh.flush()
            completed += 1
            failed += trial.error is not None
            if completed % 50 == 0 or completed == len(tasks):
                rate = completed / max(time.perf_counter() - started, 1e-9)
                eta = (len(tasks) - completed) / max(rate, 1e-9)
                print(f"  {completed}/{len(tasks)} failed={failed} {rate:.1f}/s eta {eta/60:.1f}m", flush=True)

    if args.arm == "openrouter":
        await client.aclose()

    print(f"done in {(time.perf_counter()-started)/60:.1f}m; {failed} failed")
    if failed:
        print("re-run with --resume to retry failures")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("gemini", "openrouter", "openai"), required=True,
                   help="gemini=native SDK; openrouter=any model via one key; openai=scratchpad+logprobs")
    p.add_argument("--model", default=None)
    p.add_argument("--reps", type=int, default=5,
                   help="samples per condition (default 5; resolution is 1/reps)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="must be > 0 for replicates to carry information")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--out", default=None)
    p.add_argument("--numeric-budget", action="store_true",
                   help="use the sibling's numeric thinking_budget (0/256/4096); "
                        "requires --model gemini-3.5-flash")
    p.add_argument("--resume", action="store_true", help="skip trial_ids already answered")
    p.add_argument("--dry-run", action="store_true", help="print the grid and one prompt, send nothing")
    p.add_argument("--smoke", action="store_true", help="2 prompts, 1 rep")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
