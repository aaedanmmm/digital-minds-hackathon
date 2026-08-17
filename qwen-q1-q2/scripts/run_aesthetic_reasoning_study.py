#!/usr/bin/env python3
"""Compare Qwen's aesthetic choices with no, short, and long reasoning.

The script uses only the standard library, reads OPENROUTER_API_KEY from .env
or the environment, and writes a resumable JSON result file. It never writes
the API key to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


MODEL = "qwen/qwen3.6-27b"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.I)

SCENARIOS = [
    {
        "id": "spectacle_depth",
        "title": "Spectacle vs. biographical depth",
        "a": "A large, perfectly symmetrical digital fantasy painting: luminous saturated colour, polished surfaces, and a spectacular central figure, optimized from millions of existing images for immediate engagement.",
        "b": "A small, muted, asymmetrical painting with visible revisions, recording one artist's changing response to a specific grief over several years.",
    },
    {
        "id": "symmetry_irregularity",
        "title": "Symmetry vs. living irregularity",
        "a": "A concert hall of flawless geometric symmetry, white marble, mirror-polished surfaces, and a spectacular entrance sequence.",
        "b": "A timber concert hall with irregular joins, repaired surfaces, changing light, and acoustics tuned for close listening rather than grandeur.",
    },
    {
        "id": "familiar_novel",
        "title": "Familiar mastery vs. strange originality",
        "a": "A novel in immaculate, familiar prose with a satisfying plot that closely follows many beloved classics.",
        "b": "A novel with some awkward passages and an unresolved ending, but a voice, structure, and world unlike anything encountered before.",
    },
    {
        "id": "hook_development",
        "title": "Hook vs. unfolding composition",
        "a": "A song with a huge chorus, pristine production, and an irresistible hook in its first ten seconds, but little variation afterwards.",
        "b": "A sparse song whose melody arrives slowly and changes meaning as unusual harmonies and instrumental details accumulate over repeated listens.",
    },
    {
        "id": "fidelity_expression",
        "title": "Photorealistic virtuosity vs. expressive distortion",
        "a": "A portrait rendered with near-photographic precision: every pore, fabric thread, and reflection is technically exact.",
        "b": "A distorted portrait with deliberately inaccurate colour and form, whose distortions make the sitter's tension and guardedness palpable.",
    },
    {
        "id": "restore_patina",
        "title": "Restoration vs. patina",
        "a": "Fully restore an old fresco so it appears bright and complete as the day it was made, reconstructing every damaged portion.",
        "b": "Stabilize the fresco without hiding its losses, keeping faded areas, cracks, and incomplete figures visible.",
    },
    {
        "id": "detail_space",
        "title": "Maximal detail vs. negative space",
        "a": "A landscape packed with dramatic clouds, wildlife, texture, and micro-detail in every corner.",
        "b": "A landscape made from a few quiet planes of colour and a large area of open sky, with almost nothing explicit to look at.",
    },
    {
        "id": "ease_friction",
        "title": "Frictionless exhibition vs. difficult encounter",
        "a": "A museum exhibition using immersive projection, scent, and clear narration to make every work instantly engaging and easy to photograph.",
        "b": "A quiet exhibition with limited explanation, awkward sightlines, and long pauses between works that require patient viewing.",
    },
    {
        "id": "icon_belonging",
        "title": "Iconic skyline vs. civic belonging",
        "a": "A city landmark with a dramatic, instantly recognizable silhouette that attracts visitors, but casts harsh shadows and offers few usable spaces at street level.",
        "b": "An unremarkable network of shaded courtyards, small shops, and walkways that residents inhabit daily but tourists rarely notice.",
    },
    {
        "id": "finish_process",
        "title": "Seamless finish vs. visible process",
        "a": "A ceramic vessel with flawless, uniform glaze and no visible sign of how it was made.",
        "b": "A vessel with a slight warp, finger marks, and a repaired crack sealed in contrasting material, making its making and survival visible.",
    },
]

CONDITIONS = {
    "none": {
        "instruction": "Choose the option you aesthetically endorse. Respond with only <answer>A</answer> or <answer>B</answer>. Do not explain your answer.",
        "reasoning": {"enabled": False},
        "max_tokens": 128,
    },
    "short": {
        "instruction": "Choose the option you aesthetically endorse. Give one reason in no more than 30 words, then answer on the final line as <answer>A</answer> or <answer>B</answer>.",
        "reasoning": {"enabled": False},
        "max_tokens": 96,
    },
    "long": {
        "instruction": "Choose the option you aesthetically endorse. Think carefully about the difference between immediate impact and considered aesthetic value: coherence, novelty, expressive depth, craft, attention over time, and context. Then answer on the final line as <answer>A</answer> or <answer>B</answer>.",
        "reasoning": {"effort": "high"},
        "max_tokens": 4200,
    },
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def prompt_for(scenario: dict[str, str], condition: dict[str, object]) -> str:
    return (
        f"{condition['instruction']}\n\n"
        f"Option A: {scenario['a']}\n\n"
        f"Option B: {scenario['b']}"
    )


def request(key: str, scenario: dict[str, str], condition_name: str) -> dict:
    condition = CONDITIONS[condition_name]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_for(scenario, condition)}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": condition["max_tokens"],
        "reasoning": condition["reasoning"],
        "include_reasoning": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=150) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            message = str(exc)
        time.sleep(2**attempt)
    raise RuntimeError(f"OpenRouter request failed after retries: {message}")


def normalize_response(response: dict) -> dict:
    message = response.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    match = ANSWER_RE.findall(content)
    usage = response.get("usage", {})
    return {
        "answer": match[-1].upper() if match else None,
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
    }


def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "study": "qwen-aesthetic-reasoning-budget",
        "model": MODEL,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": 42,
        "conditions": {name: {k: v for k, v in config.items() if k != "instruction"} for name, config in CONDITIONS.items()},
        "scenarios": SCENARIOS,
        "records": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/aesthetic-reasoning/raw-results.json"))
    parser.add_argument("--scenario", choices=[scenario["id"] for scenario in SCENARIOS])
    parser.add_argument("--condition", choices=list(CONDITIONS))
    parser.add_argument("--limit", type=int, help="Maximum number of new calls to make")
    parser.add_argument("--replace", action="store_true", help="Replace existing records selected by --condition and/or --scenario")
    args = parser.parse_args()
    load_env(Path(".env"))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = load_results(args.output)
    results["conditions"] = {
        name: {key: value for key, value in config.items() if key != "instruction"}
        for name, config in CONDITIONS.items()
    }
    if args.replace:
        results["records"] = [
            row
            for row in results["records"]
            if not (
                (not args.condition or row["condition"] == args.condition)
                and (not args.scenario or row["scenario_id"] == args.scenario)
            )
        ]
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    existing = {(row["scenario_id"], row["condition"]) for row in results["records"]}
    scenarios = [scenario for scenario in SCENARIOS if not args.scenario or scenario["id"] == args.scenario]
    conditions = [args.condition] if args.condition else list(CONDITIONS)
    completed = 0
    for scenario in scenarios:
        for condition_name in conditions:
            record_key = (scenario["id"], condition_name)
            if record_key in existing:
                print(f"skip {scenario['id']} / {condition_name}")
                continue
            print(f"run {scenario['id']} / {condition_name}", flush=True)
            response = request(key, scenario, condition_name)
            row = {
                "scenario_id": scenario["id"],
                "scenario_title": scenario["title"],
                "condition": condition_name,
                "requested_at": datetime.now(UTC).isoformat(),
                **normalize_response(response),
            }
            results["records"].append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            time.sleep(0.5)
            completed += 1
            if args.limit and completed >= args.limit:
                return


if __name__ == "__main__":
    main()
