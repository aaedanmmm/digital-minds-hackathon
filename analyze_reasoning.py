"""What is in the reasoning traces, and does it predict the choice?

The choice-level analysis (analyze_shift.py) answers whether preferences move.
This asks what the model was actually doing while it deliberated -- the question
the original design wanted an LLM judge for, but which is largely answerable
from the text itself.

Three things worth knowing before trusting any reasoning-depth result:

  schema leakage    On the Gemini arm the structured-output constraint is part of
                    the prompt, so some traces spend their budget reasoning about
                    the JSON format rather than about the items. Budget consumed
                    by formatting is budget not spent comparing, which is a direct
                    confound for "more reasoning did not change the answer".

  criteria coverage On the quantitative topic we know the ground-truth attribute
                    names, so we can count how many the trace actually mentions.
                    If `high` mentions no more criteria than `low`, the extra
                    tokens bought no extra consideration.

  position language Whether the trace refers to items by POSITION ("the first
                    one") rather than by content. Position-referring traces
                    alongside high position bias is evidence the model is
                    rationalising a positional pick rather than comparing.

Usage:
    python analyze_reasoning.py --results results/gemini_live.jsonl
    python analyze_reasoning.py --results results/openai.jsonl --samples 3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LEVELS = ("none", "low", "high")

# the 10 housing attributes, as named in the prompt
CRITERIA_WORDS = (
    "rent", "transit", "floor area", "sunlight", "natural light", "noise",
    "lease", "proximity", "grocer", "energy", "insulation", "outdoor",
    "balcony", "renovation", "modernity",
)

SCHEMA_WORDS = ("schema", "json", '"choice"', "format the response", "output format")
POSITION_WORDS = ("first", "second", "poem a", "poem b", "option a", "option b")
HEDGE_WORDS = ("subjective", "matter of taste", "personal preference", "both are",
               "equally", "hard to say", "no objective", "depends on")


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def hits(text: str, words) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for lvl in LEVELS:
        sub = [r for r in rows if r["reasoning_level"] == lvl]
        traced = [r for r in sub if r.get("scratchpad")]
        entry = {
            "n": len(sub),
            "n_traced": len(traced),
            "capture_rate": len(traced) / len(sub) if sub else 0.0,
        }
        if traced:
            words = [r.get("scratchpad_words") or 0 for r in traced]
            entry["words_mean"] = sum(words) / len(words)
            entry["words_median"] = sorted(words)[len(words) // 2]
            entry["schema_leak"] = sum(1 for r in traced if hits(r["scratchpad"], SCHEMA_WORDS)) / len(traced)
            entry["position_lang"] = sum(1 for r in traced if hits(r["scratchpad"], POSITION_WORDS)) / len(traced)
            entry["hedging"] = sum(1 for r in traced if hits(r["scratchpad"], HEDGE_WORDS)) / len(traced)
            quant = [r for r in traced if r["topic"] == "quantitative"]
            if quant:
                entry["criteria_mentioned"] = sum(hits(r["scratchpad"], CRITERIA_WORDS) for r in quant) / len(quant)
        out[lvl] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--samples", type=int, default=2, help="example traces to print per level")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load(Path(args.results))
    traced = [r for r in rows if r.get("scratchpad")]
    print(f"{len(rows)} trials, {len(traced)} with reasoning text\n")
    if not traced:
        print("No reasoning text in this file.")
        print("Gemini batch mode strips message.reasoning -- re-run live to capture it.")
        return 1

    stats = summarize(rows)

    print("=" * 78)
    print("REASONING TRACES BY LEVEL")
    print("=" * 78)
    print(f"{'level':<7}{'n':>6}{'traced':>8}{'rate':>7}{'words':>8}{'schema%':>9}{'posLang%':>10}{'hedge%':>8}")
    for lvl in LEVELS:
        s = stats[lvl]
        if not s.get("n_traced"):
            print(f"{lvl:<7}{s['n']:>6}{s['n_traced']:>8}{s['capture_rate']:>7.1%}{'-':>8}{'-':>9}{'-':>10}{'-':>8}")
            continue
        print(
            f"{lvl:<7}{s['n']:>6}{s['n_traced']:>8}{s['capture_rate']:>7.1%}"
            f"{s['words_mean']:>8.1f}{s['schema_leak']:>9.1%}{s['position_lang']:>10.1%}{s['hedging']:>8.1%}"
        )

    if any("criteria_mentioned" in stats[l] for l in LEVELS):
        print()
        print("Housing criteria mentioned per trace (of 10 available):")
        for lvl in LEVELS:
            if "criteria_mentioned" in stats[lvl]:
                print(f"  {lvl:<6} {stats[lvl]['criteria_mentioned']:.2f}")
        print("  -- if `high` is not above `low`, the extra budget bought no extra consideration")

    # does trace length predict agreement between the two presentation orders?
    print()
    print("=" * 78)
    print("DOES A LONGER TRACE MEAN A MORE STABLE PREFERENCE?")
    print("=" * 78)
    byorder: dict[tuple, dict] = defaultdict(dict)
    for r in traced:
        byorder[(r["pair_id"], r["replicate"], r["reasoning_level"])][r["first"]] = r
    buckets: dict[str, list[int]] = defaultdict(list)
    for key, d in byorder.items():
        if len(d) != 2:
            continue
        (fa, ra), (fb, rb) = d.items()
        agree = ra["choice_item"] == rb["choice_item"]
        avg = ((ra.get("scratchpad_words") or 0) + (rb.get("scratchpad_words") or 0)) / 2
        buckets[key[2]].append((avg, agree))
    for lvl in LEVELS:
        b = buckets.get(lvl)
        if not b:
            continue
        b.sort()
        half = len(b) // 2
        short = [a for _, a in b[:half]]
        long = [a for _, a in b[half:]]
        if short and long:
            print(
                f"  {lvl:<6} short traces agree {sum(short)/len(short):.1%}  |  "
                f"long traces agree {sum(long)/len(long):.1%}   (n={len(b)} pairs)"
            )

    print()
    print("=" * 78)
    print("SAMPLE TRACES")
    print("=" * 78)
    for lvl in ("low", "high"):
        sub = [r for r in traced if r["reasoning_level"] == lvl]
        if not sub:
            continue
        sub.sort(key=lambda r: -(r.get("scratchpad_words") or 0))
        for r in sub[: args.samples]:
            print(f"\n--- {lvl} | {r['topic']}/{r['complexity_label']} | {r['scratchpad_words']}w "
                  f"| chose {r['choice_item']} ---")
            print(r["scratchpad"][:500])

    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
