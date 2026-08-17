"""Side-by-side report across all arms.

Reads whichever result files exist and prints one comparison, so the two
mechanisms (Gemini native thinking vs GPT written scratchpad) can be read
against each other rather than one at a time.

Usage:
    python report.py
    python report.py --out REPORT.md
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from analyze_shift import (
    LEVELS,
    cycle_rate,
    load,
    order_consistency,
    position_bias,
    compare,
    noise_floor,
)

RESULTS = Path(__file__).parent / "results"

ARMS = [
    ("gemini-batch", "gemini.jsonl", "native thinking (batch; reasoning text stripped)"),
    ("gemini-live", "gemini_live.jsonl", "native thinking (live; reasoning text captured)"),
    ("openai", "openai.jsonl", "written scratchpad + logprobs"),
]


def logprob_margin(r: dict) -> float | None:
    """|logP(A) - logP(B)| on the answer token, where both letters are in top-k."""
    lp = r.get("logprobs") or {}
    content = lp.get("content") or []
    if not content:
        return None
    tops = content[0].get("top_logprobs") or []
    d = {t["token"].strip().upper(): t["logprob"] for t in tops if t.get("token")}
    a, b = r["first_letter"], r["second_letter"]
    if a in d and b in d:
        return abs(d[a] - d[b])
    return None


def spend(rows: list[dict], lvl: str) -> str:
    """Reasoning spend for one level.

    Always reports completion tokens, which is the one measure comparable across
    arms -- reporting words where a scratchpad exists and tokens where it does
    not would silently compare 42w against 127t for the same manipulation.
    Scratchpad words are appended as a secondary figure where available.
    """
    sub = [r for r in rows if r["reasoning_level"] == lvl]
    if not sub:
        return "-"
    comp = [
        ((r.get("raw_response") or {}).get("usage") or {}).get("completion_tokens")
        for r in sub
    ]
    comp = [c for c in comp if c is not None]
    words = [r["scratchpad_words"] for r in sub if r.get("scratchpad_words") is not None]
    if not comp:
        return f"{sum(words)/len(words):.0f}w" if words else "-"
    out = f"{sum(comp)/len(comp):.0f}t"
    if words:
        out += f" ({sum(words)/len(words):.0f}w)"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("# Reasoning depth and preference change\n")
        print("Does giving a model more room to deliberate change *what* it prefers?\n")

        loaded: dict[str, list[dict]] = {}
        print("## Data\n")
        print("| arm | rows | mechanism |")
        print("|---|---|---|")
        for name, fn, desc in ARMS:
            p = RESULTS / fn
            if not p.exists():
                continue
            rows = load(p)
            if not rows:
                continue
            loaded[name] = rows
            print(f"| `{name}` | {len(rows):,} | {desc} |")

        if not loaded:
            print("\nNo results yet.")
            return

        print("\n## Manipulation check\n")
        print("Did the reasoning levels actually separate? (`t` = completion tokens, `w` = scratchpad words)\n")
        print("| arm | none | low | high |")
        print("|---|---|---|---|")
        for name, rows in loaded.items():
            print(f"| `{name}` | {spend(rows,'none')} | {spend(rows,'low')} | {spend(rows,'high')} |")

        print("\n## Preference stability\n")
        print("`ordcons` = fraction of pairs where both presentation orders agree.")
        print("`posbias` = P(chose the first-shown option). 0.5 is unbiased.\n")
        print("| arm | cell | none | low | high |")
        print("|---|---|---|---|---|")
        for name, rows in loaded.items():
            cells = sorted({(r["topic"], r["complexity_label"]) for r in rows})
            for cell in cells:
                vals = []
                for lvl in LEVELS:
                    sub = [
                        r for r in rows
                        if (r["topic"], r["complexity_label"]) == cell and r["reasoning_level"] == lvl
                    ]
                    vals.append(f"{order_consistency(sub):.2f}/{position_bias(sub):.2f}" if sub else "-")
                print(f"| `{name}` | {cell[0]}/{cell[1]} | " + " | ".join(vals) + " |")

        print("\n## Preference shift vs the `none` baseline\n")
        print("`rho` = Spearman of the 10-item ranking. `|dp|` = mean absolute change in")
        print("per-pair choice probability, over contested pairs only. `NOISE` is the same")
        print("statistic computed between replicates of one condition -- a shift only counts")
        print("as an effect insofar as it exceeds this.\n")
        print("| arm | cell | vs | rho | \\|dp\\| contested | p |")
        print("|---|---|---|---|---|---|")
        for name, rows in loaded.items():
            cells = sorted({(r["topic"], r["complexity_label"]) for r in rows})
            for cell in cells:
                base = [
                    r for r in rows
                    if (r["topic"], r["complexity_label"]) == cell and r["reasoning_level"] == "none"
                ]
                if not base:
                    continue
                nf = noise_floor(base)
                if nf:
                    print(
                        f"| `{name}` | {cell[0]}/{cell[1]} | NOISE | {nf['spearman']:.3f} "
                        f"| {nf['mean_abs_dp_contested']:.3f} | |"
                    )
                for lvl in ("low", "high"):
                    other = [
                        r for r in rows
                        if (r["topic"], r["complexity_label"]) == cell and r["reasoning_level"] == lvl
                    ]
                    if not other:
                        continue
                    c = compare(base, other)
                    if c:
                        print(
                            f"| | | {lvl} | {c['spearman']:.3f} "
                            f"| {c['mean_abs_dp_contested']:.3f} | {c['mcnemar_p']:.3f} |"
                        )

        # logprob margins, where available
        for name, rows in loaded.items():
            margins: dict[str, list[float]] = {}
            for r in rows:
                m = logprob_margin(r)
                if m is not None:
                    margins.setdefault(r["reasoning_level"], []).append(m)
            if margins:
                print(f"\n## Answer-token decisiveness (`{name}`)\n")
                print("|logP(A) - logP(B)| on the answer token. This moves even when the")
                print("hard choice does not, so it detects effects the choice data cannot.\n")
                print("| level | n | mean margin |")
                print("|---|---|---|")
                for lvl in LEVELS:
                    v = margins.get(lvl)
                    if v:
                        print(f"| {lvl} | {len(v):,} | {sum(v)/len(v):.3f} |")

        print("\n## Cycle rate\n")
        print("Fraction of item triples that are intransitive. The sibling project found")
        print("this pinned near 0, which is why it is a sanity check here and not the outcome.\n")
        print("| arm | none | low | high |")
        print("|---|---|---|---|")
        for name, rows in loaded.items():
            vals = [f"{cycle_rate([r for r in rows if r['reasoning_level']==l]):.3f}" for l in LEVELS]
            print(f"| `{name}` | " + " | ".join(vals) + " |")

    text = buf.getvalue()
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
