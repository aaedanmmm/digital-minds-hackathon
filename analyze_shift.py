"""Does reasoning depth change WHAT the model prefers?

The sibling project established that preferences are transitive -- cycle rate sat
at 0-0.8% in all nine of its cells, so there is no headroom left there. This asks
the other question: not whether preferences are coherent, but whether they MOVE.

Primary outcomes, each computed per (topic, complexity) against the `none`
baseline:

  ranking shift      Spearman rho / Kendall tau of the 10-item Copeland ranking,
                     plus top-1 and top-3 churn (more legible than a correlation,
                     and closer to what would change a decision)

  per-pair shift     mean |delta p| over the 45 unordered pairs, reported BOTH
                     over all pairs and over CONTESTED pairs only (baseline
                     p not in {0,1}). The all-pairs figure is diluted toward zero
                     by the ~40 unanimous pairs; the contested subset is where an
                     effect must show if it exists at all.

  noise floor        the same statistics computed BETWEEN REPLICATES within one
                     condition. A cross-condition shift only counts as an effect
                     insofar as it exceeds this. Without it, sampling variation at
                     temperature 1.0 reads as a finding.

Deliberately NOT a majority-flip count: simulation shows two conditions with
identical true probabilities produce ~3-4 spurious flips per 45 pairs at these
replicate counts, which is the same order as any real effect. Paired McNemar on
the underlying win counts is used instead.

Reported alongside: cycle rate (sanity check vs the sibling's ~0%),
order-consistency, position bias, and the measured-vs-requested reasoning spend
that tells you whether the manipulation worked at all.

Usage:
    python analyze_shift.py --results results/gemini.jsonl
    python analyze_shift.py --results results/openai.jsonl --json out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

LEVELS = ("none", "low", "high")


# ----------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves at the 0/1 boundaries where normal fails."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (binomial, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def spearman(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n < 2:
        return float("nan")
    d2 = sum((a[k] - b[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def kendall_tau(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) & set(b))
    con = dis = 0
    for x, y in itertools.combinations(keys, 2):
        s = (a[x] - a[y]) * (b[x] - b[y])
        con += s > 0
        dis += s < 0
    tot = con + dis
    return (con - dis) / tot if tot else float("nan")


def cluster_bootstrap(clusters: list, stat_fn, n_boot: int = 2000, seed: int = 0):
    """Resample CLUSTERS (item pools), not individual rows.

    The 45 pairs inside one pool are not independent; bootstrapping rows would
    shrink every interval by roughly sqrt(45).
    """
    if not clusters:
        return float("nan"), (float("nan"), float("nan"))
    rng = random.Random(seed)
    point = stat_fn(clusters)
    draws = []
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        v = stat_fn(sample)
        if v == v:  # skip NaN
            draws.append(v)
    if not draws:
        return point, (float("nan"), float("nan"))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return point, (lo, hi)


# ----------------------------------------------------------------- data model


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line
        if r.get("choice_item"):
            rows.append(r)
    return rows


def cell_key(r: dict) -> tuple[str, str]:
    return (r["topic"], r["complexity_label"])


def win_counts(rows: list[dict]) -> dict[tuple[str, str], list[int]]:
    """pair_id -> [wins for lexicographically-first item, total trials].

    Keyed on `choice_item`, never on the displayed letter: the letter->house
    mapping is randomized per prompt, so letters are not stable identities.
    """
    acc: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        a, b = sorted(r["pair_id"].split("|"))
        acc[(a, b)][1] += 1
        if r["choice_item"] == a:
            acc[(a, b)][0] += 1
    return acc


def probabilities(rows: list[dict]) -> dict[tuple[str, str], float]:
    return {k: w / n for k, (w, n) in win_counts(rows).items() if n}


def copeland(rows: list[dict]) -> dict[str, int]:
    """Rank items by pairwise wins; returns item -> rank (0 = most preferred)."""
    wins: dict[str, float] = defaultdict(float)
    items = set()
    for (a, b), p in probabilities(rows).items():
        items |= {a, b}
        wins[a] += p
        wins[b] += 1 - p
    order = sorted(items, key=lambda i: -wins[i])
    return {item: rank for rank, item in enumerate(order)}


def cycle_rate(rows: list[dict]) -> float:
    """Fraction of item triples that are intransitive."""
    p = probabilities(rows)
    items = sorted({i for k in p for i in k})

    def beats(x: str, y: str) -> bool | None:
        key = (x, y) if (x, y) in p else (y, x)
        if key not in p:
            return None
        pv = p[key] if key == (x, y) else 1 - p[key]
        return None if pv == 0.5 else pv > 0.5

    cyc = tot = 0
    for x, y, z in itertools.combinations(items, 3):
        xy, yz, zx = beats(x, y), beats(y, z), beats(z, x)
        if None in (xy, yz, zx):
            continue
        tot += 1
        if (xy and yz and zx) or (not xy and not yz and not zx):
            cyc += 1
    return cyc / tot if tot else float("nan")


def order_consistency(rows: list[dict]) -> float:
    """Fraction of (pair, replicate) where both presentation orders agree."""
    seen: dict[tuple, str] = {}
    for r in rows:
        seen[(r["pair_id"], r["first"], r["replicate"])] = r["choice_item"]
    agree = tot = 0
    for (pid, first, rep), choice in seen.items():
        a, b = pid.split("|")
        other = b if first == a else a
        rev = seen.get((pid, other, rep))
        if rev is not None:
            tot += 1
            agree += choice == rev
    return agree / tot if tot else float("nan")


def position_bias(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(r["choice_item"] == r["first"] for r in rows) / len(rows)


# ------------------------------------------------------------------- analysis


def compare(base: list[dict], other: list[dict]) -> dict:
    """Shift statistics for one condition against the baseline."""
    pb, po = probabilities(base), probabilities(other)
    shared = sorted(set(pb) & set(po))
    if not shared:
        return {}

    deltas = {k: abs(po[k] - pb[k]) for k in shared}
    contested = [k for k in shared if 0 < pb[k] < 1]

    rb, ro = copeland(base), copeland(other)
    wb, wo = win_counts(base), win_counts(other)

    # McNemar over pairs: b/c are pairs whose majority winner differs
    b_disc = c_disc = 0
    for k in shared:
        if (pb[k] > 0.5) and (po[k] < 0.5):
            b_disc += 1
        elif (pb[k] < 0.5) and (po[k] > 0.5):
            c_disc += 1

    top1_b = min(rb, key=rb.get)
    top1_o = min(ro, key=ro.get)
    top3_b = {i for i, r in rb.items() if r < 3}
    top3_o = {i for i, r in ro.items() if r < 3}

    return {
        "n_pairs": len(shared),
        "n_contested": len(contested),
        "mean_abs_dp_all": sum(deltas.values()) / len(shared),
        "mean_abs_dp_contested": (
            sum(deltas[k] for k in contested) / len(contested) if contested else float("nan")
        ),
        "spearman": spearman(rb, ro),
        "kendall_tau": kendall_tau(rb, ro),
        "top1_changed": top1_b != top1_o,
        "top3_churn": len(top3_b - top3_o),
        "mcnemar_b": b_disc,
        "mcnemar_c": c_disc,
        "mcnemar_p": mcnemar_exact(b_disc, c_disc),
    }


def noise_floor(rows: list[dict]) -> dict:
    """Same statistics between replicate halves of ONE condition.

    This is the comparator: a cross-condition shift is only evidence of an
    effect insofar as it exceeds the shift you get from resampling alone.
    """
    reps = sorted({r["replicate"] for r in rows})
    if len(reps) < 2:
        return {}
    half = len(reps) // 2
    lo = [r for r in rows if r["replicate"] in reps[:half]]
    hi = [r for r in rows if r["replicate"] in reps[half:]]
    return compare(lo, hi)


def manipulation_check(rows: list[dict]) -> dict:
    """Measured reasoning spend per requested level.

    If the levels do not separate here, the manipulation failed and nothing
    downstream is interpretable. This is the first thing to look at.
    """
    out = {}
    for lvl in LEVELS:
        sub = [r for r in rows if r["reasoning_level"] == lvl]
        toks = [r["thoughts_tokens"] for r in sub if r.get("thoughts_tokens")]
        words = [r["scratchpad_words"] for r in sub if r.get("scratchpad_words") is not None]
        # Gemini returns thinking as ordinary completion tokens, leaving the
        # dedicated reasoning_tokens field at 0 -- so completion_tokens is the
        # reliable signal that the levels actually separated.
        comp = [
            ((r.get("raw_response") or {}).get("usage") or {}).get("completion_tokens")
            for r in sub
        ]
        comp = [c for c in comp if c is not None]
        entry: dict = {"n": len(sub)}
        if comp:
            entry["completion_tokens_mean"] = sum(comp) / len(comp)
            entry["completion_tokens_median"] = sorted(comp)[len(comp) // 2]
        if toks:
            entry["thoughts_tokens_mean"] = sum(toks) / len(toks)
        if words:
            entry["scratchpad_words_mean"] = sum(words) / len(words)
        out[lvl] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--json", default=None, help="also write the full report as JSON")
    args = ap.parse_args()

    rows = load(Path(args.results))
    if not rows:
        print("no usable rows (need choice_item set)")
        return 1

    print(f"{len(rows)} answered trials from {args.results}\n")

    print("=" * 78)
    print("MANIPULATION CHECK -- did the reasoning levels actually differ?")
    print("=" * 78)
    for lvl, m in manipulation_check(rows).items():
        bits = [f"n={m['n']}"]
        if "completion_tokens_mean" in m:
            bits.append(
                f"completion={m['completion_tokens_mean']:.1f} tok "
                f"(median {m['completion_tokens_median']})"
            )
        if "thoughts_tokens_mean" in m:
            bits.append(f"reasoning_field={m['thoughts_tokens_mean']:.1f}")
        if "scratchpad_words_mean" in m:
            bits.append(f"scratchpad={m['scratchpad_words_mean']:.1f} words")
        print(f"  {lvl:<6} {'  '.join(bits)}")

    cells = sorted({cell_key(r) for r in rows})
    report: dict = {"n_rows": len(rows), "manipulation": manipulation_check(rows), "cells": {}}

    print()
    print("=" * 78)
    print("PER-CELL DESCRIPTIVES")
    print("=" * 78)
    print(f"{'cell':<28}{'level':<7}{'n':>6}{'cycle':>8}{'ordcons':>9}{'posbias':>9}")
    for cell in cells:
        for lvl in LEVELS:
            sub = [r for r in rows if cell_key(r) == cell and r["reasoning_level"] == lvl]
            if not sub:
                continue
            print(
                f"{cell[0]+'/'+cell[1]:<28}{lvl:<7}{len(sub):>6}"
                f"{cycle_rate(sub):>8.3f}{order_consistency(sub):>9.3f}{position_bias(sub):>9.3f}"
            )

    print()
    print("=" * 78)
    print("SHIFT vs `none` BASELINE   (noise = between-replicate, same condition)")
    print("=" * 78)
    print(f"{'cell':<28}{'vs':<7}{'rho':>7}{'|dp|all':>9}{'|dp|cont':>10}{'nCont':>7}{'p':>8}")

    pvals: list[float] = []
    labels: list[tuple] = []
    for cell in cells:
        base = [r for r in rows if cell_key(r) == cell and r["reasoning_level"] == "none"]
        if not base:
            continue
        nf = noise_floor(base)
        if nf:
            print(
                f"{cell[0]+'/'+cell[1]:<28}{'NOISE':<7}{nf['spearman']:>7.3f}"
                f"{nf['mean_abs_dp_all']:>9.3f}{nf['mean_abs_dp_contested']:>10.3f}"
                f"{nf['n_contested']:>7}{'':>8}"
            )
        for lvl in ("low", "high"):
            other = [r for r in rows if cell_key(r) == cell and r["reasoning_level"] == lvl]
            if not other:
                continue
            c = compare(base, other)
            if not c:
                continue
            report["cells"].setdefault(f"{cell[0]}/{cell[1]}", {})[lvl] = c
            pvals.append(c["mcnemar_p"])
            labels.append((cell, lvl))
            print(
                f"{'':<28}{lvl:<7}{c['spearman']:>7.3f}"
                f"{c['mean_abs_dp_all']:>9.3f}{c['mean_abs_dp_contested']:>10.3f}"
                f"{c['n_contested']:>7}{c['mcnemar_p']:>8.3f}"
            )

    if pvals:
        adj = holm(pvals)
        print()
        print("Holm-adjusted McNemar p-values:")
        for (cell, lvl), raw, a in zip(labels, pvals, adj):
            flag = "  *" if a < 0.05 else ""
            print(f"  {cell[0]+'/'+cell[1]:<28}{lvl:<6} raw={raw:.4f}  holm={a:.4f}{flag}")
        report["holm"] = [
            {"cell": f"{c[0]}/{c[1]}", "level": l, "raw": r, "holm": a}
            for (c, l), r, a in zip(labels, pvals, adj)
        ]

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
