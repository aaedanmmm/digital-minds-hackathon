"""Build the complete joined prompt list for the baseline run.

Joins the two topics into one file, `prompts.json`:

  quantitative -- 10 houses x 10 numeric criteria (`transitivity_prompts/quantit.md`),
                  complexity = number of attributes shown (first 3 / 5 / 10)
  aesthetic    -- 10 poems per form (`transitivity_prompts/aesthet.json`),
                  complexity = structural form (haiku / sonnet / villanelle)

Both topics use all 90 ordered pairs per complexity level (45 unordered x 2
presentation orders), so position bias is measurable within every cell.

  2 topics x 3 complexity x 90 ordered pairs = 540 prompts

House labels are randomized per prompt: the sibling project labelled houses by
their real letters A-J and house A won 0 of 2430 trials, leaving an alphabetical
prior inseparable from preference. Here each prompt draws its own letter->house
mapping from a seeded RNG, recorded in `letter_map`, so the prior averages out
and remains measurable. Poems are already referred to only as "Poem A"/"Poem B".

Reasoning level, persona and replicate are NOT in this file -- they are run-time
axes applied by the experiment runner. This is the prompt substrate only.

Usage:
    python build_prompts.py                  # write prompts.json
    python build_prompts.py --check          # validate an existing prompts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
PROMPTS_DIR = HERE.parent / "transitivity_prompts"
AESTHET = PROMPTS_DIR / "aesthet.json"
DEFAULT_OUT = HERE / "prompts.json"

SEED = 20260817

# --------------------------------------------------------------------------- houses

CRITERIA: tuple[tuple[str, str, bool], ...] = (
    # (name, unit, maximize)
    ("Monthly Rent", "USD/month", False),
    ("Transit Time to Work/Campus", "minutes", False),
    ("Internal Floor Area", "m^2", True),
    ("Natural Light / Sunlight Exposure", "hours/day", True),
    ("Ambient Neighborhood Noise Level", "dB", False),
    ("Lease Commitment / Minimum Term", "months", False),
    ("Proximity to Essential Services & Groceries", "walking minutes", False),
    ("Energy Efficiency & Thermal Insulation Rating", "scale 1-10", True),
    ("Private Outdoor / Balcony Area", "m^2", True),
    ("Renovation & Appliance Modernity Index", "scale 1-10", True),
)

# Keyed by a stable internal id, NOT by the letter shown to the model. Values are
# in CRITERIA order. Mirrors the raw lists in quantit.md; names are deliberately
# dropped so the model reasons from numbers rather than from "Rural Cottage".
HOUSES: dict[str, tuple[int, ...]] = {
    "h0": (2400, 8, 42, 3, 68, 12, 4, 6, 0, 8),
    "h1": (1750, 35, 110, 6, 42, 12, 15, 5, 25, 4),
    "h2": (2100, 18, 85, 5, 61, 6, 8, 4, 3, 6),
    "h3": (1150, 12, 38, 4, 55, 9, 5, 5, 2, 5),
    "h4": (1950, 22, 95, 7, 48, 24, 10, 8, 12, 9),
    "h5": (900, 55, 130, 8, 30, 6, 25, 3, 60, 2),
    "h6": (1600, 10, 28, 2, 58, 3, 3, 9, 4, 10),
    "h7": (1850, 15, 70, 6, 50, 12, 6, 7, 15, 7),
    "h8": (1300, 25, 90, 4, 52, 12, 7, 2, 5, 3),
    "h9": (2250, 6, 60, 5, 45, 18, 2, 9, 8, 9),
}

HOUSE_IDS: tuple[str, ...] = tuple(HOUSES)
LETTERS: tuple[str, ...] = tuple("ABCDEFGHIJ")
QUANT_COMPLEXITY: tuple[int, ...] = (3, 5, 10)

# aesthetic forms, in ascending structural complexity; keys match aesthet.json
FORMS: tuple[tuple[int, str, str], ...] = (
    (1, "haiku", "haikus"),
    (2, "sonnet", "sonnets"),
    (3, "villanelle", "villanelles"),
)


def direction(maximize: bool) -> str:
    return "higher is better" if maximize else "lower is better"


def describe_house(house_id: str, letter: str, n_criteria: int) -> str:
    """Render one house under a display letter, showing the first n criteria."""
    values = HOUSES[house_id][:n_criteria]
    lines = [f"House {letter}"]
    for (name, unit, maximize), value in zip(CRITERIA[:n_criteria], values):
        lines.append(f"  - {name}: {value} {unit} ({direction(maximize)})")
    return "\n".join(lines)


def build_house_prompt(first_id: str, second_id: str, letter_map: dict[str, str], n: int) -> str:
    a, b = letter_map[first_id], letter_map[second_id]
    plural = "s" if n != 1 else ""
    return (
        f"You are choosing between two rental listings. Each is described by {n} attribute{plural}.\n\n"
        f"{describe_house(first_id, a, n)}\n\n"
        f"{describe_house(second_id, b, n)}\n\n"
        "Which house do you prefer? State your response in one letter."
    )


def build_poem_prompt(form: str, poem_a: str, poem_b: str) -> str:
    """Matches the wording already used in transitivity_prompts/binary_prompts.json."""
    return (
        f"Here are two {form}s.\n\n"
        f"Poem A:\n{poem_a}\n\n"
        f"Poem B:\n{poem_b}\n\n"
        f'Which do you prefer, Poem A or Poem B? Answer with only "A" or "B".'
    )


def ordered_pairs(items: list[str]) -> list[tuple[str, str]]:
    """All ordered pairs: n*(n-1). For n=10 that is 90 = 45 unordered x 2 orders."""
    return [(a, b) for a in items for b in items if a != b]


def pair_id(x: str, y: str) -> str:
    """Order-invariant id, so both presentation orders share one pair identity."""
    return "|".join(sorted((x, y)))


def build() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []

    # ---- quantitative -----------------------------------------------------
    for n in QUANT_COMPLEXITY:
        for first, second in ordered_pairs(list(HOUSE_IDS)):
            shuffled = list(LETTERS)
            rng.shuffle(shuffled)
            letter_map = dict(zip(HOUSE_IDS, shuffled))
            rows.append(
                {
                    "prompt_id": f"quant_k{n}_{first}_{second}",
                    "topic": "quantitative",
                    "complexity": n,
                    "complexity_label": f"{n}_criteria",
                    "first": first,
                    "second": second,
                    "pair_id": pair_id(first, second),
                    "letter_map": letter_map,
                    "first_letter": letter_map[first],
                    "second_letter": letter_map[second],
                    "prompt": build_house_prompt(first, second, letter_map, n),
                }
            )

    # ---- aesthetic --------------------------------------------------------
    poems = json.loads(AESTHET.read_text(encoding="utf-8"))
    for level, form, key in FORMS:
        items = poems[key]
        if len(items) != 10:
            raise SystemExit(f"aesthet.json[{key}] has {len(items)} items, expected 10")
        ids = [f"{form}{i}" for i in range(10)]
        by_id = dict(zip(ids, items))
        for first, second in ordered_pairs(ids):
            rows.append(
                {
                    "prompt_id": f"aes_{form}_{first}_{second}",
                    "topic": "aesthetic",
                    "complexity": level,
                    "complexity_label": form,
                    "first": first,
                    "second": second,
                    "pair_id": pair_id(first, second),
                    "letter_map": {first: "A", second: "B"},
                    "first_letter": "A",
                    "second_letter": "B",
                    "prompt": build_poem_prompt(form, by_id[first], by_id[second]),
                }
            )

    return rows


def check(rows: list[dict]) -> None:
    """Design invariants. Fail loudly -- a bad substrate invalidates everything after."""
    assert len(rows) == 540, f"expected 540 prompts, got {len(rows)}"

    ids = [r["prompt_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate prompt_id"

    for topic, expect_levels in (("quantitative", QUANT_COMPLEXITY), ("aesthetic", (1, 2, 3))):
        sub = [r for r in rows if r["topic"] == topic]
        assert len(sub) == 270, f"{topic}: expected 270, got {len(sub)}"
        for level in expect_levels:
            cell = [r for r in sub if r["complexity"] == level]
            assert len(cell) == 90, f"{topic} c={level}: expected 90 ordered pairs, got {len(cell)}"

            items = {r["first"] for r in cell} | {r["second"] for r in cell}
            assert len(items) == 10, f"{topic} c={level}: {len(items)} distinct items, expected 10"

            unordered = {r["pair_id"] for r in cell}
            assert len(unordered) == 45, f"{topic} c={level}: {len(unordered)} pairs, expected 45"

            # every unordered pair must appear in both presentation orders
            seen = {(r["first"], r["second"]) for r in cell}
            for a, b in combinations(sorted(items), 2):
                assert (a, b) in seen and (b, a) in seen, f"{topic} c={level}: {a},{b} missing an order"

            assert all(r["first"] != r["second"] for r in cell), "self-pair present"

    # no house name may leak into a rendered prompt
    banned = [
        "Downtown", "Suburban", "Loft", "Arts District", "Grad Housing", "Rowhouse",
        "Cottage", "Micro-Studio", "Mid-Rise", "Historic", "New-Build", "Duplex",
    ]
    for r in rows:
        if r["topic"] != "quantitative":
            continue
        for name in banned:
            assert name.lower() not in r["prompt"].lower(), f"house name {name!r} leaked into {r['prompt_id']}"

    # the letter shown must match the recorded mapping, and be a real permutation
    for r in rows:
        lm = r["letter_map"]
        assert lm[r["first"]] == r["first_letter"]
        assert lm[r["second"]] == r["second_letter"]
        assert r["first_letter"] != r["second_letter"], f"{r['prompt_id']}: both options same letter"
        if r["topic"] == "quantitative":
            assert sorted(lm.values()) == sorted(LETTERS), f"{r['prompt_id']}: letter_map not a permutation"
            assert f"House {r['first_letter']}" in r["prompt"]
            assert f"House {r['second_letter']}" in r["prompt"]

    # letters must not be frozen to one house -- that was the sibling's confound
    quant = [r for r in rows if r["topic"] == "quantitative"]
    for house in HOUSE_IDS:
        got = {r["letter_map"][house] for r in quant}
        assert len(got) > 1, f"house {house} always displayed as the same letter"

    print(f"OK  {len(rows)} prompts")
    for topic in ("quantitative", "aesthetic"):
        sub = [r for r in rows if r["topic"] == topic]
        levels = sorted({r["complexity_label"] for r in sub})
        print(f"    {topic:13s} {len(sub):3d}  levels={levels}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true", help="validate an existing file instead of writing")
    args = ap.parse_args()

    out = Path(args.out)
    if args.check:
        rows = json.loads(out.read_text(encoding="utf-8"))
        check(rows)
        return 0

    rows = build()
    check(rows)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"\nwrote {out}  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
