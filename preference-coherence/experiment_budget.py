#!/usr/bin/env python3
"""Print call, cost, and runtime estimates for larger study designs."""

from __future__ import annotations

import argparse


# Empirical averages from the completed 880-call Vertex run.
USD_PER_CALL = 1.408 / 880
CALLS_PER_SECOND = 5.0


def estimate(sets: int, options: int, complexities: int, orders: int, reps: int) -> dict:
    pairs = options * (options - 1) // 2
    calls = sets * pairs * complexities * orders * reps
    return {
        "sets": sets,
        "options": options,
        "pairs": sets * pairs,
        "calls": calls,
        "estimated_usd": calls * USD_PER_CALL,
        "ideal_minutes_at_observed_rate": calls / CALLS_PER_SECOND / 60,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=int, default=12)
    parser.add_argument("--options", type=int, default=6)
    parser.add_argument("--complexities", type=int, default=4)
    parser.add_argument("--orders", type=int, default=2)
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    result = estimate(args.sets, args.options, args.complexities, args.orders, args.reps)
    print(f"sets: {result['sets']}")
    print(f"options per set: {result['options']}")
    print(f"nested unordered pairs: {result['pairs']}")
    print(f"calls: {result['calls']:,}")
    print(f"estimated cost: US${result['estimated_usd']:.2f}")
    print(f"ideal runtime at observed throughput: {result['ideal_minutes_at_observed_rate']:.1f} minutes")


if __name__ == "__main__":
    main()
