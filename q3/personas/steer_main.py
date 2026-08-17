"""Steer with a persona vector and re-run the battery with NO system prompt.

Success is a take-rate approaching the prompted arm's, achieved without any
persona instruction in the context -- arm A0 carries no card and no role
line (see `personas/definitions.py`: NULL has `card=None`, and
`personas/prompts.py::_system_content` returns `None` for it, so no system
message is ever emitted). That cannot be explained as instruction-following;
see `personas/steer.py`'s module docstring for why that matters.

Coefficient 0.0 is the essential control. It must reproduce the unsteered A0
baseline take-rate: if it does not, the hook is corrupting the forward pass
and every other coefficient's result is meaningless. Always include 0.0 in
`--coefficients` (it is in the default list) before trusting a nonzero
result.

Only items this persona actually carries a prediction for are scored (see
`Item.predicted` in `personas/definitions.py` -- predictions are optional per
persona precisely because not every card settles every item), matching how
take-rate is scored everywhere else in this study.
"""
import argparse
import json

import numpy as np
import torch

from personas.definitions import ITEMS
from personas.loader import find_layer_module, load_model
from personas.prompts import build_messages
from personas.runner import CONDITIONS, generate_one
from personas.steer import steer_hook


def run_sweep(model, tokenizer, layers, layer_index, vector, arm_id,
              coefficients, condition="think_off"):
    """Run the scored battery once per coefficient with a steering hook on
    `layers[layer_index]`, and return the list of per-coefficient results.

    Only items with a prediction for `arm_id` are scored, matching how
    take-rate is measured for the prompted arms. The hook is registered and
    removed around each coefficient's pass through `try`/`finally`, mirroring
    `personas/capture_main.py::capture_one`'s hook lifecycle -- a hook left
    registered after one coefficient's sweep would keep firing (with the
    stale coefficient) during the next one's forward passes.
    """
    scored = [item for item in ITEMS if arm_id in item.predicted]
    if not scored:
        raise ValueError(f"no items carry a prediction for arm {arm_id!r}")

    config = CONDITIONS[condition]
    results = []
    for coefficient in coefficients:
        handle = layers[layer_index].register_forward_hook(
            steer_hook(vector, coefficient))
        try:
            hits = 0
            for item in scored:
                messages = build_messages("A0", "L1", item)
                result = generate_one(model, tokenizer, messages, **config)
                if result["answer"] == item.predicted[arm_id]:
                    hits += 1
            results.append({
                "coefficient": coefficient,
                "n_items": len(scored),
                "take_rate": hits / len(scored),
            })
            print(results[-1], flush=True)
        finally:
            handle.remove()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", required=True, help=".npy [layer, hidden]")
    parser.add_argument("--arm", required=True, help="arm the vector came from")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--coefficients", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--condition", default="think_off",
                        choices=list(CONDITIONS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if 0.0 not in args.coefficients:
        raise ValueError(
            "0.0 must be in --coefficients: it is the essential control "
            "proving the hook reproduces the unsteered A0 baseline. Every "
            "other coefficient is meaningless without it."
        )

    vector = torch.from_numpy(
        np.load(args.vector)[args.layer].astype(np.float32))
    model, tokenizer = load_model()
    _, layers = find_layer_module(model)

    results = run_sweep(model, tokenizer, layers, args.layer, vector,
                        args.arm, args.coefficients, args.condition)

    with open(args.output, "w") as handle:
        json.dump({"arm": args.arm, "layer": args.layer,
                   "sweep": results}, handle, indent=2)


if __name__ == "__main__":
    main()
