"""Run persona-vector extraction, separability, and cosine geometry over a
local directory of captured `.npy` activations, and write the results to
disk as JSON.

This is an analysis entrypoint, not a Vertex job: it reads captures already
pulled to local disk (e.g. via `gcloud storage cp` from
`gs://.../persona-elicitation/captures/`) and does not touch the model or
GCS itself. See `q3/results/persona-vectors/README.md` for what the output
files mean and the float32 caveat.

For every persona arm present in `--arm-rungs`, the persona vector is
computed against `--control-arm` (A1, the length-matched control) captured
at `--control-rung`, using the *same* item order for both sides so
`persona_vector`'s paired mean-difference lines up index-for-index.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from personas.definitions import ITEMS
from personas.vectors import cosine_matrix, load_arm_captures, persona_vector, separability


def build_vectors(captures_dir: str, arm_rungs: dict[str, str],
                  control_arm: str, control_rung: str,
                  item_ids: list[str]) -> dict[str, np.ndarray]:
    """arm_id -> [layer, hidden] persona vector, contrasted against
    control_arm/control_rung, for every arm in arm_rungs."""
    control_acts = load_arm_captures(
        captures_dir, control_arm, control_rung, item_ids)
    vectors = {}
    for arm_id, rung in arm_rungs.items():
        arm_acts = load_arm_captures(captures_dir, arm_id, rung, item_ids)
        vectors[arm_id] = persona_vector(arm_acts, control_acts)
    return vectors


def per_layer_separability(captures_dir: str, arm_rungs: dict[str, str],
                           pairs: list[tuple[str, str]],
                           item_ids: list[str], n_layers: int) -> dict:
    """{"A vs B": [accuracy at layer 0, 1, ..., n_layers - 1]} for every pair
    in `pairs`. Loads each arm's captures once and reuses them across every
    layer and every pair it appears in."""
    acts = {arm_id: load_arm_captures(captures_dir, arm_id, rung, item_ids)
           for arm_id, rung in arm_rungs.items()}
    out = {}
    for a, b in pairs:
        key = f"{a} vs {b}"
        out[key] = [separability({a: acts[a], b: acts[b]}, layer=layer)
                    for layer in range(n_layers)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", required=True,
                        help="local directory of {arm}_{rung}_{item}.npy files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-arm", default="A1")
    parser.add_argument("--control-rung", default="L1")
    parser.add_argument("--arm", action="append", default=[],
                        metavar="ARM:RUNG",
                        help="e.g. --arm A3:L3 --arm A4:L2 ... one per "
                             "persona arm to include, at the rung it was "
                             "captured at")
    parser.add_argument("--cosine-layers", type=int, nargs="+",
                        default=[0, 15, 31, 47, 63])
    args = parser.parse_args()

    arm_rungs = dict(pair.split(":") for pair in args.arm)
    item_ids = [item.id for item in ITEMS]

    sample = np.load(Path(args.captures_dir) /
                     f"{args.control_arm}_{args.control_rung}_{item_ids[0]}.npy")
    n_layers = sample.shape[0]

    vectors = build_vectors(args.captures_dir, arm_rungs, args.control_arm,
                            args.control_rung, item_ids)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_norms = {
        arm_id: [float(np.linalg.norm(vec[layer])) for layer in range(n_layers)]
        for arm_id, vec in vectors.items()
    }
    with open(out_dir / "diff_norms.json", "w") as f:
        json.dump({"n_layers": n_layers, "control": f"{args.control_arm}_{args.control_rung}",
                   "per_layer_norm": diff_norms}, f, indent=2)

    cosine_by_layer = {
        str(layer): cosine_matrix(vectors, layer) for layer in args.cosine_layers
    }
    with open(out_dir / "cosine_matrix.json", "w") as f:
        json.dump(cosine_by_layer, f, indent=2)

    arm_ids = sorted(vectors)
    pairs = [(arm_ids[i], arm_ids[j])
            for i in range(len(arm_ids)) for j in range(i + 1, len(arm_ids))]
    sep = per_layer_separability(args.captures_dir, arm_rungs, pairs,
                                 item_ids, n_layers)
    with open(out_dir / "separability.json", "w") as f:
        json.dump({"n_layers": n_layers, "per_layer": sep}, f, indent=2)

    print(f"wrote diff_norms.json, cosine_matrix.json, separability.json to {out_dir}")


if __name__ == "__main__":
    main()
