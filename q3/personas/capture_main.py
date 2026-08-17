"""Activation-capture entrypoint.

Writes one residual-stream array per (arm, rung, item), covering every
decoder layer (see `personas/activations.py`'s module docstring for why all
64 layers, not a stride-4 subset). `--arms` and `--rungs` are paired
positionally -- the caller names exactly the (arm, rung) combinations to
capture, e.g. `--arms A0 A3 --rungs L1 L2` captures (A0, L1) and (A3, L2),
not the (A0, L1), (A0, L2), (A3, L1), (A3, L2) cross product -- so a job can
be scoped to, say, only the winning rung Stage A found for each arm.

Resumability: before generating, `is_capture_complete` checks whether a
valid `.npy` already exists for this (arm, rung, item); if so, the capture
is skipped. Each capture is written by `save_capture`, which writes to a
temp file and atomically renames it into place, so a preemption mid-write
never leaves a half-written `.npy` that a resumed run would misread as
done. This runs on preemptible A100s (Stage A was preempted for real, 21
minutes into a 30-minute run), so a job that dies mid-battery must be
restartable with the same `--output`/`--gcs-prefix` and pick up only the
missing captures -- exactly the guarantee `personas/storage.py`'s
write_record/completed_keys gives the JSON runner, adapted here for numpy
arrays (see `personas/activations.py` for why the adaptation looks
different).

Hooks are registered and removed once per item, inside `capture_one`, via a
try/finally -- never left registered across items and never removed only
once at the very end of the run. A hook left registered after its forward
pass fires again on every later item's forward pass, silently corrupting
later captures or leaking GPU memory over the run.
"""
import argparse
import os

import torch

from personas.activations import (
    is_capture_complete,
    register_capture_hooks,
    save_capture,
    stack_captures,
)
from personas.definitions import ITEMS
from personas.gcs import sync_down, upload_file
from personas.loader import find_layer_module, load_model
from personas.prompts import build_messages


@torch.no_grad()
def capture_one(model, tokenizer, layers, arm_id, rung, item):
    """Run one forward pass with hooks on every layer of `layers`, and
    return the `[layer, position, hidden]` float16 numpy array.

    Hooks are always removed before this returns -- including when the
    forward pass raises -- so a single bad item can never leave a hook
    registered to corrupt every capture after it.
    """
    messages = build_messages(arm_id, rung, item)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    store, handles = register_capture_hooks(layers)
    try:
        model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return stack_captures(store).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gcs-prefix", default=None)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--rungs", nargs="+", required=True,
                        help="Paired positionally with --arms: the Nth rung "
                             "goes with the Nth arm, not every rung with "
                             "every arm.")
    args = parser.parse_args()

    if len(args.arms) != len(args.rungs):
        raise ValueError(
            f"--arms and --rungs must pair up one to one (Nth rung with "
            f"Nth arm), got {len(args.arms)} arms and {len(args.rungs)} "
            "rungs"
        )

    os.makedirs(args.output, exist_ok=True)
    if args.gcs_prefix:
        # Pulled once, before any capture is attempted, so a resumed worker
        # sees every capture a previous (preempted) worker already pushed --
        # same ordering personas.runner uses for the JSON records.
        sync_down(args.gcs_prefix, args.output)

    model, tokenizer = load_model()

    expected_num_layers = None
    try:
        expected_num_layers = int(model.config.text_config.num_hidden_layers)
    except AttributeError:
        pass  # stub/test models: fall back to no cross-check
    path, layers = find_layer_module(model, expected_num_layers=expected_num_layers)
    print(f"hooking {len(layers)} layers at {path}", flush=True)

    for arm_id, rung in zip(args.arms, args.rungs):
        for item in ITEMS:
            if is_capture_complete(args.output, arm_id, rung, item.id,
                                   expected_layers=len(layers)):
                print(f"skip complete capture {arm_id}|{rung}|{item.id}", flush=True)
                continue
            array = capture_one(model, tokenizer, layers, arm_id, rung, item)
            out_path = save_capture(args.output, arm_id, rung, item.id, array)
            if args.gcs_prefix:
                # Upload only the file just written (O(1)) -- see
                # personas/gcs.py for why re-syncing the whole directory
                # after every capture would be O(n^2) over the run.
                upload_file(str(out_path), args.gcs_prefix)
            print(f"captured {arm_id} {rung} {item.id} {array.shape}", flush=True)


if __name__ == "__main__":
    main()
