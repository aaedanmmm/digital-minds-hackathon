import argparse
import re
import torch
from personas.definitions import ARMS, ITEMS, PERTURBATIONS, RUNGS
from personas.gcs import sync_down, upload_file
from personas.loader import load_model
from personas.prompts import build_battery_conversation, build_messages, prefill_for
from personas.storage import completed_keys, shard_key, write_record

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.I)

CONDITIONS = {
    "think_off":  {"thinking": False, "max_new_tokens": 128},
    "think_low":  {"thinking": True,  "max_new_tokens": 1400},
    "think_high": {"thinking": True,  "max_new_tokens": 4200},
}


def parse_answer(text: str) -> str | None:
    """Last answer tag outside any thinking block wins."""
    # Strip both closed and unclosed thinking blocks. Unclosed blocks run to EOF
    # because generation can be truncated mid-thought. If the only answer is
    # inside a thinking block (open or closed), return None—the model never
    # committed to an answer outside thinking.
    visible = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S | re.I)
    matches = ANSWER_RE.findall(visible)
    return matches[-1].upper() if matches else None


@torch.no_grad()
def generate_one(model, tokenizer, messages, *, thinking, max_new_tokens,
                 prefill=None) -> dict:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking)
    if prefill:
        text = text + prefill
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    if prefill:
        completion = prefill + completion
    return {
        "completion": completion,
        "answer": parse_answer(completion),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "completion_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
    }


@torch.no_grad()
def run_conversation(model, tokenizer, arm_id, rung, condition, items,
                     perturbations, output, gcs_prefix=None,
                     max_new_tokens_override=None):
    """Issue the battery as one conversation, appending each reply before the
    next user turn so later turns condition on earlier ones. Every turn --
    item or perturbation -- is written as its own record, carrying its
    `position` in the conversation so persistence can be scored by position,
    and an `is_item` flag so perturbation turns (which have no <answer> tag
    and are not part of the scored battery) can be told apart from item turns
    at analysis time.

    The persona is stated exactly once: `build_battery_conversation` builds
    the system turn (plus, at L3/L4, the fabricated self-evidence pairs) only
    once up front, and this function only ever appends user/assistant turns
    after that -- never a second system message.

    Resumability is conversation-granularity, not mid-conversation: this
    (arm, rung, condition) conversation is treated as one atomic unit. If
    every turn it would produce already has a record on disk (checked via
    `completed_keys`, the same mechanism the single-item path uses), the
    whole conversation is skipped; otherwise it is regenerated from position
    0, even for turns that already have a record. A preemption then costs at
    most one in-flight conversation, not the whole Stage B run -- with 7 arms
    x 1 winning rung x 3 conditions that is a bounded ~21 conversations, so
    the granularity is coarse but tolerable.

    Mid-conversation resume (reconstructing history from stored records and
    continuing from the first missing turn) was deliberately not built: it
    would need to reconstruct history byte-for-byte, including the exact
    order of assistant replies and self-evidence turns, or a "resumed"
    conversation would silently condition on a subtly different history than
    the one it started with -- and nothing downstream would reveal that
    divergence. Restarting the whole conversation cannot go wrong that way:
    every regenerated turn conditions on a freshly, identically constructed
    history, so a resumed run is byte-identical to an uninterrupted one.
    """
    planned = build_battery_conversation(arm_id, rung, items, perturbations)
    # The system turn (and, at L3/L4, the self-evidence user/assistant pairs)
    # sits at the front of build_messages(arm_id, rung, items[0]) minus its
    # trailing placeholder item turn; everything build_battery_conversation
    # appended after that point is the ordered sequence of user turns to
    # issue one at a time below.
    context = build_messages(arm_id, rung, items[0])[:-1]
    history = list(context)
    user_turns = planned[len(context):]

    # Precompute every turn's identity (position, is_item, item_id, storage
    # key) up front, without generating anything yet, so the whole
    # conversation's completeness can be decided before any generation call.
    item_ids = iter([i.id for i in items])
    turns = []
    for position, turn in enumerate(user_turns):
        is_item = "<answer>" in turn["content"]
        item_id = next(item_ids) if is_item else f"perturbation{position}"
        key = shard_key(arm_id, rung, f"{condition}|multiturn", item_id)
        turns.append((position, turn, is_item, item_id, key))

    expected_keys = {key for *_, key in turns}
    if expected_keys <= completed_keys(output):
        print(f"skip complete conversation {arm_id}|{rung}|{condition} "
              f"({len(expected_keys)} turns already recorded)", flush=True)
        return

    config = dict(CONDITIONS[condition])
    if max_new_tokens_override is not None:
        config["max_new_tokens"] = max_new_tokens_override
    prefill = prefill_for(arm_id, rung)

    for position, turn, is_item, item_id, key in turns:
        history.append(turn)
        result = generate_one(model, tokenizer, history, prefill=prefill, **config)
        history.append({"role": "assistant", "content": result["completion"]})
        path = write_record(output, {
            "key": key, "arm": arm_id, "rung": rung, "condition": condition,
            "item": item_id, "position": position, "is_item": is_item,
            **result})
        if gcs_prefix:
            # upload_file uploads exactly the one record just written -- see
            # personas/gcs.py. Calling sync_up here instead would re-upload
            # the whole output directory after every turn (O(turns^2) over
            # the battery), the exact defect the single-item runner below
            # was already fixed for.
            upload_file(path, gcs_prefix)
        print(f"done {key} -> {result['answer']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--rungs", nargs="+", default=list(RUNGS))
    parser.add_argument("--conditions", nargs="+", default=["think_off"])
    parser.add_argument("--gcs-prefix", default=None)
    parser.add_argument("--multi-turn", action="store_true",
                        help="Issue the battery as one conversation per "
                             "arm/rung/condition, with perturbation turns "
                             "interleaved (Stage B), instead of one "
                             "independent generation per item (Stage A).")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override max_new_tokens for every condition in "
                             "this run. Leaves the CONDITIONS defaults "
                             "untouched, so a run without this flag stays "
                             "reproducible. Use this to raise think_off's "
                             "cap past the point where a prefilled rung "
                             "(L4) is truncated before it ever emits an "
                             "<answer> tag.")
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.gcs_prefix:
        sync_down(args.gcs_prefix, args.output)
    done = completed_keys(args.output)
    model, tokenizer = load_model()

    for arm_id in args.arms:
        arm = ARMS[arm_id]
        rungs = args.rungs if arm.kind == "persona" else ["L1"]
        for rung in rungs:
            for condition in args.conditions:
                if args.multi_turn:
                    run_conversation(
                        model, tokenizer, arm_id, rung, condition, ITEMS,
                        PERTURBATIONS, args.output,
                        gcs_prefix=args.gcs_prefix,
                        max_new_tokens_override=args.max_new_tokens)
                    continue
                for item in ITEMS:
                    key = shard_key(arm_id, rung, condition, item.id)
                    if key in done:
                        continue
                    config = dict(CONDITIONS[condition])
                    if args.max_new_tokens is not None:
                        config["max_new_tokens"] = args.max_new_tokens
                    result = generate_one(
                        model, tokenizer,
                        build_messages(arm_id, rung, item),
                        prefill=prefill_for(arm_id, rung),
                        **config)
                    path = write_record(args.output, {
                        "key": key, "arm": arm_id, "rung": rung,
                        "condition": condition, "item": item.id,
                        **result})
                    if args.gcs_prefix:
                        # Upload only the record just written (O(1)), not a
                        # full resync of the output directory (O(records)) --
                        # see personas/gcs.py for why that distinction matters.
                        upload_file(path, args.gcs_prefix)
                    print(f"done {key} -> {result['answer']}", flush=True)


if __name__ == "__main__":
    main()
