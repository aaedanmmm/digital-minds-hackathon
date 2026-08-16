import argparse
import re
import torch
from personas.definitions import ARMS, ITEMS, RUNGS
from personas.gcs import sync_down, upload_file
from personas.loader import load_model
from personas.prompts import build_messages, prefill_for
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--rungs", nargs="+", default=list(RUNGS))
    parser.add_argument("--conditions", nargs="+", default=["think_off"])
    parser.add_argument("--gcs-prefix", default=None)
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
                for item in ITEMS:
                    key = shard_key(arm_id, rung, condition, item.id)
                    if key in done:
                        continue
                    result = generate_one(
                        model, tokenizer,
                        build_messages(arm_id, rung, item),
                        prefill=prefill_for(arm_id, rung),
                        **CONDITIONS[condition])
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
