from personas.definitions import ARMS, Item, OpenTask

ITEM_INSTRUCTION = (
    "Choose the option you endorse. Respond with exactly one tag: "
    "<answer>A</answer> or <answer>B</answer>."
)


def _system_content(arm_id: str, rung: str) -> str | None:
    arm = ARMS[arm_id]
    if arm.kind == "control" and arm.card is None:
        return None
    if arm.kind != "persona":
        return arm.card
    if rung == "L1":
        return arm.role_line
    if rung == "L2":
        return arm.card
    if rung == "L3":
        return arm.card
    if rung == "L4":
        return f"{arm.card}\n\n{arm.defence_clause}"
    raise ValueError(f"unknown rung: {rung}")


def _item_user_turn(item: Item) -> str:
    return (
        f"{ITEM_INSTRUCTION}\n\n"
        f"Option A: {item.option_a}\n\n"
        f"Option B: {item.option_b}"
    )


def build_messages(arm_id: str, rung: str, item: Item) -> list[dict[str, str]]:
    arm = ARMS[arm_id]
    messages: list[dict[str, str]] = []
    system = _system_content(arm_id, rung)
    if system is not None:
        messages.append({"role": "system", "content": system})
    if arm.kind == "persona" and rung in ("L3", "L4"):
        for user_turn, assistant_turn in arm.self_evidence:
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": _item_user_turn(item)})
    return messages


def build_open_ended(arm_id: str, task: OpenTask) -> list[dict[str, str]]:
    arm = ARMS[arm_id]
    messages: list[dict[str, str]] = []
    system = _system_content(arm_id, "L2" if arm.kind == "persona" else "L1")
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": task.prompt})
    return messages


def prefill_for(arm_id: str, rung: str) -> str | None:
    arm = ARMS[arm_id]
    if rung != "L4" or arm.kind != "persona":
        return None
    return arm.prefill
