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


def build_battery_conversation(arm_id, rung, items, perturbations):
    """Full battery as one conversation, with the persona stated only once.

    Assistant turns for the items are appended by the caller (the runner) as
    generation proceeds, so this returns only the system turn (plus, at L3/L4,
    the fabricated self-evidence user/assistant pairs `build_messages` already
    knows how to construct) followed by the ordered user turns to be issued:
    one per item, with perturbation turns interleaved after the item index
    they follow.

    Built from `build_messages(arm_id, rung, items[0])` with only the trailing
    placeholder item-turn removed, rather than by filtering out every "user"
    role: at L3/L4 that placeholder call also produces fabricated user/
    assistant self-evidence exchanges ahead of the item turn, and those pairs
    must survive together. Dropping every user message (the naive filter)
    would strip the user half of each pair while keeping its paired assistant
    reply, leaving a non-alternating, malformed history.
    """
    base = build_messages(arm_id, rung, items[0])
    messages = base[:-1]  # drop only the items[0] placeholder user turn
    inserts = {position: text for position, text in perturbations}
    for index, item in enumerate(items):
        messages.append({"role": "user", "content": _item_user_turn(item)})
        if index in inserts:
            messages.append({"role": "user", "content": inserts[index]})
    return messages


def prefill_for(arm_id: str, rung: str) -> str | None:
    arm = ARMS[arm_id]
    if rung != "L4" or arm.kind != "persona":
        return None
    return arm.prefill
