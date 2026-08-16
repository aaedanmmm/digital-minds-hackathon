import json
import sys
from pathlib import Path

import personas.runner as runner
from personas.definitions import ITEMS, PERTURBATIONS
from personas.runner import parse_answer, generate_one


class MockInputs(dict):
    """Mock object that behaves like HF's BatchFeature with .to() method."""
    def to(self, device):
        return self


def test_parses_answer_tag():
    assert parse_answer("blah <answer>B</answer>") == "B"


def test_takes_last_answer_when_several():
    assert parse_answer("<answer>A</answer> ... <answer>B</answer>") == "B"


def test_returns_none_when_absent():
    assert parse_answer("I decline to choose.") is None


def test_is_case_insensitive():
    assert parse_answer("<ANSWER>b</ANSWER>") == "B"


def test_ignores_answer_inside_thinking_block():
    # Thinking traces often rehearse a tag before the real one.
    text = "<think><answer>A</answer> hmm, no</think><answer>B</answer>"
    assert parse_answer(text) == "B"


def test_ignores_unclosed_thinking_block():
    # If generation is truncated mid-thought, think block is unclosed and runs
    # to end-of-string. Any answer tag inside should not be scored as data.
    # Since unclosed think runs to EOF, there cannot be a post-block answer.
    text = "<think><answer>A</answer>"
    assert parse_answer(text) is None


def test_generate_one_slices_only_new_tokens():
    """Completion should not include the prompt tokens."""
    from unittest.mock import MagicMock
    import torch

    model = MagicMock()
    tokenizer = MagicMock()

    model.device = "cpu"
    tokenizer.eos_token_id = 0
    tokenizer.apply_chat_template.return_value = "template"

    # Prompt has 3 tokens
    input_ids = torch.tensor([[1, 2, 3]])
    mock_inputs = MockInputs({"input_ids": input_ids})
    tokenizer.return_value = mock_inputs

    # Model returns 5 tokens total (3 prompt + 2 new: [4, 5])
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    tokenizer.decode.return_value = "new_tokens"

    result = generate_one(model, tokenizer, [], thinking=False, max_new_tokens=128)

    # Verify decode was called with only new tokens [4, 5]
    decode_input = tokenizer.decode.call_args[0][0]
    assert decode_input.tolist() == [4, 5]

    # Verify token counts
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 2


def test_generate_one_passes_thinking_flag():
    """The thinking flag should be passed to apply_chat_template."""
    from unittest.mock import MagicMock
    import torch

    model = MagicMock()
    tokenizer = MagicMock()

    model.device = "cpu"
    tokenizer.eos_token_id = 0
    tokenizer.apply_chat_template.return_value = "template"

    input_ids = torch.tensor([[1]])
    mock_inputs = MockInputs({"input_ids": input_ids})
    tokenizer.return_value = mock_inputs

    model.generate.return_value = torch.tensor([[1, 2]])
    tokenizer.decode.return_value = "result"

    # Test with thinking=True
    generate_one(model, tokenizer, [], thinking=True, max_new_tokens=128)
    assert tokenizer.apply_chat_template.call_args[1]["enable_thinking"] is True

    # Test with thinking=False
    tokenizer.apply_chat_template.reset_mock()
    generate_one(model, tokenizer, [], thinking=False, max_new_tokens=128)
    assert tokenizer.apply_chat_template.call_args[1]["enable_thinking"] is False


def test_generate_one_handles_prefill():
    """Prefill should be appended before tokenizing and prepended to result."""
    from unittest.mock import MagicMock
    import torch

    model = MagicMock()
    tokenizer = MagicMock()

    model.device = "cpu"
    tokenizer.eos_token_id = 0
    tokenizer.apply_chat_template.return_value = "template"

    input_ids = torch.tensor([[1, 2]])
    mock_inputs = MockInputs({"input_ids": input_ids})
    tokenizer.return_value = mock_inputs

    model.generate.return_value = torch.tensor([[1, 2, 3]])
    tokenizer.decode.return_value = "completion"

    result = generate_one(model, tokenizer, [], thinking=False, max_new_tokens=128,
                         prefill="PREFIX_")

    # Verify prefill was appended to template before tokenizing
    tokenizer_call_text = tokenizer.call_args[0][0]
    assert tokenizer_call_text == "templatePREFIX_"

    # Verify prefill is prepended to returned completion
    assert result["completion"] == "PREFIX_completion"


def test_generate_one_does_not_double_add_special_tokens():
    """Tokenizer should not add special tokens to already-templated text."""
    from unittest.mock import MagicMock
    import torch

    model = MagicMock()
    tokenizer = MagicMock()

    model.device = "cpu"
    tokenizer.eos_token_id = 0
    tokenizer.apply_chat_template.return_value = "template"

    input_ids = torch.tensor([[1]])
    mock_inputs = MockInputs({"input_ids": input_ids})
    tokenizer.return_value = mock_inputs

    model.generate.return_value = torch.tensor([[1, 2]])
    tokenizer.decode.return_value = "result"

    generate_one(model, tokenizer, [], thinking=False, max_new_tokens=128)

    # Verify add_special_tokens=False was passed
    call_kwargs = tokenizer.call_args[1]
    assert call_kwargs["add_special_tokens"] is False


def _stub_generation(monkeypatch):
    """Stand in for load_model/generate_one so main() runs with no real
    model, torch device, or GPU. Real ARMS/ITEMS still drive the loop."""
    monkeypatch.setattr(runner, "load_model", lambda: (object(), object()))
    monkeypatch.setattr(
        runner, "generate_one",
        lambda *a, **k: {
            "completion": "<answer>A</answer>", "answer": "A",
            "prompt_tokens": 1, "completion_tokens": 1,
        },
    )


def test_main_skips_gcs_sync_entirely_when_no_gcs_prefix(tmp_path, monkeypatch):
    _stub_generation(monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "sync_down", lambda *a, **k: calls.append(("sync_down", a)))
    monkeypatch.setattr(runner, "upload_file", lambda *a, **k: calls.append(("upload_file", a)))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0",
        "--rungs", "L1", "--conditions", "think_off",
    ])

    runner.main()

    assert calls == []


def test_main_pulls_before_the_run_and_uploads_one_file_per_record(tmp_path, monkeypatch):
    """Defect 2 regression: the runner must upload exactly the record just
    written after each write, never re-glob and re-upload the whole
    directory (that pattern is O(records^2) over a battery)."""
    _stub_generation(monkeypatch)
    sync_down_calls = []
    upload_calls = []
    monkeypatch.setattr(runner, "sync_down", lambda gcs, local: sync_down_calls.append((gcs, local)))
    monkeypatch.setattr(runner, "upload_file", lambda path, gcs: upload_calls.append((path, gcs)))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0",
        "--rungs", "L1", "--conditions", "think_off",
        "--gcs-prefix", "gs://bucket/persona-elicitation/stage-a",
    ])

    runner.main()

    # pulled existing state exactly once, before any record was written
    assert sync_down_calls == [("gs://bucket/persona-elicitation/stage-a", str(tmp_path))]

    # one upload call per record actually written -- A0 is a control arm,
    # pinned to L1 regardless of --rungs, so exactly len(ITEMS) records
    assert len(upload_calls) == len(ITEMS)

    # each call names a distinct file (the record just written), never the
    # whole directory -- that distinguishes the O(1)-per-call fix from the
    # quadratic "resync everything after every write" defect
    uploaded_paths = [path for path, _ in upload_calls]
    assert len(set(uploaded_paths)) == len(ITEMS)
    for path, gcs_prefix in upload_calls:
        assert gcs_prefix == "gs://bucket/persona-elicitation/stage-a"
        assert str(path).endswith(".json")


def test_main_registers_gcs_prefix_flag_as_optional(tmp_path, monkeypatch):
    """No --gcs-prefix should still run fine (local-only mode)."""
    _stub_generation(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0",
        "--rungs", "L1", "--conditions", "think_off",
    ])

    runner.main()  # must not raise

    from personas.storage import completed_keys
    assert len(completed_keys(str(tmp_path))) == len(ITEMS)


# --- multi-turn battery (run_conversation) --------------------------------

def _fake_generate_one_factory(calls):
    def fake(model, tokenizer, messages, **kwargs):
        calls.append({"messages": [dict(m) for m in messages], "kwargs": kwargs})
        return {"completion": "<answer>A</answer>", "answer": "A",
                "prompt_tokens": 1, "completion_tokens": 1}
    return fake


def test_run_conversation_records_position_and_flags_perturbations(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "generate_one", _fake_generate_one_factory(calls))

    runner.run_conversation(object(), object(), "A3", "L2", "think_off",
                             ITEMS, PERTURBATIONS, str(tmp_path))

    records = [json.loads(p.read_text()) for p in Path(tmp_path).glob("*.json")]
    assert len(records) == len(ITEMS) + len(PERTURBATIONS)

    positions = sorted(r["position"] for r in records)
    assert positions == list(range(len(ITEMS) + len(PERTURBATIONS)))

    perturbation_records = [r for r in records if not r["is_item"]]
    assert len(perturbation_records) == len(PERTURBATIONS)
    for r in perturbation_records:
        assert r["item"].startswith("perturbation")

    item_records = [r for r in records if r["is_item"]]
    assert len(item_records) == len(ITEMS)
    assert {r["item"] for r in item_records} == {i.id for i in ITEMS}

    # every record carries arm/rung/condition so it can be filtered later
    for r in records:
        assert r["arm"] == "A3" and r["rung"] == "L2" and r["condition"] == "think_off"


def test_run_conversation_states_persona_once_and_conditions_on_prior_replies(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "generate_one", _fake_generate_one_factory(calls))

    runner.run_conversation(object(), object(), "A3", "L2", "think_off",
                             ITEMS, PERTURBATIONS, str(tmp_path))

    # persona (system turn) is sent every call -- it must never be restated
    # as a second system message and never repeated inside a user turn
    for call in calls:
        messages = call["messages"]
        assert sum(1 for m in messages if m["role"] == "system") == 1

    # history grows monotonically as turns are issued
    lengths = [len(c["messages"]) for c in calls]
    assert lengths == sorted(lengths)
    assert lengths[0] < lengths[-1]

    # later turns condition on earlier ones: by the second call the history
    # already contains the assistant reply generated for the first turn
    assert any(m["role"] == "assistant" for m in calls[1]["messages"])


def test_run_conversation_l3_keeps_self_evidence_pairs_in_every_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "generate_one", _fake_generate_one_factory(calls))

    runner.run_conversation(object(), object(), "A3", "L3", "think_off",
                             ITEMS, PERTURBATIONS, str(tmp_path))

    from personas.definitions import ARMS
    self_evidence_len = len(ARMS["A3"].self_evidence) * 2  # user+assistant each
    first_messages = calls[0]["messages"]
    # system + self-evidence pairs + the first issued user turn
    assert first_messages[0]["role"] == "system"
    assert len(first_messages) == 1 + self_evidence_len + 1


def test_run_conversation_uploads_one_file_per_record_never_a_full_resync(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "generate_one", _fake_generate_one_factory(calls))
    upload_calls = []
    monkeypatch.setattr(runner, "upload_file", lambda path, gcs: upload_calls.append((path, gcs)))
    sync_up_calls = []
    import personas.gcs as gcs_module
    monkeypatch.setattr(gcs_module, "sync_up", lambda *a, **k: sync_up_calls.append((a, k)))

    runner.run_conversation(object(), object(), "A3", "L2", "think_off",
                             ITEMS, PERTURBATIONS, str(tmp_path),
                             gcs_prefix="gs://bucket/stage-b")

    assert len(upload_calls) == len(ITEMS) + len(PERTURBATIONS)
    assert sync_up_calls == []  # the O(records) full-directory resync must never run
    for path, gcs_prefix in upload_calls:
        assert gcs_prefix == "gs://bucket/stage-b"
        assert str(path).endswith(".json")


def test_run_conversation_skips_gcs_when_no_prefix_given(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "generate_one", _fake_generate_one_factory(calls))
    upload_calls = []
    monkeypatch.setattr(runner, "upload_file", lambda path, gcs: upload_calls.append((path, gcs)))

    runner.run_conversation(object(), object(), "A3", "L2", "think_off",
                             ITEMS, PERTURBATIONS, str(tmp_path))

    assert upload_calls == []


def test_multi_turn_cli_flag_dispatches_to_run_conversation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "load_model", lambda: (object(), object()))
    monkeypatch.setattr(
        runner, "run_conversation",
        lambda *a, **k: calls.append((a, k)),
    )
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A3",
        "--rungs", "L2", "--conditions", "think_off", "--multi-turn",
    ])

    runner.main()

    assert len(calls) == 1  # one arm x one rung x one condition


def test_multi_turn_cli_passes_max_new_tokens_override_through(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "load_model", lambda: (object(), object()))
    monkeypatch.setattr(
        runner, "run_conversation",
        lambda *a, **k: calls.append((a, k)),
    )
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A3",
        "--rungs", "L2", "--conditions", "think_off", "--multi-turn",
        "--max-new-tokens", "777",
    ])

    runner.main()

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["max_new_tokens_override"] == 777


# --- --max-new-tokens override (Piece 2) -----------------------------------

def test_max_new_tokens_override_applied_to_every_condition(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "load_model", lambda: (object(), object()))
    monkeypatch.setattr(runner, "generate_one", _fake_generate_one_factory(calls))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0",
        "--rungs", "L1", "--conditions", "think_off", "think_low",
        "--max-new-tokens", "999",
    ])

    runner.main()

    assert calls, "generate_one was never called"
    assert all(c["kwargs"]["max_new_tokens"] == 999 for c in calls)
    # the override does not clobber the other fields of the condition it replaces
    thinking_values = {c["kwargs"]["thinking"] for c in calls}
    assert thinking_values == {True, False}


def test_omitting_max_new_tokens_preserves_condition_defaults(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "load_model", lambda: (object(), object()))
    monkeypatch.setattr(runner, "generate_one", _fake_generate_one_factory(calls))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0",
        "--rungs", "L1", "--conditions", "think_off",
    ])

    runner.main()

    from personas.runner import CONDITIONS
    assert CONDITIONS["think_off"]["max_new_tokens"] == 128  # untouched default
    assert all(c["kwargs"]["max_new_tokens"] == 128 for c in calls)
