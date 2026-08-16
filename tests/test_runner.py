import sys

import personas.runner as runner
from personas.definitions import ITEMS
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
