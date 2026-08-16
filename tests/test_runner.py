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
