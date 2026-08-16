from personas.runner import parse_answer

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
