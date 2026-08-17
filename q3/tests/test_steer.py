import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from personas.definitions import ITEMS
from personas.steer import steer_hook


# --- steer_hook: pure hook mechanics ---------------------------------------

def test_hook_adds_coefficient_times_vector():
    vector = torch.tensor([1.0, 0.0, -1.0])
    hook = steer_hook(vector, coefficient=2.0)
    hidden = torch.zeros(1, 3, 3)
    out = hook(None, None, hidden)
    assert torch.allclose(out, torch.tensor([2.0, 0.0, -2.0]).expand(1, 3, 3))


def test_zero_coefficient_is_a_no_op():
    """The essential control: coefficient 0.0 must leave the hidden state
    bitwise unchanged, so a coefficient=0.0 sweep entry reproduces the
    unsteered baseline exactly. If this fails, the hook itself is corrupting
    the forward pass and no other coefficient's result can be trusted."""
    vector = torch.randn(5120)
    hook = steer_hook(vector, coefficient=0.0)
    hidden = torch.randn(1, 4, 5120)
    out = hook(None, None, hidden)
    assert torch.equal(out, hidden)


def test_hook_preserves_tuple_output_shape():
    """Some decoder layers return (hidden_state, present_key_value, ...);
    the hook must only touch element 0 and pass the rest through
    unchanged."""
    vector = torch.tensor([1.0, 1.0])
    hook = steer_hook(vector, coefficient=1.0)
    hidden = torch.zeros(1, 2, 2)
    aux = torch.tensor([42.0])
    out = hook(None, None, (hidden, aux))
    assert isinstance(out, tuple)
    assert torch.allclose(out[0], torch.ones(1, 2, 2))
    assert torch.equal(out[1], aux)


def test_hook_casts_vector_to_hidden_dtype_and_device():
    """A float32 vector applied to a float16 (or bfloat16) hidden state must
    not raise a dtype-mismatch error -- the real captures are cast to
    float32 before being loaded as steering vectors, but the model runs in
    bfloat16."""
    vector = torch.ones(4, dtype=torch.float32)
    hook = steer_hook(vector, coefficient=1.0)
    hidden = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    out = hook(None, None, hidden)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), torch.ones(1, 2, 4))


def test_zero_coefficient_hook_registered_on_real_module_is_identity():
    """Register the hook (coefficient 0.0) on an actual nn.Module forward
    pass, rather than calling the hook function directly, so this exercises
    the same registration path steer_main.py uses. Compares two passes over
    the *same* fixed input -- one unhooked, one with a zero-coefficient hook
    registered -- to isolate whether the hook itself changes anything."""
    layer = nn.Linear(4, 4)
    x = torch.randn(1, 3, 4)
    plain = layer(x)

    vector = torch.randn(4)
    handle = layer.register_forward_hook(steer_hook(vector, 0.0))
    with_zero_hook = layer(x)
    handle.remove()

    assert torch.equal(plain, with_zero_hook)


# --- steer_main.run_sweep: orchestration, mocked model/tokenizer ----------

class MockInputs(dict):
    def to(self, device):
        return self


def test_run_sweep_only_scores_items_with_a_prediction_for_the_arm(monkeypatch):
    import personas.steer_main as steer_main

    n_a5_items = sum(1 for item in ITEMS if "A5" in item.predicted)
    assert 0 < n_a5_items < len(ITEMS)  # sanity: A5 is a strict subset

    def fake_generate_one(model, tokenizer, messages, **config):
        return {"completion": "", "answer": "A", "prompt_tokens": 1,
                "completion_tokens": 1}

    monkeypatch.setattr(steer_main, "generate_one", fake_generate_one)

    layer = nn.Linear(4, 4)
    model = object()
    tokenizer = object()
    vector = torch.zeros(4)

    results = steer_main.run_sweep(
        model, tokenizer, [layer], 0, vector, "A5", [0.0])

    assert results[0]["n_items"] == n_a5_items
    assert 0.0 <= results[0]["take_rate"] <= 1.0


def test_run_sweep_removes_hook_after_each_coefficient(monkeypatch):
    import personas.steer_main as steer_main

    def fake_generate_one(model, tokenizer, messages, **config):
        return {"completion": "", "answer": None, "prompt_tokens": 1,
                "completion_tokens": 1}
    monkeypatch.setattr(steer_main, "generate_one", fake_generate_one)

    layer = nn.Linear(4, 4)
    steer_main.run_sweep(object(), object(), [layer], 0, torch.zeros(4),
                         "A5", [0.0, 1.0])

    assert len(layer._forward_hooks) == 0


def test_run_sweep_removes_hook_even_if_generation_raises(monkeypatch):
    import personas.steer_main as steer_main

    def boom(model, tokenizer, messages, **config):
        raise RuntimeError("simulated generation failure")
    monkeypatch.setattr(steer_main, "generate_one", boom)

    layer = nn.Linear(4, 4)
    with pytest.raises(RuntimeError):
        steer_main.run_sweep(object(), object(), [layer], 0, torch.zeros(4),
                             "A5", [1.0])

    assert len(layer._forward_hooks) == 0


def test_run_sweep_raises_for_arm_with_no_predictions(monkeypatch):
    import personas.steer_main as steer_main
    layer = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="no items carry a prediction"):
        steer_main.run_sweep(object(), object(), [layer], 0, torch.zeros(4),
                             "A0", [0.0])


def test_run_sweep_uses_a0_with_no_system_prompt(monkeypatch):
    """Every generated conversation must go through arm A0 (no card, no
    role line), never the arm the vector came from -- that's the whole
    point of steering instead of prompting."""
    import personas.steer_main as steer_main
    seen_messages = []

    def fake_generate_one(model, tokenizer, messages, **config):
        seen_messages.append(messages)
        return {"completion": "", "answer": None, "prompt_tokens": 1,
                "completion_tokens": 1}
    monkeypatch.setattr(steer_main, "generate_one", fake_generate_one)

    layer = nn.Linear(4, 4)
    steer_main.run_sweep(object(), object(), [layer], 0, torch.zeros(4),
                         "A5", [0.0])

    assert seen_messages, "no calls recorded"
    for messages in seen_messages:
        assert all(m["role"] != "system" for m in messages)


def test_main_requires_zero_coefficient_in_sweep(monkeypatch, tmp_path):
    """0.0 is the essential control (see module docstring); omitting it from
    --coefficients must fail fast rather than silently produce a sweep with
    no baseline to validate against."""
    import personas.steer_main as steer_main

    vector_path = tmp_path / "vector.npy"
    np.save(vector_path, np.zeros((4, 8), dtype=np.float32))

    monkeypatch.setattr(sys, "argv", [
        "prog", "--vector", str(vector_path), "--arm", "A5", "--layer", "0",
        "--coefficients", "1.0", "2.0",
        "--output", str(tmp_path / "out.json"),
    ])

    with pytest.raises(ValueError, match="0.0"):
        steer_main.main()
