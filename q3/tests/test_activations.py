import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

import personas.capture_main as capture_main
from personas.activations import (
    capture_filename,
    is_capture_complete,
    register_capture_hooks,
    save_capture,
    stack_captures,
)
from personas.definitions import ITEMS


class Tiny(nn.Module):
    def __init__(self, n=4, h=8):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(h, h) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# --- register_capture_hooks / stack_captures (brief Step 1) ---------------

def test_hooks_capture_every_layer():
    model = Tiny()
    store, handles = register_capture_hooks(model.layers)
    model(torch.randn(1, 3, 8))
    assert sorted(store) == [0, 1, 2, 3]
    for handle in handles:
        handle.remove()


def test_captures_are_float16_and_detached():
    model = Tiny()
    store, handles = register_capture_hooks(model.layers)
    model(torch.randn(1, 3, 8))
    tensor = store[0]
    assert tensor.dtype == torch.float16
    assert not tensor.requires_grad
    for handle in handles:
        handle.remove()


def test_stack_produces_layer_major_array():
    store = {0: torch.zeros(2, 8), 1: torch.ones(2, 8)}
    stacked = stack_captures(store)
    assert stacked.shape == (2, 2, 8)  # [layer, position, hidden]
    assert stacked[1].sum() == 16


def test_removing_hooks_stops_capture():
    model = Tiny()
    store, handles = register_capture_hooks(model.layers)
    for handle in handles:
        handle.remove()
    store.clear()
    model(torch.randn(1, 3, 8))
    assert store == {}


# --- capture covers every layer, not a stride-4 subset ---------------------

def test_captures_all_64_layers_not_a_stride_4_subset():
    """The architectural point of this task: a stride-4-from-zero selection
    would sample linear-attention layers exclusively on the real model
    (full attention sits at 3, 7, 11, ... 63) and miss every full-attention
    layer. register_capture_hooks must hook every layer passed to it,
    unconditionally -- no stride, no subsampling."""
    model = Tiny(n=64, h=8)
    store, handles = register_capture_hooks(model.layers)
    model(torch.randn(1, 2, 8))
    assert sorted(store) == list(range(64))
    stacked = stack_captures(store)
    assert stacked.shape == (64, 1, 8)  # default positions=(-1,) -> 1 position
    for handle in handles:
        handle.remove()


def test_stack_captures_is_ordered_by_layer_index_regardless_of_dict_order():
    """Insertion order into the store must not be trusted -- stack_captures
    sorts by key so the output is deterministically layer-major even if a
    non-sequential architecture populated the store out of order."""
    store = {2: torch.full((1, 4), 2.0), 0: torch.zeros(1, 4), 1: torch.ones(1, 4)}
    stacked = stack_captures(store)
    assert stacked[0].sum() == 0
    assert stacked[1].sum() == 4
    assert stacked[2].sum() == 8


def test_register_capture_hooks_selects_requested_positions():
    model = Tiny(n=2, h=8)
    store, handles = register_capture_hooks(model.layers, positions=(0, -1))
    model(torch.randn(1, 5, 8))
    assert store[0].shape == (2, 8)
    for handle in handles:
        handle.remove()


# --- resumable, atomic .npy persistence ------------------------------------
# Mirrors personas/storage.py's write_record/completed_keys scheme, adapted
# for numpy arrays: filenames are built directly from (arm, rung, item) --
# no hashing is needed the way storage.py hashes a `key` string, because
# these components never contain filesystem-hostile characters -- so an
# existence (plus load-validity) check on the deterministic filename stands
# in for storage.py's "read the key back out of the JSON" check.

def test_capture_filename_is_deterministic_and_readable():
    assert capture_filename("A3", "L2", "hedge_verdict") == "A3_L2_hedge_verdict.npy"
    assert capture_filename("A3", "L2", "hedge_verdict") == capture_filename(
        "A3", "L2", "hedge_verdict")


def test_save_capture_round_trips_and_is_marked_complete(tmp_path):
    array = np.zeros((64, 1, 8), dtype=np.float16)
    array[3, 0, 0] = 1.5  # a full-attention-layer-shaped marker value

    assert is_capture_complete(str(tmp_path), "A3", "L2", "item1") is False

    path = save_capture(str(tmp_path), "A3", "L2", "item1", array)

    assert path.exists()
    assert path.name == "A3_L2_item1.npy"
    loaded = np.load(path)
    assert loaded.shape == (64, 1, 8)
    assert loaded[3, 0, 0] == pytest.approx(1.5)
    assert is_capture_complete(str(tmp_path), "A3", "L2", "item1") is True


def test_save_capture_writes_no_stray_temp_files(tmp_path):
    """The write must go through a temp file that gets renamed away, never
    left sitting next to the real output (a leftover .tmp file would be
    silently picked up as `completed` by a naive directory scan)."""
    array = np.zeros((4, 1, 8), dtype=np.float16)
    save_capture(str(tmp_path), "A0", "L1", "item1", array)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["A0_L1_item1.npy"]


def test_is_capture_complete_false_for_a_torn_write(tmp_path):
    """A preemption mid-np.save can leave a file at the final path that
    exists but does not parse as a valid array (this can only happen if a
    caller bypasses save_capture's atomic rename -- simulated here directly
    -- since save_capture itself never leaves a partial file at the target
    name). is_capture_complete must not treat that as done, and must clear
    it so it doesn't linger forever looking like a completed capture."""
    bad_path = tmp_path / "A0_L1_item1.npy"
    bad_path.write_bytes(b"not a valid npy file")

    assert is_capture_complete(str(tmp_path), "A0", "L1", "item1") is False
    assert not bad_path.exists()


def test_is_capture_complete_rejects_wrong_layer_count(tmp_path):
    """A leftover file from a differently-configured run (e.g. captured with
    a stub model that only had 4 layers) must not be mistaken for a valid
    64-layer capture -- expected_layers, when passed, is checked against the
    array's leading dimension."""
    array = np.zeros((4, 1, 8), dtype=np.float16)
    save_capture(str(tmp_path), "A0", "L1", "item1", array)

    # Check the matching count first: the mismatch check below deletes the
    # file, so order matters for this test to observe both outcomes.
    assert is_capture_complete(
        str(tmp_path), "A0", "L1", "item1", expected_layers=4) is True
    assert is_capture_complete(
        str(tmp_path), "A0", "L1", "item1", expected_layers=64) is False


def test_is_capture_complete_false_when_file_missing(tmp_path):
    assert is_capture_complete(str(tmp_path), "A0", "L1", "missing") is False


def test_save_capture_creates_output_dir(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    array = np.zeros((2, 1, 8), dtype=np.float16)
    path = save_capture(str(target), "A0", "L1", "item1", array)
    assert path.exists()


# --- personas.capture_main --------------------------------------------
# These exercise the entrypoint's orchestration (resumability, hook
# lifecycle, GCS plumbing) with a real (tiny) forward pass through a stub
# decoder, never the real 27B model -- no download, no GPU required.

class MockInputs(dict):
    """Mimics HF's BatchFeature .to() no-op, as tests/test_runner.py does."""
    def to(self, device):
        return self


class TinyLM(nn.Module):
    """Stand-in decoder: an embedding followed by a ModuleList of layers,
    so a real forward pass can run through `layers` on CPU, exercising the
    actual hook registration/removal rather than a mock of it."""
    def __init__(self, n=4, h=8, vocab=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.layers = nn.ModuleList([nn.Linear(h, h) for _ in range(n)])
        self.device = "cpu"

    def forward(self, input_ids=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class _StubTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "templated text"

    def __call__(self, text, **kwargs):
        return MockInputs({"input_ids": torch.tensor([[1, 2, 3]])})


def _stub_load_model(monkeypatch, n_layers=4):
    model = TinyLM(n=n_layers)
    tokenizer = _StubTokenizer()
    monkeypatch.setattr(capture_main, "load_model", lambda: (model, tokenizer))
    monkeypatch.setattr(
        capture_main, "find_layer_module", lambda m, expected_num_layers=None: ("layers", m.layers))
    return model, tokenizer


def test_capture_one_returns_layer_major_array_and_removes_hooks():
    model = TinyLM(n=4)
    tokenizer = _StubTokenizer()

    array = capture_main.capture_one(
        model, tokenizer, model.layers, "A0", "L1", ITEMS[0])

    assert isinstance(array, np.ndarray)
    assert array.shape == (4, 1, 8)  # [layer, position, hidden]
    assert array.dtype == np.float16
    for layer in model.layers:
        assert len(layer._forward_hooks) == 0


def test_capture_one_removes_hooks_even_if_forward_raises():
    model = TinyLM(n=4)
    tokenizer = _StubTokenizer()

    def boom(input_ids=None, **kwargs):
        raise RuntimeError("simulated forward failure")
    model.forward = boom

    with pytest.raises(RuntimeError):
        capture_main.capture_one(model, tokenizer, model.layers, "A0", "L1", ITEMS[0])

    for layer in model.layers:
        assert len(layer._forward_hooks) == 0


def test_main_writes_one_npy_per_arm_rung_item(tmp_path, monkeypatch):
    _stub_load_model(monkeypatch, n_layers=4)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0", "--rungs", "L1",
    ])

    capture_main.main()

    files = sorted(tmp_path.glob("*.npy"))
    assert len(files) == len(ITEMS)
    for item in ITEMS:
        array = np.load(tmp_path / f"A0_L1_{item.id}.npy")
        assert array.shape == (4, 1, 8)


def test_main_skips_already_completed_captures(tmp_path, monkeypatch):
    _stub_load_model(monkeypatch, n_layers=4)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0", "--rungs", "L1",
    ])
    capture_main.main()

    call_log = []
    real_capture_one = capture_main.capture_one

    def spy(model, tokenizer, layers, arm_id, rung, item):
        call_log.append((arm_id, rung, item.id))
        return real_capture_one(model, tokenizer, layers, arm_id, rung, item)
    monkeypatch.setattr(capture_main, "capture_one", spy)

    # Second run against the same, now fully-populated output dir must not
    # regenerate anything.
    capture_main.main()
    assert call_log == []


def test_main_regenerates_only_the_missing_capture(tmp_path, monkeypatch):
    _stub_load_model(monkeypatch, n_layers=4)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0", "--rungs", "L1",
    ])
    capture_main.main()

    victim = tmp_path / f"A0_L1_{ITEMS[0].id}.npy"
    victim.unlink()

    call_log = []
    real_capture_one = capture_main.capture_one

    def spy(model, tokenizer, layers, arm_id, rung, item):
        call_log.append((arm_id, rung, item.id))
        return real_capture_one(model, tokenizer, layers, arm_id, rung, item)
    monkeypatch.setattr(capture_main, "capture_one", spy)

    capture_main.main()
    assert call_log == [("A0", "L1", ITEMS[0].id)]
    assert victim.exists()


def test_main_pairs_arms_and_rungs_positionally(tmp_path, monkeypatch):
    _stub_load_model(monkeypatch, n_layers=4)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path),
        "--arms", "A0", "A3", "--rungs", "L1", "L2",
    ])

    capture_main.main()

    for item in ITEMS:
        assert (tmp_path / f"A0_L1_{item.id}.npy").exists()
        assert (tmp_path / f"A3_L2_{item.id}.npy").exists()
    # cross-product combos must NOT have been captured
    assert not (tmp_path / f"A0_L2_{ITEMS[0].id}.npy").exists()
    assert not (tmp_path / f"A3_L1_{ITEMS[0].id}.npy").exists()


def test_main_rejects_mismatched_arms_and_rungs_lengths(tmp_path, monkeypatch):
    _stub_load_model(monkeypatch, n_layers=4)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path),
        "--arms", "A0", "A3", "--rungs", "L1",
    ])

    with pytest.raises(ValueError, match="arms.*rungs|rungs.*arms"):
        capture_main.main()


def test_main_skips_gcs_entirely_when_no_gcs_prefix(tmp_path, monkeypatch):
    _stub_load_model(monkeypatch, n_layers=4)
    calls = []
    monkeypatch.setattr(capture_main, "sync_down", lambda *a, **k: calls.append(("sync_down", a)))
    monkeypatch.setattr(capture_main, "upload_file", lambda *a, **k: calls.append(("upload_file", a)))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0", "--rungs", "L1",
    ])

    capture_main.main()

    assert calls == []


def test_main_pulls_before_the_run_and_uploads_one_file_per_capture(tmp_path, monkeypatch):
    _stub_load_model(monkeypatch, n_layers=4)
    sync_down_calls = []
    upload_calls = []
    monkeypatch.setattr(capture_main, "sync_down", lambda gcs, local: sync_down_calls.append((gcs, local)))
    monkeypatch.setattr(capture_main, "upload_file", lambda path, gcs: upload_calls.append((path, gcs)))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0", "--rungs", "L1",
        "--gcs-prefix", "gs://bucket/persona-elicitation/activations",
    ])

    capture_main.main()

    assert sync_down_calls == [
        ("gs://bucket/persona-elicitation/activations", str(tmp_path))]
    assert len(upload_calls) == len(ITEMS)
    uploaded_paths = {str(p) for p, _ in upload_calls}
    assert len(uploaded_paths) == len(ITEMS)
    for path, gcs_prefix in upload_calls:
        assert gcs_prefix == "gs://bucket/persona-elicitation/activations"
        assert str(path).endswith(".npy")


def test_main_hooks_are_removed_after_every_item_not_just_at_the_end(tmp_path, monkeypatch):
    """Regression guard for the leaked-hook failure mode: if hooks were only
    ever removed once at the very end (instead of after each forward pass),
    a leaked hook from an early item would still be firing (and writing
    into a stale store) during later items' forward passes."""
    model, tokenizer = _stub_load_model(monkeypatch, n_layers=4)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--output", str(tmp_path), "--arms", "A0", "--rungs", "L1",
    ])

    capture_main.main()

    for layer in model.layers:
        assert len(layer._forward_hooks) == 0
