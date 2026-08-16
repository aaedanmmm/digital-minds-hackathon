from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from personas.loader import find_layer_module, load_model, _is_transient_load_error


class FakeLayer(nn.Module):
    pass


class FakeDecoderLayer(nn.Module):
    pass


class FakeVisionLayer(nn.Module):
    pass


def _tree(depth_path: str, n: int = 64):
    """Build a nested module tree with layers at the given dotted path."""
    root = nn.Module()
    node = root
    parts = depth_path.split(".")
    for part in parts[:-1]:
        child = nn.Module()
        setattr(node, part, child)
        node = child
    setattr(node, parts[-1], nn.ModuleList([FakeLayer() for _ in range(n)]))
    return root


def test_finds_layers_in_multimodal_nesting():
    model = _tree("model.language_model.layers")
    path, layers = find_layer_module(model)
    assert path == "model.language_model.layers"
    assert len(layers) == 64


def test_finds_layers_in_flat_nesting():
    model = _tree("model.layers")
    path, layers = find_layer_module(model)
    assert path == "model.layers"
    assert len(layers) == 64


def test_prefers_longest_modulelist_when_several_exist():
    model = _tree("model.language_model.layers", n=64)
    model.vision_tower = nn.Module()
    model.vision_tower.layers = nn.ModuleList([FakeLayer() for _ in range(24)])
    path, layers = find_layer_module(model)
    assert len(layers) == 64
    assert "language_model" in path


def test_raises_when_no_layer_list_found():
    with pytest.raises(RuntimeError, match="no decoder layer"):
        find_layer_module(nn.Module())


def test_prefers_decoder_named_layer_even_when_shorter():
    """A vision tower with more blocks than the text decoder has layers
    must not win just because its ModuleList is longer — this is the
    concrete failure mode a checkpoint reorganisation could hit."""
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.layers = nn.ModuleList(
        [FakeDecoderLayer() for _ in range(64)])
    model.vision_tower = nn.Module()
    model.vision_tower.layers = nn.ModuleList(
        [FakeVisionLayer() for _ in range(100)])

    path, layers = find_layer_module(model)
    assert len(layers) == 64
    assert "language_model" in path


def test_raises_on_layer_count_mismatch_against_expected():
    model = _tree("model.language_model.layers", n=64)
    with pytest.raises(RuntimeError, match="mismatch"):
        find_layer_module(model, expected_num_layers=32)


# --- load_model exception branching -----------------------------------
#
# These mock every transformers.Auto* attribute the loader resolves, so no
# model is ever downloaded and no GPU is required. They exercise exactly
# the branch that decides which model object the rest of the study runs
# on: transient resource/IO failures must abort immediately, structural
# (wrong-class) failures must fall through to the next class.

def _patch_transformers(first_cls, second_cls, third_cls):
    """Patch transformers.AutoConfig/AutoTokenizer plus the three AutoModel
    classes load_model tries, in priority order."""
    mock_config_cls = MagicMock()
    mock_config_cls.from_pretrained.return_value = MagicMock()
    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
    return patch.multiple(
        "transformers",
        AutoConfig=mock_config_cls,
        AutoTokenizer=mock_tokenizer_cls,
        AutoModelForImageTextToText=first_cls,
        AutoModelForCausalLM=second_cls,
        AutoModel=third_cls,
    )


@pytest.mark.parametrize("transient_exc", [
    ConnectionError("network blip during weight fetch"),
    MemoryError("out of memory"),
    torch.cuda.OutOfMemoryError("CUDA out of memory"),
])
def test_load_model_reraises_transient_error_without_trying_next_class(transient_exc):
    first_cls = MagicMock()
    first_cls.from_pretrained.side_effect = transient_exc
    second_cls = MagicMock()
    third_cls = MagicMock()

    with _patch_transformers(first_cls, second_cls, third_cls):
        with pytest.raises(RuntimeError, match="transient"):
            load_model("fake/model-id")

    second_cls.from_pretrained.assert_not_called()
    third_cls.from_pretrained.assert_not_called()


def test_load_model_falls_through_on_structural_error():
    first_cls = MagicMock()
    first_cls.from_pretrained.side_effect = ValueError(
        "Unrecognized configuration class for this kind of AutoModel")

    fake_model = MagicMock()
    second_cls = MagicMock()
    second_cls.from_pretrained.return_value = fake_model

    third_cls = MagicMock()  # should never be reached

    with _patch_transformers(first_cls, second_cls, third_cls):
        model, tokenizer = load_model("fake/model-id")

    second_cls.from_pretrained.assert_called_once()
    third_cls.from_pretrained.assert_not_called()
    assert model is fake_model
    assert model.loaded_with_class == "AutoModelForCausalLM"


def test_is_transient_load_error_true_for_resource_and_io_failures():
    assert _is_transient_load_error(ConnectionError("dropped")) is True
    assert _is_transient_load_error(TimeoutError("timed out")) is True
    assert _is_transient_load_error(MemoryError("oom")) is True
    assert _is_transient_load_error(torch.cuda.OutOfMemoryError("cuda oom")) is True
    assert _is_transient_load_error(OSError("disk error")) is True
    # OSError subclasses from a bad model_id/cache path: deliberately
    # treated as transient too (see the comment in loader.py).
    assert _is_transient_load_error(FileNotFoundError("no such file")) is True
    assert _is_transient_load_error(PermissionError("cache dir not writable")) is True


def test_is_transient_load_error_false_for_architecture_errors():
    assert _is_transient_load_error(
        ValueError("Unrecognized configuration class")) is False
    assert _is_transient_load_error(KeyError("qwen3_5")) is False
