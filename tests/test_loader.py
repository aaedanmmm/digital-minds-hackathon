import pytest
import torch.nn as nn
from personas.loader import find_layer_module


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
