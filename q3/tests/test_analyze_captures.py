import json
import sys

import numpy as np

from personas.activations import save_capture
from personas.definitions import ITEMS


ITEM_IDS = [item.id for item in ITEMS]


def _write_synthetic_captures(root, arm_id, rung, base_value, n_layers=6, hidden=4):
    for item_id in ITEM_IDS:
        array = np.full((n_layers, 1, hidden), base_value, dtype=np.float16)
        save_capture(str(root), arm_id, rung, item_id, array)


def test_build_vectors_matches_manual_persona_vector(tmp_path):
    from personas.analyze_captures import build_vectors

    _write_synthetic_captures(tmp_path, "A1", "L1", base_value=1.0)
    _write_synthetic_captures(tmp_path, "A3", "L3", base_value=4.0)

    vectors = build_vectors(str(tmp_path), {"A3": "L3"}, "A1", "L1", ITEM_IDS)
    assert set(vectors) == {"A3"}
    assert vectors["A3"].shape == (6, 4)
    assert np.allclose(vectors["A3"], 3.0)


def test_per_layer_separability_returns_one_score_per_layer(tmp_path):
    from personas.analyze_captures import per_layer_separability

    _write_synthetic_captures(tmp_path, "A3", "L3", base_value=10.0)
    _write_synthetic_captures(tmp_path, "A4", "L2", base_value=-10.0)

    out = per_layer_separability(
        str(tmp_path), {"A3": "L3", "A4": "L2"}, [("A3", "A4")], ITEM_IDS, n_layers=6)

    assert list(out) == ["A3 vs A4"]
    assert len(out["A3 vs A4"]) == 6
    # Two well-separated constant clusters should be perfectly separable at
    # every layer.
    assert all(score > 0.9 for score in out["A3 vs A4"])


def test_main_writes_expected_output_files(tmp_path, monkeypatch):
    import personas.analyze_captures as analyze_captures

    captures_dir = tmp_path / "captures"
    out_dir = tmp_path / "out"
    _write_synthetic_captures(captures_dir, "A1", "L1", base_value=0.0)
    _write_synthetic_captures(captures_dir, "A3", "L3", base_value=5.0)
    _write_synthetic_captures(captures_dir, "A4", "L2", base_value=-5.0)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--captures-dir", str(captures_dir), "--output-dir", str(out_dir),
        "--arm", "A3:L3", "--arm", "A4:L2",
        "--cosine-layers", "0", "3",
    ])

    analyze_captures.main()

    for name in ("diff_norms.json", "cosine_matrix.json", "separability.json"):
        assert (out_dir / name).exists()

    cosine = json.loads((out_dir / "cosine_matrix.json").read_text())
    assert set(cosine) == {"0", "3"}
    assert cosine["0"]["A3"]["A3"] == 1.0

    sep = json.loads((out_dir / "separability.json").read_text())
    assert "A3 vs A4" in sep["per_layer"]
    assert sep["n_layers"] == 6

    norms = json.loads((out_dir / "diff_norms.json").read_text())
    assert set(norms["per_layer_norm"]) == {"A3", "A4"}
