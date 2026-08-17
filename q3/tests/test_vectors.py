import numpy as np
import pytest

from personas.vectors import (
    cosine_matrix,
    load_arm_captures,
    persona_vector,
    separability,
)


# --- persona_vector ----------------------------------------------------

def test_vector_is_mean_difference():
    arm = np.array([[[2.0, 4.0]], [[6.0, 8.0]]])      # [n, layer, hidden]
    control = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    vector = persona_vector(arm, control)
    assert vector.shape == (1, 2)
    assert np.allclose(vector, [[2.0, 3.0]])


def test_vector_rejects_mismatched_layer_hidden_shapes():
    arm = np.zeros((3, 4, 8))
    control = np.zeros((3, 4, 16))
    with pytest.raises(ValueError, match="layer, hidden"):
        persona_vector(arm, control)


def test_vector_works_with_different_sample_counts():
    """arm and control need not have the same n -- Stage A batteries can
    differ in item count between arms."""
    arm = np.full((5, 2, 3), 4.0)
    control = np.full((9, 2, 3), 1.0)
    vector = persona_vector(arm, control)
    assert np.allclose(vector, np.full((2, 3), 3.0))


# --- separability --------------------------------------------------------

def test_separability_is_high_for_separated_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, size=(20, 1, 8)) + 5.0
    b = rng.normal(0, 0.1, size=(20, 1, 8)) - 5.0
    assert separability({"A3": a, "A1": b}, layer=0) > 0.9


def test_separability_is_chance_for_identical_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1.0, size=(30, 1, 8))
    b = rng.normal(0, 1.0, size=(30, 1, 8))
    assert 0.3 < separability({"A3": a, "A1": b}, layer=0) < 0.7


def test_separability_rejects_more_than_two_arms():
    rng = np.random.default_rng(0)
    acts = {k: rng.normal(size=(10, 1, 4)) for k in ("A3", "A4", "A5")}
    with pytest.raises(ValueError, match="exactly two"):
        separability(acts, layer=0)


def test_separability_rejects_a_single_arm():
    rng = np.random.default_rng(0)
    acts = {"A3": rng.normal(size=(10, 1, 4))}
    with pytest.raises(ValueError, match="exactly two"):
        separability(acts, layer=0)


def test_separability_caps_folds_at_smaller_arm_sample_count():
    """12 items per arm (the real capture count) is smaller than the
    default cv=5's usual expectations for some sklearn defaults but still
    comfortably >= 2, so this must not raise -- it must silently cap folds
    at the smaller side's count."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, size=(3, 1, 4)) + 3.0
    b = rng.normal(0, 0.1, size=(12, 1, 4)) - 3.0
    score = separability({"A3": a, "A1": b}, layer=0)
    assert 0.0 <= score <= 1.0


def test_separability_raises_when_too_few_samples_for_any_fold():
    a = np.zeros((1, 1, 4))
    b = np.zeros((5, 1, 4))
    with pytest.raises(ValueError, match="at least 2"):
        separability({"A3": a, "A1": b}, layer=0)


def test_separability_uses_held_out_accuracy_not_train_accuracy():
    """A classifier fit and scored on the very same rows can reach ~1.0 even
    on noise that is not actually separable; cross-validated accuracy on
    held-out folds should not. This guards against a regression to the
    (wrong) fit-and-score-on-everything shortcut."""
    rng = np.random.default_rng(0)
    # High-dimensional, low-sample, pure noise: an unregularised classifier
    # fit and scored on the same rows can trivially separate this by
    # overfitting noise, but no genuine held-out signal exists.
    a = rng.normal(0, 1.0, size=(6, 1, 50))
    b = rng.normal(0, 1.0, size=(6, 1, 50))
    score = separability({"A3": a, "A1": b}, layer=0)
    assert score < 0.9


# --- cosine_matrix ---------------------------------------------------------

def test_cosine_matrix_is_symmetric_with_unit_diagonal():
    vectors = {"A3": np.array([[1.0, 0.0]]), "A4": np.array([[0.0, 1.0]])}
    matrix = cosine_matrix(vectors, layer=0)
    assert abs(matrix["A3"]["A3"] - 1.0) < 1e-6
    assert abs(matrix["A3"]["A4"] - 0.0) < 1e-6
    assert matrix["A3"]["A4"] == matrix["A4"]["A3"]


def test_cosine_matrix_detects_opposite_directions():
    vectors = {"A5": np.array([[1.0, 2.0, 3.0]]), "A6": np.array([[-1.0, -2.0, -3.0]])}
    matrix = cosine_matrix(vectors, layer=0)
    assert matrix["A5"]["A6"] == pytest.approx(-1.0, abs=1e-6)


def test_cosine_matrix_zero_vector_yields_zero_not_nan():
    vectors = {"A3": np.array([[0.0, 0.0]]), "A4": np.array([[1.0, 1.0]])}
    matrix = cosine_matrix(vectors, layer=0)
    assert matrix["A3"]["A4"] == 0.0
    assert matrix["A3"]["A3"] == 0.0  # 0/0 must not surface as nan
    assert not np.isnan(matrix["A3"]["A4"])


def test_cosine_matrix_selects_the_requested_layer():
    vectors = {
        "A3": np.array([[1.0, 0.0], [0.0, 1.0]]),  # layer0=x-axis, layer1=y-axis
        "A4": np.array([[0.0, 1.0], [0.0, 1.0]]),  # layer0=y-axis, layer1=y-axis
    }
    assert cosine_matrix(vectors, layer=0)["A3"]["A4"] == pytest.approx(0.0, abs=1e-6)
    assert cosine_matrix(vectors, layer=1)["A3"]["A4"] == pytest.approx(1.0, abs=1e-6)


# --- float16 overflow: the critical regression guard ----------------------

def test_float32_cast_prevents_overflow():
    """Construct float16 inputs whose raw float16 arithmetic overflows to
    inf (mirroring the real deep-layer captures, whose raw norm is inf in
    float16 even though every stored value is finite), and assert every
    public function here returns a finite result.

    This is the failure mode the brief calls out explicitly: computing a
    norm or dot product directly on the stored float16 arrays overflows at
    exactly the deep layers where the persona signal is strongest, silently
    turning a real effect into nan.
    """
    large = np.full((1, 5120), 50.0, dtype=np.float16)  # matches real layer-63 magnitudes
    # Sanity: confirm this actually overflows in raw float16 first, so the
    # test is proven to exercise the failure mode it claims to guard against.
    with np.errstate(over="ignore"):
        raw_norm = np.linalg.norm(large[0])
    assert np.isinf(raw_norm), "fixture does not reproduce the float16 overflow"

    small = np.full((1, 5120), -30.0, dtype=np.float16)

    vector = persona_vector(
        np.stack([large, large]), np.stack([small, small]))
    assert np.all(np.isfinite(vector))

    score = separability({"A3": np.stack([large, small]),
                          "A1": np.stack([small, large])}, layer=0)
    assert np.isfinite(score)

    matrix = cosine_matrix({"A5": large, "A6": small}, layer=0)
    assert np.isfinite(matrix["A5"]["A6"])
    assert np.isfinite(matrix["A5"]["A5"])


def test_float32_cast_matches_manual_float32_computation():
    """The overflow guard should not just avoid inf -- the float32 result
    must equal what you'd get computing directly in float32, i.e. the cast
    only widens precision and never changes the answer."""
    rng = np.random.default_rng(2)
    arm16 = rng.normal(0, 20, size=(4, 1, 16)).astype(np.float16)
    ctrl16 = rng.normal(0, 20, size=(4, 1, 16)).astype(np.float16)

    vector = persona_vector(arm16, ctrl16)
    expected = (arm16.astype(np.float32).mean(axis=0)
                - ctrl16.astype(np.float32).mean(axis=0))
    assert np.allclose(vector, expected)


# --- load_arm_captures ------------------------------------------------

def test_load_arm_captures_stacks_in_requested_item_order(tmp_path):
    from personas.activations import save_capture

    save_capture(str(tmp_path), "A3", "L3", "alpha",
                np.full((4, 1, 8), 1.0, dtype=np.float16))
    save_capture(str(tmp_path), "A3", "L3", "beta",
                np.full((4, 1, 8), 2.0, dtype=np.float16))

    stacked = load_arm_captures(str(tmp_path), "A3", "L3", ["beta", "alpha"])
    assert stacked.shape == (2, 4, 8)
    assert np.allclose(stacked[0], 2.0)  # beta first, as requested
    assert np.allclose(stacked[1], 1.0)  # alpha second


def test_load_arm_captures_preserves_float16_dtype(tmp_path):
    from personas.activations import save_capture

    save_capture(str(tmp_path), "A0", "L1", "item1",
                np.zeros((4, 1, 8), dtype=np.float16))
    stacked = load_arm_captures(str(tmp_path), "A0", "L1", ["item1"])
    assert stacked.dtype == np.float16
