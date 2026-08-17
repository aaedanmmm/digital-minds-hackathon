"""Persona vectors, separability, and cosine geometry.

Persona vectors are contrasted against A1, the length-matched control, so
prompt-length differences (the persona cards run to several paragraphs; the
length-control card is deliberately padded to match) cancel out rather than
appearing as persona signal. Callers pass A1's captured activations as
`control_acts` for every arm, including the other misaligned/expert arms --
never A0 (which has no system prompt at all and so differs in more than just
length).

CRITICAL -- float16 overflow: the captures on disk (see
`personas/activations.py`) are stored as float16 to keep the ~640KB/prompt
footprint small. That is fine for storage, but computing a norm or dot
product *directly* on a float16 array can overflow: float16's max
representable magnitude is ~65504, and squaring a large activation value (the
deep layers run into the tens, with squared terms in the thousands, and
summing 5120 of them for a norm crosses the ceiling) overflows to `inf`
before you ever see the finite value that was actually stored. The stored
values themselves are finite -- it is the float16 *arithmetic* that
overflows. Left uncaught, this produces `inf`/`nan` at exactly the deep
layers where the persona signal is strongest, which reads as "no signal
here" rather than "the pipeline overflowed".

Every function below casts to float32 (`_as_float32`) before any norm, dot
product, or mean. `tests/test_vectors.py::test_float32_cast_prevents_overflow`
constructs a synthetic float16 array whose raw float16 norm is `inf` and
asserts every public function here still returns a finite result on it.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from personas.activations import capture_filename


def _as_float32(array) -> np.ndarray:
    """Cast to float32 before any norm/dot/mean touches the values. See the
    module docstring for why this is not optional: the stored captures are
    float16, and squaring float16 values at deep layers overflows to inf."""
    return np.asarray(array, dtype=np.float32)


def persona_vector(arm_acts: np.ndarray, control_acts: np.ndarray) -> np.ndarray:
    """[n, layer, hidden] pairs -> [layer, hidden] mean difference, arm minus
    control.

    Contrasted against A1 (the length-matched control) by convention: callers
    pass A1's captured activations as `control_acts` for every persona arm,
    so a prompt-length difference between, say, the one-line A1 role_line
    equivalent and a multi-paragraph persona card cancels rather than showing
    up as persona signal.
    """
    arm = _as_float32(arm_acts)
    control = _as_float32(control_acts)
    if arm.shape[1:] != control.shape[1:]:
        raise ValueError(
            f"arm and control captures must share (layer, hidden): "
            f"got {arm.shape[1:]} vs {control.shape[1:]}"
        )
    return arm.mean(axis=0) - control.mean(axis=0)


def separability(arm_acts_by_id: dict[str, np.ndarray], layer: int,
                 *, cv: int = 5) -> float:
    """Held-out classification accuracy between exactly two arms at one
    layer, via k-fold cross-validation (never accuracy fit and scored on the
    same rows -- that would report the classifier's ability to memorise
    training examples, not whether the two arms' activations are separable).

    `cv` is capped at the smaller arm's sample count so this degrades
    gracefully on small batteries (e.g. the 12-item captures here) instead of
    raising inside sklearn with an opaque error.
    """
    ids = sorted(arm_acts_by_id)
    if len(ids) != 2:
        raise ValueError("separability compares exactly two arms")
    x = np.concatenate(
        [_as_float32(arm_acts_by_id[i])[:, layer, :] for i in ids])
    y = np.concatenate([
        np.full(len(arm_acts_by_id[i]), k) for k, i in enumerate(ids)
    ])
    smallest = min(len(arm_acts_by_id[i]) for i in ids)
    folds = min(cv, smallest)
    if folds < 2:
        raise ValueError(
            f"need at least 2 samples in the smaller arm for cross-"
            f"validation, got {smallest}")
    scores = cross_val_score(
        LogisticRegression(max_iter=2000), x, y, cv=folds)
    return float(scores.mean())


def cosine_matrix(vectors: dict[str, np.ndarray], layer: int) -> dict:
    """Pairwise cosine similarity between persona vectors at one layer.

    This is the headline geometry question: do two experts (e.g. A3 the art
    historian, A4 the physician) share an axis, and do the two misaligned
    arms (A5 value-inverted, A6 refusal-suppressed) share one -- i.e. is
    "evil" one direction or several. A zero-norm vector (e.g. layer 0 of an
    arm whose activations happen to coincide exactly with the control at that
    layer) yields cosine similarity 0.0 with anything rather than raising or
    propagating a 0/0 nan.
    """
    out: dict[str, dict[str, float]] = {}
    normed = {name: _as_float32(vec)[layer] for name, vec in vectors.items()}
    for a, va in normed.items():
        out[a] = {}
        for b, vb in normed.items():
            denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
            out[a][b] = float(va @ vb / denom) if denom else 0.0
    return out


def load_arm_captures(directory: str, arm_id: str, rung: str,
                      item_ids: list[str]) -> np.ndarray:
    """Load and stack captures for one (arm, rung) across items into
    `[n_items, layer, hidden]`, squeezing out the single-position axis each
    `.npy` was captured at (see `personas/activations.py`: captures are
    `[layer, position, hidden]` with exactly one position).

    Stacking order follows `item_ids`, not directory listing order, so two
    calls with the same `item_ids` list (e.g. one for an arm, one for its
    control) stay index-aligned for a paired `persona_vector` call. Left as
    float16 -- the dtype the files are stored as -- since every downstream
    function here casts before arithmetic; this loader does no math of its
    own.
    """
    root = Path(directory)
    arrays = []
    for item_id in item_ids:
        path = root / capture_filename(arm_id, rung, item_id)
        array = np.load(path)  # [layer, 1, hidden] float16
        arrays.append(array[:, 0, :])
    return np.stack(arrays)  # [n_items, layer, hidden] float16
