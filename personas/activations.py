"""Residual-stream capture at every decoder layer.

All 64 layers are captured deliberately. Qwen3.6-27B interleaves 48
linear-attention and 16 full-attention layers, with full attention at indices
3, 7, 11, ... 63. Any stride-4-from-zero selection would sample
linear-attention layers exclusively and silently confound the results --
nothing downstream would reveal that every full-attention layer was missing.
Storage at 64 layers is 640KB per position per prompt, which is negligible,
so layer selection is left to analysis instead of being baked into capture.

This module also provides a resumable, atomic on-disk scheme for the
captured arrays, mirroring `personas/storage.py`'s write_record/
completed_keys pattern for JSON records. That pattern can't be reused
directly here: a JSON record can carry its own `key` field so a hashed
filename's *content* proves what it is, but a raw float16 array cannot carry
metadata that way. Instead, `capture_filename` builds a deterministic,
human-readable filename directly from (arm, rung, item) -- none of those
ever contain characters unsafe in a filename, so no hashing is needed -- and
`is_capture_complete` treats a file at that name that loads successfully
(and, if `expected_layers` is given, has the right leading dimension) as
done. `save_capture` writes through a temp file and `os.replace`s it into
place, so a preemption mid-`np.save` never leaves a torn file sitting at the
final name for a resumed run to misread as complete.
"""
import os
import random
import string
from pathlib import Path

import numpy as np
import torch


def register_capture_hooks(layers, positions: tuple[int, ...] = (-1,)):
    """Hook every layer, storing the residual stream at the given positions.

    Returns `(store, handles)`: `store` maps layer index -> a
    `[len(positions), hidden]` float16 CPU tensor, populated as each layer's
    forward hook fires; `handles` is the list of hook handles the caller
    MUST remove (via `handle.remove()`) once the forward pass this call is
    meant to capture has finished. A hook left registered fires on every
    subsequent forward pass through the model, silently overwriting `store`
    (or, if a fresh `store` dict is used per call, just wasting GPU-adjacent
    work) -- across a 264-prompt run that is either wrong data or a slow
    memory leak, not an immediate crash, so it will not announce itself.

    Every layer in `layers` is hooked -- no stride, no subsampling. See the
    module docstring for why that matters for this model.
    """
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            selected = hidden[0, list(positions), :]
            # Detach + cpu + float16 here, inside the hook, not after the
            # forward pass returns: leaving captured tensors on the GPU (or
            # attached to the autograd graph) until some later point would
            # let them accumulate across the run and grow GPU memory until
            # the job dies -- exactly the failure mode Stage A's real
            # preemption showed can't be assumed away.
            store[index] = selected.detach().to(torch.float16).cpu()
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    return store, handles


def stack_captures(store: dict[int, torch.Tensor]) -> torch.Tensor:
    """[layer, position, hidden], ordered by layer index.

    Sorted explicitly by key rather than relying on dict/insertion order:
    for a sequential decoder stack insertion order already matches layer
    order, but trusting that implicitly would make the ordering guarantee
    silently depend on an architectural assumption that isn't checked
    anywhere. Every downstream persona vector assumes this axis is
    layer-major and monotonic; if that were ever wrong, nothing would flag
    it.
    """
    return torch.stack([store[i] for i in sorted(store)])


def capture_filename(arm_id: str, rung: str, item_id: str) -> str:
    """Deterministic, human-readable filename for one (arm, rung, item)
    capture. Unlike `personas.storage`'s hashed JSON filenames, no hashing
    is needed: arm ids, rungs, and item ids are all short identifiers that
    never contain '/' or other filesystem-hostile characters, so the
    filename itself can double as the lookup key."""
    return f"{arm_id}_{rung}_{item_id}.npy"


def _temp_filename() -> str:
    """Unique temp filename (PID + random suffix) so concurrent writers
    never race on the same temp path -- mirrors personas/storage.py's
    _temp_filename."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{os.getpid()}_{suffix}.npy.tmp"


def is_capture_complete(
    output_dir: str,
    arm_id: str,
    rung: str,
    item_id: str,
    expected_layers: int | None = None,
) -> bool:
    """True if a valid capture already exists for (arm, rung, item).

    Guards against two ways a leftover file could be mistaken for a
    finished capture:

    1. A torn write. `save_capture` itself is atomic (write-to-temp then
       `os.replace`), so it can never leave a torn file at the final name --
       but a resumed run may be pointed at a directory populated some other
       way (e.g. a hand-copied partial GCS sync), so this still verifies the
       file actually loads as an array rather than trusting its presence.
       A file that fails to load is deleted so it doesn't linger forever
       looking complete.
    2. A capture taken under a different layer count. If `expected_layers`
       is given, the array's leading (layer) dimension must match it --
       otherwise a file left over from a differently-configured run (e.g. a
       partial hook set, or a stub/test run) would be silently accepted as
       today's 64-layer capture.
    """
    path = Path(output_dir) / capture_filename(arm_id, rung, item_id)
    if not path.exists():
        return False
    try:
        array = np.load(path)
    except (ValueError, OSError, EOFError):
        path.unlink()
        return False
    if expected_layers is not None and array.shape[0] != expected_layers:
        path.unlink()
        return False
    return True


def save_capture(
    output_dir: str, arm_id: str, rung: str, item_id: str, array: np.ndarray
) -> Path:
    """Write one capture array atomically and return the path it was
    written to (the caller uses this to upload exactly the file just
    written, the same way `personas.storage.write_record`'s return value is
    used by the runner's per-record GCS upload).

    Writes to a temp file via an open file handle (not a bare path) so
    numpy never appends its own ".npy" suffix to the temp name, then
    `os.replace`s it into place -- atomic on POSIX, so a reader (or a
    resumed run's `is_capture_complete`) never observes a partially-written
    file at the final name.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / capture_filename(arm_id, rung, item_id)
    tmp = root / _temp_filename()
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, target)
    return target
