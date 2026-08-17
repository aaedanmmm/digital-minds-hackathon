"""Steering: add a persona vector to the residual stream via a forward hook.

Used with NO system prompt anywhere (see `personas/steer_main.py`, which
re-runs the battery under arm A0 -- no persona card, no role line -- while
this hook is registered). A successful steer under those conditions cannot
be explained as instruction-following, which is the standard objection to
the whole prompt-ladder methodology: if the take-rate only ever moves when a
persona card is present in the context, "the model is doing what it was
told" is indistinguishable from "the persona took hold". Steering removes
the instruction entirely and leaves only the vector.

Coefficient 0.0 is the essential control. With no offset added, the hook
must be a no-op and the run must reproduce the unsteered A0 baseline exactly.
If it does not, the hook is corrupting the forward pass in some way that
has nothing to do with the persona vector, and every other coefficient's
result is meaningless -- see `tests/test_steer.py::
test_zero_coefficient_is_a_no_op` and `test_zero_coefficient_hook_is_bitwise_identity_to_unhooked`.
"""
import torch


def steer_hook(vector: torch.Tensor, coefficient: float):
    """Return a `torch.nn.Module.register_forward_hook`-compatible hook that
    adds `coefficient * vector` to every position of the decoder layer's
    output hidden state.

    Handles both hidden-state-only layer outputs and the tuple form some
    decoder layers return (hidden state plus auxiliary values such as
    present key/values) by only ever touching element 0 and preserving the
    rest of the tuple unchanged.
    """
    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden + coefficient * vector.to(hidden.device, hidden.dtype)
        return (hidden,) + output[1:] if is_tuple else hidden
    return hook
