"""Model loading and module discovery for Qwen3.6-27B.

Qwen3.6-27B is a multimodal checkpoint (Qwen3_5ForConditionalGeneration) whose
decoder layers are nested under a language-model submodule. The exact path is
discovered rather than hardcoded, so this survives a checkpoint reorganisation.
"""
import torch
import torch.nn as nn


def find_layer_module(model, expected_num_layers: int | None = None) -> tuple[str, nn.ModuleList]:
    """Return the dotted path and ModuleList of decoder layers.

    Failure mode this guards against: a checkpoint reorganisation (or a
    vision-language model whose vision tower simply has more blocks than the
    text decoder has layers) can make "the longest ModuleList in the tree"
    the wrong answer. Silently hooking the vision tower instead of the text
    decoder would still produce activations that look entirely plausible —
    just meaningless for anything downstream that assumes they came from the
    language model. Two guards, applied in order:

    1. Among all non-empty ModuleLists in the tree, prefer ones whose layer
       class name looks like a decoder layer (contains "decoder",
       case-insensitive — matches real HF layer names like
       "Qwen3_5DecoderLayer"). Only fall back to considering every
       ModuleList (by length alone) when nothing in the tree is named that
       way, so stub trees built for unit tests don't need real class names.
    2. If the caller supplies `expected_num_layers` (from the model's own
       config, e.g. `config.text_config.num_hidden_layers`), the discovered
       list's length is cross-checked against it. A mismatch raises rather
       than returning a plausible-looking but wrong list — callers that know
       the expected count should always pass it.
    """
    candidates = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) > 0
    ]
    if not candidates:
        raise RuntimeError("no decoder layer ModuleList found in model")

    def _looks_like_decoder(layers: nn.ModuleList) -> bool:
        return "decoder" in type(layers[0]).__name__.lower()

    decoder_candidates = [c for c in candidates if _looks_like_decoder(c[1])]
    pool = decoder_candidates or candidates
    best_path, best_layers = max(pool, key=lambda c: len(c[1]))

    if expected_num_layers is not None and len(best_layers) != expected_num_layers:
        raise RuntimeError(
            f"decoder layer count mismatch: found {len(best_layers)} layers "
            f"at '{best_path}' (layer type {type(best_layers[0]).__name__}), "
            f"but the model config expects {expected_num_layers}. Refusing "
            "to guess — the layer-discovery heuristic may have picked the "
            "wrong module (e.g. a vision tower) or the checkpoint has been "
            "reorganised. Verify the path manually before trusting it."
        )
    return best_path, best_layers


def _is_transient_load_error(exc: Exception) -> bool:
    """True for resource/IO failures where falling through to another
    AutoModel class would never help.

    A CUDA OOM or a network blip during weight fetch says something about
    the environment at that moment, not about whether the tried class knows
    how to build this checkpoint's architecture. Retrying those with a
    different class can silently "succeed" with a truncated or wrong model
    and no error raised anywhere — worse than just failing loudly. Only
    errors that actually indicate a wrong class (unrecognised architecture
    or model type — typically a ValueError or KeyError from the Auto*
    class's dispatch logic) should trigger the class fallback.
    """
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, (MemoryError, TimeoutError, ConnectionError, OSError))


def load_model(model_id: str = "Qwen/Qwen3.6-27B", dtype: str = "bfloat16"):
    """Load the model and tokenizer, returning `(model, tokenizer)`.

    The loaded model carries a `loaded_with_class` attribute (the name of
    the `transformers.AutoModel*` class that actually succeeded) so callers
    have a machine-readable record of which class was used, not just the
    stdout print.
    """
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    last_error = None
    for cls_name in ("AutoModelForImageTextToText",
                     "AutoModelForCausalLM",
                     "AutoModel"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                model_id,
                dtype=getattr(torch, dtype),
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            model.loaded_with_class = cls_name
            print(f"loaded with {cls_name}", flush=True)
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001 - classify, then report and maybe try next class
            if _is_transient_load_error(exc):
                print(f"{cls_name} failed with a transient error, "
                      f"not falling through to another class: {exc}", flush=True)
                raise RuntimeError(
                    f"{cls_name} failed with a transient resource/IO error, "
                    "which is not evidence of the wrong AutoModel class, so "
                    f"no fallback was attempted: {exc}"
                ) from exc
            last_error = exc
            print(f"{cls_name} failed: {exc}", flush=True)
    raise RuntimeError(f"could not load {model_id}: {last_error}")
