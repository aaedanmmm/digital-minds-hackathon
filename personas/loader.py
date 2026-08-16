"""Model loading and module discovery for Qwen3.6-27B.

Qwen3.6-27B is a multimodal checkpoint (Qwen3_5ForConditionalGeneration) whose
decoder layers are nested under a language-model submodule. The exact path is
discovered rather than hardcoded, so this survives a checkpoint reorganisation.
"""
import torch
import torch.nn as nn


def find_layer_module(model) -> tuple[str, nn.ModuleList]:
    """Return the dotted path and ModuleList of decoder layers.

    Picks the longest nn.ModuleList in the tree, which for a multimodal
    checkpoint is the text decoder rather than the vision tower.
    """
    best_path, best_layers = None, None
    for path, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            if best_layers is None or len(module) > len(best_layers):
                best_path, best_layers = path, module
    if best_layers is None:
        raise RuntimeError("no decoder layer ModuleList found in model")
    return best_path, best_layers


def load_model(model_id: str = "Qwen/Qwen3.6-27B", dtype: str = "bfloat16"):
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
            print(f"loaded with {cls_name}", flush=True)
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001 - report and try next class
            last_error = exc
            print(f"{cls_name} failed: {exc}", flush=True)
    raise RuntimeError(f"could not load {model_id}: {last_error}")
