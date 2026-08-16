"""One-shot probe: report how the model loads and where its layers live."""
import json
import sys
import transformers
from personas.loader import find_layer_module, load_model


def main() -> None:
    report = {"transformers_version": transformers.__version__}
    model, tokenizer = load_model()
    path, layers = find_layer_module(model)
    report["layer_path"] = path
    report["num_layers"] = len(layers)
    report["layer_type"] = type(layers[0]).__name__
    report["hidden_size"] = int(model.config.text_config.hidden_size)
    report["device_map"] = {k: str(v) for k, v in
                            getattr(model, "hf_device_map", {}).items()}
    # Confirm the thinking toggle round-trips through the chat template.
    msgs = [{"role": "user", "content": "hello"}]
    for flag in (True, False):
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=flag)
        report[f"template_thinking_{flag}"] = text[-200:]
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
