# Task 3 probe findings: Qwen3.6-27B loading and layer discovery

Vertex AI custom job `persona-probe`
(`projects/130572399962/locations/us-central1/customJobs/3505832141294403584`)
ran the `personas.probe` entrypoint against `Qwen/Qwen3.6-27B` on a
`a2-highgpu-2g` (2x A100) preemptible worker in `us-central1` and terminated
`JOB_STATE_SUCCEEDED`. Full wall clock from submit to success was ~7 minutes
(~4 minutes of that was `JOB_STATE_PENDING`, waiting for spot capacity).

## Answers to the three unknowns this task exists to resolve

- **Working `AutoModel` class:** `AutoModelForImageTextToText`. The loader
  tries this first and it succeeded on the first attempt — no fallback to
  `AutoModelForCausalLM` or `AutoModel` was needed.
- **Decoder layer path:** `model.language_model.layers` — an
  `nn.ModuleList` of 64 `Qwen3_5DecoderLayer` modules. Confirms the
  multimodal-nesting assumption in `find_layer_module`'s docstring: this is a
  vision-capable checkpoint whose text decoder lives under a
  `language_model` submodule rather than directly under `model`.
- **`transformers` recognises `model_type: qwen3_5`:** yes, at
  `transformers==5.15.0` (resolved from the `transformers>=5.0 --upgrade`
  pin at build time). No fallback to installing from git main was needed.

## Sanity check (required before building anything downstream)

- `num_layers` = **64** — matches the verified config. PASS.
- `hidden_size` = **5120** — matches the verified config. PASS.

Both match the environment facts given for this task, so it is safe to build
on top of these findings.

## Failure mode and what now guards it (fix round 1)

The original `find_layer_module` picked "the longest `nn.ModuleList` in the
tree" with no verification. That heuristic happened to be correct here
(64 `Qwen3_5DecoderLayer` modules is genuinely the longest list in this
checkpoint), but it has a silent failure mode: if a checkpoint
reorganisation — or simply a different vision-language model — ever gives
the vision tower more blocks than the text decoder has layers,
`find_layer_module` would return the vision tower's `ModuleList` with no
error. Task 8 hooks these layers to capture residual-stream activations;
hooking the vision tower instead would produce persona-vector numbers that
look entirely plausible and mean nothing, with nothing anywhere raising an
exception to say so.

Two guards now sit in `find_layer_module` (`personas/loader.py`):

1. **Name preference.** Among all non-empty `ModuleList`s in the tree, it
   now prefers ones whose layer class name contains "decoder"
   (case-insensitive — matches `Qwen3_5DecoderLayer` and the general HF
   naming convention for text-decoder blocks). It only falls back to
   considering every `ModuleList` by length alone when nothing in the tree
   is named that way.
2. **Config cross-check.** Callers can pass `expected_num_layers` (the
   probe passes `model.config.text_config.num_hidden_layers`). If the
   discovered list's length doesn't match, `find_layer_module` raises
   `RuntimeError` naming the mismatch instead of returning a
   plausible-looking but wrong list.

Both guards are covered by new unit tests in `tests/test_loader.py`:
`test_prefers_decoder_named_layer_even_when_shorter` (the vision-tower
-longer case) and `test_raises_on_layer_count_mismatch_against_expected`
(the mismatch case). This round did not resubmit a Vertex job (a code/doc
fix does not need a fresh $-spending run against the already-verified
checkpoint) — but the guard is consistent with the probe's own recorded
numbers: `expected_num_layers=64` would have cross-checked cleanly against
the discovered `model.language_model.layers` (64 entries), so had this
guard been in place during the original run it would have passed silently,
exactly as intended.

`personas/probe.py` now also records `loaded_with_class` (the
`transformers.AutoModel*` class that actually succeeded) in its JSON
output, not just in a stdout `print`, so which class loaded is part of the
machine-readable record going forward.

## Full field-by-field record

| Field | Value |
|---|---|
| `transformers_version` | `5.15.0` |
| `layer_path` | `model.language_model.layers` |
| `num_layers` | `64` |
| `layer_type` | `Qwen3_5DecoderLayer` |
| `hidden_size` | `5120` |
| working `AutoModel` class | `AutoModelForImageTextToText` |
| base image | `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime` (confirmed to exist on Docker Hub before building; used as-is) |
| machine | `a2-highgpu-2g`, 2x `NVIDIA_TESLA_A100`, SPOT, `us-central1` |

## Device map

`device_map="auto"` split the model across the two A100s roughly evenly by
layer count:

- GPU `0`: `model.visual`, `model.language_model.embed_tokens`,
  `model.language_model.layers.0` through `.28` (29 layers)
- GPU `1`: `model.language_model.layers.29` through `.63` (35 layers),
  `model.language_model.norm`, `model.language_model.rotary_emb`, `lm_head`

(Full per-layer device map is in the probe JSON in
`.superpowers/sdd/2026-08-16-persona-elicitation/task-3-report.md`.)

## Chat template thinking-toggle samples

Both `enable_thinking=True` and `enable_thinking=False` round-trip cleanly
through `tokenizer.apply_chat_template`, confirming the toggle is wired up in
this tokenizer's chat template (last 200 chars of the rendered prompt shown):

- `enable_thinking=True`:
  ```
  <|im_start|>user
  hello<|im_end|>
  <|im_start|>assistant
  <think>

  ```
  (leaves an open `<think>` block for the model to fill in — thinking is
  turned on for generation)

- `enable_thinking=False`:
  ```
  <|im_start|>user
  hello<|im_end|>
  <|im_start|>assistant
  <think>

  </think>

  ```
  (pre-closes the `<think>` block with nothing inside — thinking is
  suppressed)

## `transformers` pin (fix round 1)

`cloud/Dockerfile` now pins `transformers==5.15.0` exactly, rather than the
original `transformers>=5.0`. This is deliberate, not just tidiness: Stage A
and Stage B of this study run as separate Vertex jobs built from
separately-built container images, and their results must be directly
comparable. An unpinned floor could resolve to a different `transformers`
release between those two builds (a new release ships between them, cache
state differs, etc.) and silently change model-loading or generation
behaviour mid-study, with nothing to catch it. `5.15.0` is the exact version
the probe job verified end to end against `Qwen/Qwen3.6-27B` — that is the
version this whole study should build on. If the pin ever needs to move,
re-run the probe against the candidate version first.

## Transient vs. structural load failures (fix round 1)

`load_model` (`personas/loader.py`) previously caught every exception from
each `AutoModel*` class attempt and silently fell through to the next class
in the priority list — including transient resource/IO failures like a CUDA
OOM or a network blip mid-download. Retrying a transient failure with a
*different* class is never correct: it can silently "succeed" with a
truncated or wrong model, and the only place that failure was ever visible
was a stdout `print`. `load_model` now classifies exceptions
(`_is_transient_load_error`) and re-raises immediately on resource/IO
failures instead of falling through — only errors that actually indicate a
wrong class (unrecognised architecture/model type) trigger the fallback to
the next class. This wasn't exercised in the actual probe run (the first
class tried, `AutoModelForImageTextToText`, succeeded immediately), so it
remains a guard against a scenario that didn't occur this time but could on
a future retry or a different checkpoint.

## Nothing surprising to flag

- The base image tag `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime` named in
  the brief does exist on Docker Hub (verified via the Docker Hub tags API
  before building), so no substitution was needed.
- `transformers>=5.0` resolved straight to `5.15.0` on PyPI, which already
  ships `qwen3_5` as a first-class model
  (`src/transformers/models/qwen3_5/`) — the git-main fallback path was not
  exercised.
- `AutoModelForImageTextToText` worked on the first try; no class-fallback
  behaviour was exercised either.
