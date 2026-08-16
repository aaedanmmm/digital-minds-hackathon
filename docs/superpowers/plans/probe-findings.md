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
