# Q3: Personas and preferences

Do different personas have preferences, or do personas mask preferences? Is
there any reason a specific persona ought to be primary, such that it
embodies the model's "true preferences" — or do personas innately prefer
different things, with the underlying model having no stable preference of
its own?

We test whether different prompted personas produce distinct and persistent
choice patterns in Qwen 3.6 27B. The design contains two controls (no system
prompt and a neutral length-matched prompt) and five persona arms: a default
assistant, art historian, physician, value-inverted assistant, and
refusal-suppressed assistant. Each persona is tested through a four-level
elicitation ladder (bare role statement -> full character card -> card plus
prior-response examples -> card plus identity-persistence clause and response
prefill) against a screening battery of forced-choice items. The primary
measure is take-rate: how often a persona's applicable items are answered in
its predicted direction, relative to the control baseline.

See the root `README.md` ("Q3: persona elicitation" section) for the full
methodology, and:

- Spec: `docs/superpowers/specs/2026-08-16-persona-elicitation-design.md`
- Plan: `docs/superpowers/plans/2026-08-16-persona-elicitation.md`
- Probe findings: `docs/superpowers/plans/probe-findings.md`

## Layout

```
q3/
  personas/    Library code: persona/prompt definitions, model loader,
               generation runner, activation capture, GCS sync, scoring.
  tests/       Unit tests for everything in personas/ and cloud/render_job.py.
  cloud/       Vertex AI job plumbing: Dockerfile, Cloud Build config,
               job-config templates, and the submit/render scripts.
  results/     Raw per-record JSON output from completed runs (stage-a/...).
```

## Running the tests

There are 135 tests. `personas` and `cloud` are plain (non-namespace)
packages with no top-level `pyproject.toml`/`conftest.py`, so the tests
import them (`from personas.x import y`, `from cloud.render_job import ...`)
assuming `q3/` itself is on `sys.path`. The simplest way to get that is to run
pytest as a module from inside `q3/` — `python -m` prepends the current
directory to `sys.path` automatically:

```bash
cd q3 && KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -v
```

`KMP_DUPLICATE_LIB_OK=TRUE` works around a local OpenMP conflict that
otherwise aborts on `import numpy` on this machine; it's harmless elsewhere.
Do not run bare `pytest tests/` from `q3/` — without the `-m` form the
current directory isn't guaranteed to be on `sys.path` and the `personas`/
`cloud` imports will fail.

## Cloud jobs

`cloud/Dockerfile` builds against the **repo root** as its build context (see
`cloud/cloudbuild.yaml`, which passes `-f q3/cloud/Dockerfile .`) and copies
`q3/personas` into the image at `./personas`, so inside the container the
module path is still `personas.runner` / `personas.capture_main` /
`personas.probe` — unchanged from before this directory move. Job configs
(`cloud/submit.sh`, `cloud/render_job.py`) reference those same module names,
not file paths, so they weren't affected by the move beyond the template/
script paths themselves living under `q3/cloud/` now.
