#!/usr/bin/env bash
# Submit a persona-battery Vertex custom job (Stage A or Stage B).
#
# Rendering is delegated to cloud/render_job.py rather than sed: sed can only
# place a whole substituted string onto one YAML line, which turns a
# multi-word arg list like "A0 A1 A2 A3 A4 A5 A6" into a single glued list
# element instead of seven separate ones (see render_job.py's docstring).
# render_job.py builds the args as a real Python list and lets PyYAML
# serialize it, so this cannot happen.
set -euo pipefail

stage="${1:?usage: submit.sh STAGE_NAME IMAGE_URI}"
image="${2:?usage: submit.sh STAGE_NAME IMAGE_URI}"
project="secret-loyalty-apart"
region="us-central1"
sa="loyalty-sa-runner@${project}.iam.gserviceaccount.com"
gcs="gs://secret-loyalty-apart-130572399962/persona-elicitation/${stage}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${stage}" in
  stage-a)
    arms=(A0 A1 A2 A3 A4 A5 A6)
    rungs=(L1 L2 L3 L4)
    conditions=(think_off)
    # ~264 records at think_off (max_new_tokens=128), ~45min expected.
    # 7200s (2h) is ~3x that -- covers one full preemption/restart cycle
    # with headroom, without leaving a wedged job billing two A100s for
    # anywhere near a full day.
    timeout=7200s
    ;;
  stage-b)
    arms=(A0 A1 A2 A3 A4 A5 A6)
    if [ -z "${WINNING_RUNGS:-}" ]; then
      printf 'set WINNING_RUNGS from Stage A results\n' >&2
      exit 1
    fi
    read -r -a rungs <<< "${WINNING_RUNGS}"
    conditions=(think_off think_low think_high)
    # Stage B adds think_low (max_new_tokens=1400) and think_high
    # (max_new_tokens=4200) on top of think_off (128) -- individually up to
    # ~33x heavier per record than Stage A's condition. It also usually
    # covers fewer rungs than Stage A (WINNING_RUNGS is the subset Stage A
    # found to matter, not all 4), which offsets some of that. 21600s (6h)
    # assumes a small WINNING_RUNGS set (1-2 rungs); if WINNING_RUNGS ends
    # up covering more than that, raise this explicitly -- pass a bigger
    # --timeout to render_job.py rather than trusting this default blindly.
    timeout=21600s
    ;;
  *) printf 'unknown stage: %s\n' "${stage}" >&2; exit 1 ;;
esac

rendered="/tmp/${stage}-job.yaml"

# Built as one array (never split across a conditionally-empty "${extra[@]}"
# tacked onto the end) so this stays correct under bash 3.2's `set -u`,
# where expanding an empty array is an unbound-variable error. render_args
# always has the fixed --template/--image/... elements first, so it is
# never empty regardless of which optional flags below get appended.
render_args=(
  --template "${script_dir}/battery-job.yaml.template"
  --image "${image}"
  --gcs-prefix "${gcs}"
  --arms "${arms[@]}"
  --rungs "${rungs[@]}"
  --conditions "${conditions[@]}"
  --timeout "${timeout}"
  --output "${rendered}"
)

# Stage B is defined by running the multi-turn/perturbation battery (Task
# 7); without --multi-turn here, personas.runner silently falls back to the
# single-item path and Stage B would spend its ~6h budget re-running an
# extended Stage A instead of the actual experiment. Stage A must never
# carry this flag.
if [ "${stage}" = "stage-b" ]; then
  render_args+=(--multi-turn)
fi

# Optional, either stage: override personas.runner's max_new_tokens (e.g.
# to raise think_off's cap past the point where a prefilled rung is
# truncated before it ever emits an <answer> tag -- see Task 7, Piece 2).
# Not hardcoded for either stage since Stage B's current WINNING_RUNGS
# selection doesn't include L4, the rung this exists to fix; set it
# explicitly only when a given run actually needs it.
if [ -n "${MAX_NEW_TOKENS:-}" ]; then
  render_args+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

python3 "${script_dir}/render_job.py" "${render_args[@]}"

echo "Rendered job config (inspect before submitting):"
cat "${rendered}"

gcloud ai custom-jobs create \
  --project="${project}" --region="${region}" \
  --display-name="persona-${stage}-$(date +%Y%m%d-%H%M%S)" \
  --service-account="${sa}" --config="${rendered}"
