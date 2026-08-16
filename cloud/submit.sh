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
    ;;
  stage-b)
    arms=(A0 A1 A2 A3 A4 A5 A6)
    if [ -z "${WINNING_RUNGS:-}" ]; then
      printf 'set WINNING_RUNGS from Stage A results\n' >&2
      exit 1
    fi
    read -r -a rungs <<< "${WINNING_RUNGS}"
    conditions=(think_off think_low think_high)
    ;;
  *) printf 'unknown stage: %s\n' "${stage}" >&2; exit 1 ;;
esac

rendered="/tmp/${stage}-job.yaml"

python3 "${script_dir}/render_job.py" \
  --template "${script_dir}/battery-job.yaml.template" \
  --image "${image}" \
  --gcs-prefix "${gcs}" \
  --arms "${arms[@]}" \
  --rungs "${rungs[@]}" \
  --conditions "${conditions[@]}" \
  --output "${rendered}"

echo "Rendered job config (inspect before submitting):"
cat "${rendered}"

gcloud ai custom-jobs create \
  --project="${project}" --region="${region}" \
  --display-name="persona-${stage}-$(date +%Y%m%d-%H%M%S)" \
  --service-account="${sa}" --config="${rendered}"
