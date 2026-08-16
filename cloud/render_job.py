"""Render a Vertex custom-job YAML from cloud/battery-job.yaml.template.

Defect this exists to fix: the original approach built the container's
`args` list with `sed`, substituting a placeholder like `__ARMS__` with a
multi-word shell string such as `--arms A0 A1 A2 A3 A4 A5 A6` directly onto
one line of the YAML. That produces ONE YAML list element containing
spaces -- `- "--arms A0 A1 A2 A3 A4 A5 A6"` -- not seven separate elements.
Vertex hands each `args` element to the container as one argv token, so
`personas.runner`'s `argparse` (`--arms`, `nargs="+"`) would see a single
7-word string instead of seven arm ids, and `ARMS["A0 A1 A2 A3 A4 A5 A6"]`
would raise a KeyError five minutes into a paid A100 run.

The fix: never build the args list as text. Build it as a real Python
list, load the template as YAML (not as a text blob), assign the list to
`containerSpec.args`, and let PyYAML serialize it -- PyYAML always emits one
list element per Python list item, so this class of bug cannot recur here.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

TEMPLATE_PATH = Path(__file__).parent / "battery-job.yaml.template"

_TIMEOUT_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")

# Stage A runs ~264 records at think_off (max_new_tokens=128) and is
# reported to take roughly 45 minutes end to end. 7200s (2h) is ~3x that,
# which covers one full preemption-and-restart cycle with room to spare
# without leaving a wedged job sitting on two billed A100s for anywhere
# close to a full day. This is also render_job.py's own default timeout
# when a caller doesn't pass --timeout, since it's a reasonable floor for
# any single-condition run of this battery's size.
DEFAULT_TIMEOUT = "7200s"


def build_args(
    gcs_prefix: str | None,
    arms: list[str],
    rungs: list[str],
    conditions: list[str],
) -> list[str]:
    """Build the personas.runner argv as a real list -- one element per flag
    and one element per value, never a single glued string."""
    args: list[str] = ["personas.runner", "--output=/tmp/records"]
    if gcs_prefix:
        args.append(f"--gcs-prefix={gcs_prefix}")
    args += ["--arms", *arms]
    args += ["--rungs", *rungs]
    args += ["--conditions", *conditions]
    return args


def normalize_timeout(value: int | str) -> str:
    """Normalize a timeout to Vertex's protobuf Duration string format:
    digits followed by a literal 's' (e.g. "7200s").

    Fix round 2, Finding 3 bug: `scheduling.timeout` is a protobuf Duration,
    whose valid encoding is a seconds-count string suffixed with 's' (as in
    the known-good cloud/probe-job.yaml: `timeout: 3600s`). Passing a bare
    numeral like "7200" straight through into the YAML doc round-trips
    through PyYAML as the *string* '7200' -- PyYAML must quote it, since an
    unquoted 7200 would parse back as an int -- which is not a valid
    Duration and would fail (or be silently misinterpreted) at submit time.
    This function is the single place that decides what actually lands in
    `scheduling.timeout`, so every caller (CLI or library) goes through it
    and the 's' suffix can never be silently dropped again.

    Accepts a bare number of seconds ("7200", 7200) or an already-suffixed
    Duration string ("7200s") -- CLI callers may prefer to type plain
    seconds, submit.sh always passes the suffixed form -- but always
    returns the suffixed form.
    """
    text = str(value).strip()
    if text.endswith("s") and _TIMEOUT_NUMBER_RE.match(text[:-1]):
        return text
    if _TIMEOUT_NUMBER_RE.match(text):
        return f"{text}s"
    raise ValueError(
        f"invalid --timeout {value!r}: expected seconds as a bare number "
        "(e.g. 7200) or a Duration string (e.g. '7200s')"
    )


def _validate_template(doc, template_path: str | Path) -> dict:
    """Fail with a message naming what's missing, instead of a bare
    KeyError/IndexError pointing at a line inside render_job()."""
    if not isinstance(doc, dict):
        raise ValueError(
            f"template {template_path} did not parse to a YAML mapping "
            f"(got {type(doc).__name__})"
        )
    pools = doc.get("workerPoolSpecs")
    if not isinstance(pools, list) or not pools:
        raise ValueError(
            f"template {template_path} is missing a non-empty top-level "
            "'workerPoolSpecs' list"
        )
    pool0 = pools[0]
    if not isinstance(pool0, dict) or "containerSpec" not in pool0:
        raise ValueError(
            f"template {template_path}: workerPoolSpecs[0] is missing "
            "'containerSpec'"
        )
    container = pool0["containerSpec"]
    if not isinstance(container, dict) or "imageUri" not in container:
        raise ValueError(
            f"template {template_path}: workerPoolSpecs[0].containerSpec "
            "is missing 'imageUri'"
        )
    scheduling = doc.get("scheduling")
    if not isinstance(scheduling, dict):
        raise ValueError(
            f"template {template_path} is missing a top-level 'scheduling' mapping"
        )
    return doc


def render_job(
    *,
    image_uri: str,
    gcs_prefix: str | None,
    arms: list[str],
    rungs: list[str],
    conditions: list[str],
    timeout: str = DEFAULT_TIMEOUT,
    template_path: str | Path = TEMPLATE_PATH,
) -> dict:
    """Return the rendered job config as a plain dict (parsed YAML in,
    parsed YAML-shaped dict out -- never raw text substitution)."""
    doc = _validate_template(yaml.safe_load(Path(template_path).read_text()), template_path)
    container = doc["workerPoolSpecs"][0]["containerSpec"]
    container["imageUri"] = image_uri
    container["args"] = build_args(gcs_prefix, arms, rungs, conditions)
    doc["scheduling"]["timeout"] = normalize_timeout(timeout)
    return doc


def render_to_file(output_path: str | Path, **kwargs) -> None:
    doc = render_job(**kwargs)
    Path(output_path).write_text(yaml.safe_dump(doc, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(TEMPLATE_PATH))
    parser.add_argument("--image", required=True, help="container image URI")
    parser.add_argument("--gcs-prefix", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--rungs", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument(
        "--timeout", default=DEFAULT_TIMEOUT,
        help="Vertex scheduling.timeout in seconds -- '7200' or '7200s' are "
             f"both accepted, always rendered as the Duration form (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument("--output", required=True, help="path to write the rendered YAML")
    args = parser.parse_args()

    render_to_file(
        args.output,
        image_uri=args.image,
        gcs_prefix=args.gcs_prefix,
        arms=args.arms,
        rungs=args.rungs,
        conditions=args.conditions,
        timeout=args.timeout,
        template_path=args.template,
    )
    print(f"rendered {args.output}", flush=True)


if __name__ == "__main__":
    main()
