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
from pathlib import Path

import yaml

TEMPLATE_PATH = Path(__file__).parent / "battery-job.yaml.template"


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


def render_job(
    *,
    image_uri: str,
    gcs_prefix: str | None,
    arms: list[str],
    rungs: list[str],
    conditions: list[str],
    template_path: str | Path = TEMPLATE_PATH,
) -> dict:
    """Return the rendered job config as a plain dict (parsed YAML in,
    parsed YAML-shaped dict out -- never raw text substitution)."""
    doc = yaml.safe_load(Path(template_path).read_text())
    container = doc["workerPoolSpecs"][0]["containerSpec"]
    container["imageUri"] = image_uri
    container["args"] = build_args(gcs_prefix, arms, rungs, conditions)
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
    parser.add_argument("--output", required=True, help="path to write the rendered YAML")
    args = parser.parse_args()

    render_to_file(
        args.output,
        image_uri=args.image,
        gcs_prefix=args.gcs_prefix,
        arms=args.arms,
        rungs=args.rungs,
        conditions=args.conditions,
        template_path=args.template,
    )
    print(f"rendered {args.output}", flush=True)


if __name__ == "__main__":
    main()
