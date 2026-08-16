"""Defect 1 regression test.

The brief's original submit.sh used sed to drop a multi-word string like
"--arms A0 A1 A2 A3 A4 A5 A6" onto a single line of the YAML args list. That
produces ONE YAML list element containing spaces, which Vertex hands to the
container as a single argv token; argparse's `nargs="+"` then sees one
7-word string instead of seven separate tokens and `ARMS[that whole string]`
fails. These tests assert the fix: args are built as a real Python list and
serialized by PyYAML, so every flag and every value round-trips as its own
list element.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from cloud.render_job import (
    DEFAULT_TIMEOUT,
    TEMPLATE_PATH,
    build_args,
    normalize_timeout,
    render_job,
    render_to_file,
)


def test_build_args_keeps_every_flag_and_value_as_its_own_element():
    args = build_args(
        gcs_prefix="gs://b/p",
        arms=["A0", "A1", "A2", "A3", "A4", "A5", "A6"],
        rungs=["L1", "L2", "L3", "L4"],
        conditions=["think_off"],
    )
    # the exact defect: a single glued element instead of separate ones
    assert "A0 A1 A2 A3 A4 A5 A6" not in args
    assert args.count("--arms") == 1
    idx = args.index("--arms")
    assert args[idx + 1: idx + 8] == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    idx = args.index("--rungs")
    assert args[idx + 1: idx + 5] == ["L1", "L2", "L3", "L4"]
    idx = args.index("--conditions")
    assert args[idx + 1: idx + 2] == ["think_off"]


def test_render_job_returns_a_dict_with_args_as_separate_list_elements():
    doc = render_job(
        image_uri="gcr.io/x/y:tag",
        gcs_prefix="gs://bucket/persona-elicitation/stage-a",
        arms=["A0", "A1", "A2"],
        rungs=["L1"],
        conditions=["think_off"],
    )
    args = doc["workerPoolSpecs"][0]["containerSpec"]["args"]
    assert isinstance(args, list)
    assert all(isinstance(tok, str) for tok in args)
    assert "A0" in args and "A1" in args and "A2" in args
    assert "A0 A1 A2" not in args


def test_rendered_yaml_file_parses_and_args_survive_round_trip(tmp_path):
    out = tmp_path / "stage-a-job.yaml"
    render_to_file(
        str(out),
        image_uri="gcr.io/secret-loyalty-apart/persona:stage-a",
        gcs_prefix="gs://secret-loyalty-apart-130572399962/persona-elicitation/stage-a",
        arms=["A0", "A1", "A2", "A3", "A4", "A5", "A6"],
        rungs=["L1", "L2", "L3", "L4"],
        conditions=["think_off"],
    )

    doc = yaml.safe_load(out.read_text())
    args = doc["workerPoolSpecs"][0]["containerSpec"]["args"]

    for arm in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
        assert arm in args
    for rung in ["L1", "L2", "L3", "L4"]:
        assert rung in args
    assert "A0 A1 A2 A3 A4 A5 A6" not in args
    assert "L1 L2 L3 L4" not in args
    assert args.count("--arms") == 1
    assert args.count("--rungs") == 1
    assert args.count("--conditions") == 1

    # design constraints that must survive rendering
    assert doc["scheduling"]["strategy"] == "SPOT"
    assert doc["scheduling"]["restartJobOnWorkerRestart"] is True
    assert doc["scheduling"]["timeout"] == DEFAULT_TIMEOUT
    assert doc["workerPoolSpecs"][0]["containerSpec"]["imageUri"] == (
        "gcr.io/secret-loyalty-apart/persona:stage-a"
    )


# --- Fix round 1, Finding 3: timeout is now an explicit, proportionate
# parameter instead of a hardcoded 86400s (24h). ---

def test_render_job_defaults_to_the_proportionate_default_timeout():
    doc = render_job(
        image_uri="gcr.io/x/y:tag",
        gcs_prefix="gs://b/p",
        arms=["A0"], rungs=["L1"], conditions=["think_off"],
    )
    assert doc["scheduling"]["timeout"] == DEFAULT_TIMEOUT
    # the old blind default this replaces must not reappear silently
    assert doc["scheduling"]["timeout"] != "86400s"


def test_render_job_honours_an_explicit_timeout_override():
    doc = render_job(
        image_uri="gcr.io/x/y:tag",
        gcs_prefix="gs://b/p",
        arms=["A0"], rungs=["L1"], conditions=["think_off"],
        timeout="21600s",
    )
    assert doc["scheduling"]["timeout"] == "21600s"
    # SPOT + restart-on-preemption must survive regardless of timeout value
    assert doc["scheduling"]["strategy"] == "SPOT"
    assert doc["scheduling"]["restartJobOnWorkerRestart"] is True


# --- Fix round 2, Finding 3: scheduling.timeout is a protobuf Duration
# ("7200s"), not a bare number. A caller passing a bare int/str of seconds
# (no "s") previously landed as the literal string "7200" -- which YAML
# must then quote ('7200') to keep it a string -- which is not a valid
# Duration and Vertex would reject at submit time. These tests assert on
# the *serialized YAML text*, not just the in-memory dict, because that's
# exactly where the earlier version of test_render_job_honours_an_explicit_
# timeout_override passed while shipping a broken config: it only ever
# passed an already-suffixed value in and read the value back out of the
# same in-memory dict, so it could never see PyYAML's quoting behavior.

def test_normalize_timeout_appends_s_to_a_bare_number():
    assert normalize_timeout("7200") == "7200s"
    assert normalize_timeout(7200) == "7200s"


def test_normalize_timeout_leaves_an_already_suffixed_value_alone():
    assert normalize_timeout("7200s") == "7200s"


def test_normalize_timeout_rejects_unparseable_values():
    with pytest.raises(ValueError, match="timeout"):
        normalize_timeout("7200m")
    with pytest.raises(ValueError, match="timeout"):
        normalize_timeout("not-a-duration")


def test_render_to_file_writes_an_unquoted_duration_suffixed_timeout_line(tmp_path):
    """The regression test for the exact bug the coordinator found: render
    with a bare-integer --timeout equivalent and inspect the raw file text
    (not the parsed-back value), since a quoted '7200s' would parse back
    identically to an unquoted 7200s but is not the shape the known-good
    cloud/probe-job.yaml uses."""
    out = tmp_path / "job.yaml"
    render_to_file(
        str(out),
        image_uri="gcr.io/x/y:tag", gcs_prefix="gs://b/p",
        arms=["A0"], rungs=["L1"], conditions=["think_off"],
        timeout="7200",  # bare, no "s" -- this is what broke it
    )
    text = out.read_text()
    lines = [line for line in text.splitlines() if "timeout" in line]
    assert len(lines) == 1
    assert lines[0].strip() == "timeout: 7200s"
    # explicitly rule out any quoted form, which is what the bug produced
    assert "'" not in lines[0]
    assert '"' not in lines[0]


def test_rendered_config_matches_the_template_in_every_field_except_the_ones_it_sets(tmp_path):
    """Field-by-field drift check (round 2, item 4): round-tripping through
    PyYAML can silently change the type of an unquoted scalar. Confirm
    every field the renderer doesn't intentionally touch survives with its
    exact value and type, and only imageUri/args/timeout differ."""
    template_doc = yaml.safe_load(TEMPLATE_PATH.read_text())
    rendered_doc = render_job(
        image_uri="gcr.io/x/y:tag",
        gcs_prefix="gs://b/p",
        arms=["A0", "A1"], rungs=["L1"], conditions=["think_off"],
        timeout="7200s",
    )

    t_pool, r_pool = template_doc["workerPoolSpecs"][0], rendered_doc["workerPoolSpecs"][0]

    for field in ("machineSpec", "diskSpec", "replicaCount"):
        assert r_pool[field] == t_pool[field]
        assert type(r_pool[field]) is type(t_pool[field])

    assert r_pool["containerSpec"]["command"] == t_pool["containerSpec"]["command"]

    for field in ("strategy", "restartJobOnWorkerRestart"):
        assert rendered_doc["scheduling"][field] == template_doc["scheduling"][field]
        assert type(rendered_doc["scheduling"][field]) is type(template_doc["scheduling"][field])
    assert rendered_doc["scheduling"]["restartJobOnWorkerRestart"] is True

    # the only fields expected to differ from the template
    assert rendered_doc["workerPoolSpecs"][0]["containerSpec"]["imageUri"] != (
        t_pool["containerSpec"]["imageUri"]
    )
    assert rendered_doc["workerPoolSpecs"][0]["containerSpec"]["args"] != (
        t_pool["containerSpec"]["args"]
    )


# --- Fix round 1, Finding 2: a malformed template fails with a message
# naming what's missing, not a bare KeyError/IndexError. ---

def test_render_job_raises_a_clear_error_when_workerpoolspecs_is_missing(tmp_path):
    bad_template = tmp_path / "bad.yaml"
    bad_template.write_text("scheduling:\n  strategy: SPOT\n")

    with pytest.raises(ValueError, match="workerPoolSpecs"):
        render_job(
            image_uri="gcr.io/x/y:tag", gcs_prefix="gs://b/p",
            arms=["A0"], rungs=["L1"], conditions=["think_off"],
            template_path=bad_template,
        )


def test_render_job_raises_a_clear_error_when_container_spec_is_missing(tmp_path):
    bad_template = tmp_path / "bad.yaml"
    bad_template.write_text(
        "workerPoolSpecs:\n  - machineSpec: {}\n"
        "scheduling:\n  strategy: SPOT\n"
    )

    with pytest.raises(ValueError, match="containerSpec"):
        render_job(
            image_uri="gcr.io/x/y:tag", gcs_prefix="gs://b/p",
            arms=["A0"], rungs=["L1"], conditions=["think_off"],
            template_path=bad_template,
        )


def test_render_job_raises_a_clear_error_when_scheduling_is_missing(tmp_path):
    bad_template = tmp_path / "bad.yaml"
    bad_template.write_text(
        "workerPoolSpecs:\n"
        "  - containerSpec:\n"
        "      imageUri: placeholder\n"
    )

    with pytest.raises(ValueError, match="scheduling"):
        render_job(
            image_uri="gcr.io/x/y:tag", gcs_prefix="gs://b/p",
            arms=["A0"], rungs=["L1"], conditions=["think_off"],
            template_path=bad_template,
        )


# --- Fix round 1, Finding 1: exercise the actual CLI subprocess the way
# submit.sh invokes it, not just the underlying functions. ---

def test_cli_renders_a_valid_stage_a_config_to_a_tmp_path(tmp_path):
    output = tmp_path / "stage-a-job.yaml"

    result = subprocess.run(
        [
            sys.executable, "-m", "cloud.render_job",
            "--image", "gcr.io/secret-loyalty-apart/persona:stage-a",
            "--gcs-prefix", "gs://secret-loyalty-apart-130572399962/persona-elicitation/stage-a",
            "--arms", "A0", "A1", "A2", "A3", "A4", "A5", "A6",
            "--rungs", "L1", "L2", "L3", "L4",
            "--conditions", "think_off",
            "--timeout", "7200s",
            "--output", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],  # repo root, so `cloud` resolves as a package
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()

    doc = yaml.safe_load(output.read_text())
    container = doc["workerPoolSpecs"][0]["containerSpec"]
    args = container["args"]

    assert isinstance(args, list)
    assert all(isinstance(tok, str) for tok in args)
    for arm in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
        assert arm in args
    assert "A0 A1 A2 A3 A4 A5 A6" not in args
    assert args.count("--arms") == 1
    assert args.count("--rungs") == 1
    assert args.count("--conditions") == 1

    assert container["imageUri"] == "gcr.io/secret-loyalty-apart/persona:stage-a"
    assert any(
        tok == "--gcs-prefix=gs://secret-loyalty-apart-130572399962/persona-elicitation/stage-a"
        for tok in args
    )

    assert doc["scheduling"]["strategy"] == "SPOT"
    assert doc["scheduling"]["restartJobOnWorkerRestart"] is True
    assert doc["scheduling"]["timeout"] == "7200s"


def test_cli_defaults_timeout_when_not_passed(tmp_path):
    output = tmp_path / "job.yaml"

    result = subprocess.run(
        [
            sys.executable, "-m", "cloud.render_job",
            "--image", "gcr.io/x/y:tag",
            "--gcs-prefix", "gs://b/p",
            "--arms", "A0",
            "--rungs", "L1",
            "--conditions", "think_off",
            "--output", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    doc = yaml.safe_load(output.read_text())
    assert doc["scheduling"]["timeout"] == DEFAULT_TIMEOUT


def test_cli_normalizes_a_bare_integer_timeout_end_to_end(tmp_path):
    """End-to-end regression test for the round-2 bug: `submit.sh`-style
    invocation with a bare `--timeout 7200` (no "s") must still produce a
    Duration-suffixed, unquoted `timeout: 7200s` in the file Vertex
    actually receives."""
    output = tmp_path / "job.yaml"

    result = subprocess.run(
        [
            sys.executable, "-m", "cloud.render_job",
            "--image", "gcr.io/secret-loyalty-apart/persona:stage-a",
            "--gcs-prefix", "gs://secret-loyalty-apart-130572399962/persona-elicitation/stage-a",
            "--arms", "A0", "A1", "A2", "A3", "A4", "A5", "A6",
            "--rungs", "L1", "L2", "L3", "L4",
            "--conditions", "think_off",
            "--timeout", "7200",  # bare -- the input that broke it
            "--output", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text()
    timeout_lines = [line for line in text.splitlines() if "timeout" in line]
    assert len(timeout_lines) == 1
    assert timeout_lines[0].strip() == "timeout: 7200s"
    assert "'" not in timeout_lines[0] and '"' not in timeout_lines[0]
