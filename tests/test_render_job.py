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
import yaml

from cloud.render_job import build_args, render_job, render_to_file


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
    assert doc["scheduling"]["timeout"] == "86400s"
    assert doc["workerPoolSpecs"][0]["containerSpec"]["imageUri"] == (
        "gcr.io/secret-loyalty-apart/persona:stage-a"
    )
