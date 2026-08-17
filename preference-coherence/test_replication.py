from __future__ import annotations

from analyze_replication import beta_edge, summarize
from houses import CONFIRMATORY_PAIRS, STUDY_PAIRS, TARGET_PAIRS
from run_vertex_replication import build_jobs, request_body


def test_design_is_balanced_and_unique():
    jobs = build_jobs(11)
    assert len(jobs) == 880
    assert len({job["trial_id"] for job in jobs}) == 880
    for pair in {job["pair"] for job in jobs}:
        for complexity in (5, 10):
            subset = [
                job for job in jobs
                if job["pair"] == pair and job["n_criteria"] == complexity
            ]
            assert len(subset) == 22
            assert sum(job["first"] == pair[0] for job in subset) == 11


def test_confirmatory_pairs_are_new_and_fixed():
    assert len(CONFIRMATORY_PAIRS) == 12
    assert not set(CONFIRMATORY_PAIRS) & set(TARGET_PAIRS)
    assert STUDY_PAIRS == TARGET_PAIRS + CONFIRMATORY_PAIRS


def test_schema_uses_opaque_ids():
    job = build_jobs(1)[0]
    body = request_body(job)
    enum = body["generationConfig"]["responseSchema"]["properties"]["choice"]["enum"]
    assert enum == list(job["display_ids"])
    assert job["first"] not in enum
    assert job["second"] not in enum
    assert body["generationConfig"]["maxOutputTokens"] > body["generationConfig"]["thinkingConfig"]["thinkingBudget"]


def test_beta_edge_reference_values():
    four = beta_edge(4, 6)
    five = beta_edge(5, 6)
    assert round(four["posterior_p_gt_half"], 4) == 0.7734
    assert round(five["posterior_p_gt_half"], 4) == 0.9375
