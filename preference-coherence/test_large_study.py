from __future__ import annotations

from run_large_study import COMPLEXITIES, N_SETS, balanced_subsets, build_jobs


def test_balanced_attribute_subsets():
    for complexity in COMPLEXITIES:
        subsets = balanced_subsets(complexity)
        assert len(subsets) == N_SETS
        assert all(len(subset) == complexity for subset in subsets)
        counts = [sum(index in subset for subset in subsets) for index in range(10)]
        assert max(counts) - min(counts) <= 1


def test_full_job_grid_and_independent_schema_order():
    records = []
    for set_index in range(12):
        records.append({
            "set_id": f"S{set_index + 1:02d}",
            "options": [
                {"id": option, "values": [1000 + index, 10, 50, 5, 50, 12, 5, 5, 5, 5]}
                for index, option in enumerate("ABCDEF")
            ],
        })
    jobs = build_jobs(records)
    assert len(jobs) == 10_080
    assert len({job["trial_id"] for job in jobs}) == 10_080
    first_schema_matches = sum(job["schema_ids"][0] == job["display_ids"][0] for job in jobs)
    assert first_schema_matches == len(jobs) // 2
