# Focused Gemini preference-coherence replication

This is a replication of the order-instability signal in
`thunderingluck/digital-minds`. It retests the eight pairs implicated by the
published k=3/k=5/k=10 comparison and 12 pairs randomly drawn from the other
37, using Gemini 2.5 Flash through Vertex AI.

The full design is 20 pairs × 2 attribute counts × 2 presentation orders × 11
replicates = 880 calls. The odd replicate count prevents tied order-specific
majorities. Request order is deterministically shuffled, listing
identifiers are randomized opaque strings, temperature is 1.0, and the
thinking budget is fixed at 512 tokens.

```bash
cd preference-coherence
python run_vertex_replication.py --dry-run
python run_vertex_replication.py
python analyze_replication.py
python audit_published.py --input /path/to/published/raw_responses.jsonl
```

Authentication comes from `gcloud auth print-access-token`; no credential is
written to the results. Raw trials are stored individually under `results/raw`
and the run can be resumed safely by repeating the command. Analysis also
writes a consolidated, shareable `results/raw_responses.jsonl`; the resumable
per-call directory is intentionally gitignored.

See `LARGER_EXPERIMENT.md` for powered follow-up designs and
`experiment_budget.py` for call/cost/runtime estimates.

The recommended 12-set study has now been run:

```bash
python validate_prompt_sets.py
python run_large_study.py
python analyze_large_study.py
```

Its report is at `results/large-study/analysis/REPORT.md` and the consolidated
10,080-record source is `results/large-study/raw_responses.jsonl`.
