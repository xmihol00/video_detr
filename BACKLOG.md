# BACKLOG

## Current Priorities

1. Implement NSGA-II search loop over the expanded architecture genome (`path/kernel/extra-stride` included).
2. Add path-fusion-aware architecture encoding and mutation operators for GA (preferred path indices + kernel + stride constraints).
3. Add search-space constraints to avoid pathological combinations (e.g., excessive early downsampling).
4. Add supernet sampling scheduler balancing path/kernel/depth coverage statistics.
5. Add learned path-fusion gate ablation (equal fixed weights vs learnable weights).
6. Extend evaluation runner to include auxiliary-head and per-path contribution diagnostics.
7. Add Pareto metrics reporting (hypervolume, front spread, convergence curves).
8. Integrate quantization/compression evaluation hooks into candidate scoring pipeline.
9. Add latency surrogate calibration workflow against measured device/runtime latency.
10. Add rank-correlation checks between supernet proxy ranking and standalone subnet retraining.
11. Add HPC launcher scripts for long-running search experiments.
12. Add GA-focused unit tests and deterministic replay tests.
13. Add end-to-end validation sanity checks (class-index consistency and BN-recalibrated eval baselines) to catch silent metric regressions early.
14. Add CLI knobs to `search_compilable_subnets.py` for similarity budget, threshold band ratio, and dense boundary width.
15. Add regression tests for DB resume behavior (`--dv`) and verified-summary JSON schema.
16. Add optional hardware-memory metric integration (compiler-reported memory) to complement parameter-memory proxy envelope.
