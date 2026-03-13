# CNN Search Plan (ResNet Supernet + Genetic Algorithm, IMX500 Target)

## 1) Goal and Success Criteria

### Primary Goal
Build a practical NAS pipeline that:
1. Trains one ResNet-family supernet with weight sharing.
2. Searches for compact subnetworks using a genetic algorithm.
3. Selects deployable candidates for Sony IMX500 constraints.

### Final Deliverables
- Supernet training pipeline.
- Genetic search pipeline with reproducible experiments.
- Automated post-search model adaptation pipeline (quantization/compression/export checks).
- Final report with model-level and GA-level evaluation.

### What “Success” Means
- A searched model that is Pareto-superior (or near-superior) to baseline compact ResNets on at least one tradeoff axis.
- Demonstrated feasibility on IMX500-relevant constraints (latency/memory/model size envelope).
- Statistically supported conclusions about GA choices and search settings.

---

## 2) Constraints and Assumptions

### Constraints from your target scenario
- Deployment target is edge hardware (IMX500 path), so architecture must be hardware-aware.
- Training/search time is limited, so experiments must use phased fidelity (cheap early ranking, expensive late confirmation).
- Search knobs are fixed to:
	- depth,
	- input image size,
	- output feature map size,
	- channel widths throughout the network.

### Assumptions to validate early
- Which exact operator set is accepted by the final IMX500 deployment toolchain.
- Real latency estimation path (on-device profiling vs surrogate predictor vs both).
- Effective memory budget (model + activation peak if relevant).

---

## 3) Supernet Design (ResNet Family, Reusable Subnetworks)

### 3.1 Search Space Definition (ResNet-compatible)
Define a structured, hardware-friendly space based on ResNet stages:
- Stem options: small discrete choices for stem width/stride pattern.
- Stage depths: selectable number of blocks per stage.
- Stage widths: selectable channel multipliers per stage.
- Block type options: keep a narrow set (e.g., basic/bottleneck variants only if hardware permits).
- Input resolution choices: discrete list (e.g., low/medium/high buckets).
- Output feature map size control: achieved through stage stride/downsampling schedule choices.

Keep all options in a small set of discrete values to make GA encoding easy and to reduce invalid architectures.

### 3.2 Supernet Construction Pattern
Build once, sample many:
- Each decision point is represented by switchable path choices.
- Shared weights are stored per operation/path; architecture parameters represent path selection logic.
- At each training step, activate one or a few subnetworks (not all simultaneously) for memory efficiency.

### 3.3 Reusability Strategy for Smaller Models
Design for direct extraction:
- Every searchable option must map to a concrete, standalone ResNet-like module.
- Create a deterministic “architecture config” schema (JSON/YAML) that fully defines a subnetwork.
- Add an extraction utility: `supernet weights + architecture config -> standalone model`.
- Ensure naming alignment so selected paths can be copied without fragile manual mapping.

### 3.4 Weight-Sharing Quality Controls
Because supernet ranking bias is a common failure mode:
- Use balanced sampling over architecture choices to avoid overtraining easy paths.
- Track per-choice usage frequency and enforce minimum sampling coverage.
- Periodically validate supernet ranking quality by full/short retraining of a small sampled set.

---

## 4) Supernet Training Plan

### 4.1 Training Regime
- Phase A: warm-up with broad, uniform architecture sampling.
- Phase B: fairness-aware sampling to correct undertrained choices.
- Phase C: optional mild focus on promising regions after initial search feedback.

### 4.2 Optimization and Stability
- Keep training recipe simple and consistent with existing project conventions.
- Use deterministic seeds and log all randomness sources.
- Checkpoint supernet regularly with searchable metadata (epoch, sampled architecture stats).

### 4.3 Validation Signals During Supernet Training
Track:
- proxy accuracy metrics,
- ranking consistency across repeated validation samples,
- per-choice usage histogram,
- resource proxy estimates (FLOPs/params/estimated latency).

Stop supernet training when ranking quality stabilizes, not just when loss plateaus.

---

## 5) Genetic Algorithm Choice and Design

### 5.1 Recommended Algorithm
Use a multi-objective GA, preferably NSGA-II as baseline:
- Objectives conflict naturally (accuracy vs latency vs memory), so Pareto ranking is appropriate.
- NSGA-II is robust, interpretable, and widely validated.

Optional follow-up: compare NSGA-II vs a constrained GA variant (where hard latency/memory are constraints and fitness maximizes accuracy).

### 5.2 Genome Encoding
Encode each architecture as a fixed-length chromosome:
- genes for stage depth choices,
- genes for stage channel multipliers,
- gene for input size,
- gene(s) for output feature-map choice via stride/downsampling pattern.

Use categorical/integer genes only (no free continuous genes unless later justified).

### 5.3 Validity and Repair
Some gene combinations may be invalid for deployment/toolchain:
- add a fast validator,
- add repair rules to map invalid candidates to nearest valid one,
- log repair rate (high repair rate means search space definition is poor).

### 5.4 Fitness Evaluation (Multi-objective)
Objectives:
1. Maximize accuracy (proxy or confirmed).
2. Minimize memory footprint (weights + optionally activation peak proxy).
3. Minimize latency (prefer on-target measurement or calibrated surrogate).

Use staged fidelity:
- Early generations: cheap proxy accuracy + latency surrogate.
- Late generations/finalists: partial fine-tune + real profiling where possible.

### 5.5 GA Hyperparameters (starting ranges)
- Population: 64-128.
- Generations: 40-100 (budget dependent).
- Crossover rate: 0.8-0.9.
- Mutation rate: 0.05-0.2 (gene-wise, adaptive recommended).
- Elitism: retain top Pareto front portion each generation.

Run multiple independent seeds (minimum 5, ideally 10+) for statistical confidence.

---

## 6) Experimental Design and Statistics

### 6.1 Two-Level Experiment Program
Level 1 (Method Development):
- reduced dataset,
- fewer classes,
- shorter training budget,
- fast turnaround for pipeline debugging and algorithm selection.

Level 2 (Confirmation):
- larger/representative subset,
- stronger training schedule for finalists,
- deployment-aware validation.

### 6.2 Dataset Subsetting Strategy
To reduce time while preserving realism:
- Use class-balanced subset (not random-only sampling).
- Keep validation split fixed across all experiments.
- Maintain at least one “hard” subset (small objects/crowded scenes if available).

### 6.3 Hypotheses to Test
Define explicit hypotheses before large runs:
- H1: GA-searched subnetworks improve Pareto efficiency vs fixed compact ResNet baselines.
- H2: Supernet proxy ranking positively correlates with retrained standalone performance.
- H3: Multi-objective GA provides better coverage/diversity of tradeoff solutions than constrained single-objective optimization.

### 6.4 Statistical Methods
- Use repeated runs with different seeds.
- Report mean, standard deviation, and confidence intervals for key metrics.
- For pairwise comparisons, use non-parametric tests when distributions are non-Gaussian.
- For multiple comparisons, control false discovery rate.

### 6.5 GA-Specific Evaluation Metrics
Report beyond “best model”:
- Hypervolume indicator over generations.
- Pareto front size and spread/diversity.
- Convergence trend (front movement across generations).
- Run-to-run stability (variance across seeds).

---

## 7) Quantization and Compression Automation

### 7.1 Why This Must Be in the Loop
Edge deployment performance depends on post-training transforms; architecture ranking can change after quantization/compression. Therefore, evaluate candidates in near-deployment form whenever feasible.

### 7.2 Automation Pipeline (per candidate)
For each selected candidate architecture:
1. Extract standalone model from supernet.
2. Optional brief calibration/fine-tuning pass.
3. Apply quantization workflow compatible with target path.
4. Apply optional compression passes (e.g., pruning/channel slimming where valid).
5. Export/check deployable artifact.
6. Measure post-transform accuracy + latency + memory.

Implement this as a single reproducible job pipeline with artifacts and logs.

### 7.3 Candidate Screening Policy
Avoid expensive quantization for every genome:
- GA loop uses proxy estimates.
- Every N generations, quantize/evaluate top-K and diversity picks.
- Final round quantizes full shortlisted Pareto set.

---

## 8) End-to-End Execution Plan (Phased)

### Phase 0 — Feasibility and Toolchain Validation
- Verify operator compatibility assumptions for IMX500 target path.
- Implement architecture validator + constraints schema.
- Lock experiment tracking/logging format.

### Phase 1 — Supernet MVP
- Implement searchable ResNet supernet with extraction utility.
- Train on reduced dataset and validate extraction correctness.
- Verify ranking sanity on small retraining sample.

### Phase 2 — GA MVP
- Implement NSGA-II loop with validity/repair and proxy fitness.
- Produce first Pareto front on reduced dataset.
- Analyze search behavior and fix obvious pathologies.

### Phase 3 — Integrated Evaluation
- Add automated quantization/compression pipeline.
- Evaluate periodic top-K and diverse candidates.
- Calibrate latency surrogate using measured values.

### Phase 4 — Statistical Study
- Run repeated-seed experiments for selected GA settings.
- Execute hypothesis testing and uncertainty reporting.
- Compare against baseline compact ResNets.

### Phase 5 — Final Selection and Handover
- Select final shortlist by Pareto + deployment constraints.
- Produce implementation report and reproducibility bundle.
- Prepare recommendation for integration into broader VideoDETR workflow.

---

## 9) Baselines and Ablations (Mandatory)

### Baselines
- Fixed ResNet-family compact models (e.g., small depth/width presets).
- Simple random search over same space (budget-matched).

### Ablation Studies
- Without quantization-aware screening vs with screening.
- Uniform supernet sampling vs fairness-aware sampling.
- NSGA-II vs constrained single-objective GA.
- Proxy-only ranking vs proxy + measured recalibration.

These ablations are needed to justify design choices, not only final results.

---

## 10) Reproducibility and Experiment Operations

### Experiment Tracking Requirements
- Persist architecture genome + decoded config for every evaluated candidate.
- Persist supernet checkpoint ID used for each evaluation.
- Persist seed, data split hash/version, and hardware profiling context.

### Operational Guardrails
- Separate fast debug runs from publishable runs.
- Use fixed evaluation protocol for comparability.
- Enforce run naming conventions and artifact versioning.

---

## 11) Risks and Mitigations

### Risk: Supernet ranking mismatch
Mitigation:
- regular rank-correlation checks with short standalone retraining,
- fair architecture sampling,
- avoid overly wide search space at start.

### Risk: Latency surrogate is inaccurate
Mitigation:
- periodic real profiling,
- surrogate recalibration,
- confidence intervals on latency predictions.

### Risk: GA converges prematurely
Mitigation:
- diversity-preserving selection,
- adaptive mutation,
- random immigrant injection.

### Risk: Quantization breaks top candidates
Mitigation:
- include quantization-aware checks before final selection,
- maintain diversity so alternatives exist.

---

## 12) Concrete Next Actions (Implementation Backlog)

1. Define searchable ResNet config schema (depth/width/resolution/output stride options).
2. Implement supernet module + subnetwork extraction utility.
3. Implement supernet trainer with balanced sampling and usage histograms.
4. Implement candidate validator/repair rules for deployment constraints.
5. Implement NSGA-II search loop with multi-objective logging.
6. Add surrogate resource estimators and measured-latency calibration hooks.
7. Build automated quantization/compression evaluation pipeline.
8. Add experiment runner for multi-seed statistical runs.
9. Implement reporting scripts (Pareto plots, hypervolume curves, hypothesis test tables).
10. Execute phased experiments and produce final recommendation shortlist.

---

## 13) Decision Rules for Final Model Selection

Use a transparent policy:
- First filter: hard deployment constraints (memory/latency envelope).
- Second filter: top Pareto candidates by post-quantization metrics.
- Third filter: stability across seeds and calibration runs.
- Final pick: best tradeoff for target scenario (e.g., lowest latency under minimum accuracy threshold).

This avoids selecting a model that is “best” only in one uncontrolled run.

