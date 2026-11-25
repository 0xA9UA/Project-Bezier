# Nested core v4 - preserve input information

`bezier_nested_core_v4` · Nested cores

Test whether a parameter-free syntax/input skip fixes the outer representation bottleneck while preserving v3's 5% shared core, parameter count, corpus, and training recipe.

## Saved status

Completed two-arm exploratory follow-up; first positive per-seed steering criterion in this family, awaiting independent replication.

## Work completed or attempted

- Added a fixed syntax/input skip to the recurrent encoder output in both clean and shared-controller arms, using existing deterministic projection buffers and no answer execution.
- Retained 4,704,970 total parameters and a fully shared 235,210-parameter core; completed the matched seed-7 clean/shared pair with 2,256 updates each.
- Evaluated complete held-out splits of 512 answer-prefix cases from 32 program files, saved validation-selected best and full-endpoint checkpoints, and compared matched v3/v4 results and provenance.
- Recorded artifact verification, shared-role gradient audits, CUDA timing/overlap, trajectory data, and an offline interactive experiment visualization.

## Findings

**Measured.** On 512 held-out answer-prefix cases from 32 program files in seed 7, the same shared checkpoint improved from 52.73% with redirects disabled to 57.62% with redirects enabled, averaging 6.055 redirects per 16 visits. Answer NLL fell from 0.657042 to 0.646470. The separately trained clean model reached 57.03% accuracy; the 4.89-point increase is versus disabled redirects, not versus that clean model.

[S143](../evidence/bezier_nested_core_v4/runs/input_preserving/comparison.md) · [S142](../evidence/bezier_nested_core_v4/runs/input_preserving/report.md) · [S145](../evidence/bezier_nested_core_v4/runs/input_preserving/suite.json)

**Measured.** Full-turn NLL gain was 0.010572 [95% paired-program interval 0.008350, 0.012941] versus zero turns; 0.004572 [0.003005, 0.006242] versus one turn; and 0.011479 [0.001261, 0.021992] versus the separately trained clean control. The saved per-seed combined criterion passed.

[S142](../evidence/bezier_nested_core_v4/runs/input_preserving/report.md) · [S144](../evidence/bezier_nested_core_v4/runs/input_preserving/verification.json)

**Measured.** Against the matched v3 seed-7 run, v4 reduced full-turn NLL by 0.024027 [0.016075, 0.032539] for the shared arm and 0.036815 [0.024529, 0.048552] for the clean arm. Both arms benefited from the input representation change.

[S143](../evidence/bezier_nested_core_v4/runs/input_preserving/comparison.md)

**Measured.** Shared direct-core accuracy was 59.57%, but longer-program core accuracy was 49.80%. Clean direct-core accuracy was 62.50%. Useful steering in this seed did not establish robust longer-program reasoning.

[S142](../evidence/bezier_nested_core_v4/runs/input_preserving/report.md)

**Measured.** The saved verification confirms two completed runs, identical v3 corpus, validation-only checkpoint selection, exact source snapshots, whole-core participation, and the selected shared update 1,756. The full original protocol remains unexecuted; suite scientific status is pilot_only.

[S144](../evidence/bezier_nested_core_v4/runs/input_preserving/verification.json) · [S145](../evidence/bezier_nested_core_v4/runs/input_preserving/suite.json)

**Measured.** Captured concurrent batch-one latency was 1.894 ms versus 2.297 ms for the same-checkpoint pipeline reference and 1.377 ms for clean inference. The shared model therefore retained a runtime cost versus the clean baseline, despite scheduling savings within the controlled model.

[S142](../evidence/bezier_nested_core_v4/runs/input_preserving/report.md)

## Limits and open questions

- One seed and an adaptive follow-up using reused holdouts. The 512 answer-prefix cases are grouped within 32 programs; program-bootstrap intervals do not establish replication across training seeds.
- Multiple descriptive intervals were reported without a multiplicity-adjusted study-level success gate.
- Validation-selected checkpoints and margins differ between revisions and arms.
- Longer-program reasoning remains approximately chance, and the shared direct core does not outperform the clean direct core.
- The no-checker graph also contains kernel overlap; eager named-scope profiling is needed for checker/outer attribution.

## Recommended next steps

- Recommendation: replicate the clean/shared input-preserving comparison with additional seeds and a new sequestered program holdout.
- Recommendation: add ablations that isolate input bits, syntax structure, and control supervision, and evaluate control competence at later redirect checkpoints.
- Recommendation: retain the latency and longer-program limitations alongside any positive steering claim.

## Sources

- [Input-preserving intervention and completed overview](../evidence/bezier_nested_core_v4/README.md)
- [Full v4 measured report](../evidence/bezier_nested_core_v4/runs/input_preserving/report.md)
- [Matched v3-v4 comparison](../evidence/bezier_nested_core_v4/runs/input_preserving/comparison.md)
- [Detailed saved verification](../evidence/bezier_nested_core_v4/runs/input_preserving/verification.json)
- [Completion state and quality gates](../evidence/bezier_nested_core_v4/runs/input_preserving/suite.json)
- [Interactive experiment visualization](../evidence/bezier_nested_core_v4/visualization/bezier-experiment.html)

## Related projects

- [bezier_nested_core_v3](bezier_nested_core_v3.md)
- [bezier_nested_core_v2](bezier_nested_core_v2.md)
- [bezier_nested_llm_7b](bezier_nested_llm_7b.md)

Saved completed artifacts inspected November 25, 2026. This index did not rerun training, tests, or GPU benchmarks.

[Back to the project directory](README.md)
