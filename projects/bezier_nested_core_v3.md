# Nested core v3 - scaling a 5% shared core

`bezier_nested_core_v3` · Nested cores

Test larger outer models with a proportionally larger shared core, more training data, staged warmup, and validation-selected joint checkpoints.

## Saved status

Completed seven-run bounded scale-up; modest reasoning gains, but repeated-steering and combined quality criteria failed.

## Work completed or attempted

- Built small (183,239 total / 9,223 core parameters) and large (4,704,970 total / 235,210 core parameters) models, keeping the complete shared core near 5% of the model.
- Expanded training to 1,024 Boolean programs, 256 distinct functions, and 16,384 examples while preserving the original 512-example validation, test, and longer-program splits.
- Completed all seven planned arm/seed runs: small clean/shared seed 7; large clean/shared seeds 7 and 19; large private-core seed 7.
- Used 500 core-only, 256 outer-warmup, and 1,500 joint updates per run; recorded best and final checkpoints, full heldout evaluations, paired-program comparisons, and source/data verification.
- Ran CPU direct-core learning probes and a training-only input-memory diagnostic that motivated revision 4.

## Findings

**Measured.** All 7/7 runs completed all 2,256 updates, with complete 512-example heldout evaluations and validation-selected joint checkpoints. The bounded study's combined quality outcome failed across its shared runs despite passing whole-core sharing and geometry integrity checks.

[S129](../evidence/bezier_nested_core_v3/runs/scaleup/report.md) · [S131](../evidence/bezier_nested_core_v3/runs/scaleup/verification.json) · [S130](../evidence/bezier_nested_core_v3/runs/scaleup/study.json)

**Measured.** Large shared seed 7 reached 53.52% outer accuracy and 0.6705 answer NLL versus clean 50.00% and 0.6948. Seed 19 reached 51.76% and 0.6931 versus clean 50.00% and 0.6938. Small clean and shared models both remained at 50.00% outer accuracy.

[S129](../evidence/bezier_nested_core_v3/runs/scaleup/report.md)

**Measured.** Large shared seed 7 made zero redirects, so its better clean-control score cannot be attributed to useful repeated steering. Seed 19 averaged 1.980 redirects, but full versus zero-turn gain was 0.000211 with 95% paired interval [-0.000283, 0.000607], and full versus one-turn gain was -0.000012 [-0.000058, 0.000035].

[S129](../evidence/bezier_nested_core_v3/runs/scaleup/report.md)

**Measured.** Large shared direct-core accuracy was 58.79% (seed 7) and 59.77% (seed 19), but longer-program core accuracy was 50.00% and 49.02%. The enlarged core learned modest in-distribution structure without reliable longer-program reasoning.

[S129](../evidence/bezier_nested_core_v3/runs/scaleup/report.md)

**Measured.** Large seed-7 captured concurrent batch-one inference took 1.811 ms versus the same-policy reference's 2.213 ms, but clean graph inference took 1.296 ms. No-checker graphs also overlapped, so graph overlap alone cannot attribute concurrency to checker work.

[S129](../evidence/bezier_nested_core_v3/runs/scaleup/report.md)

**Measured.** Standalone 235,210-parameter core probes selected validation NLL 0.67921 at learning rate 0.003 / step 500 and 0.68337 at 0.001 / step 1,500. Both final 2,000-update NLLs worsened, motivating bounded warmup with validation selection.

[S132](../evidence/bezier_nested_core_v3/runs/scaling_diagnostics/README.md) · [S140](../evidence/bezier_nested_core_v3/runs/scaling_diagnostics/scaled_core_probes.json)

**Reported.** A training-only diagnostic over 48 programs and all 16 input assignments found strong attenuation of input-bit perturbations by the recurrent outer encoder and traversal. This identified a representation bottleneck and motivated v4's parameter-free syntax/input skip.

[S133](../evidence/bezier_nested_core_v3/runs/scaling_diagnostics/outer_memory.json) · [S141](../evidence/bezier_nested_core_v4/README.md)

## Limits and open questions

- Only two large-model seeds and one small-model seed; no cross-seed confidence estimate.
- The larger original seven-arm protocol was not executed; the private control was only run for large seed 7.
- The reused benchmark holdouts are not a new sequestered confirmation set.
- v2-to-v3 differences include capacity, core fraction, data, training duration, optimization, and selection; their results do not isolate scaling alone.
- Program-bootstrap intervals describe program uncertainty within a seed, not training-seed variation.

## Recommended next steps

- Recommendation: preserve v3 as the matched reference for v4's representation intervention.
- Recommendation: replicate the larger model across more seeds and separate input preservation, core capacity, and control supervision in future ablations.

## Sources

- [Scaled architecture and training recipe](../evidence/bezier_nested_core_v3/README.md)
- [Complete seven-run scale-up results](../evidence/bezier_nested_core_v3/runs/scaleup/report.md)
- [Planned matrix and saved completion state](../evidence/bezier_nested_core_v3/runs/scaleup/study.json)
- [Data, checkpoint, and source verification](../evidence/bezier_nested_core_v3/runs/scaleup/verification.json)
- [Core-only warmup diagnostics](../evidence/bezier_nested_core_v3/runs/scaling_diagnostics/README.md)
- [Input-memory diagnostic](../evidence/bezier_nested_core_v3/runs/scaling_diagnostics/outer_memory.json)

## Related projects

- [bezier_nested_core_v2](bezier_nested_core_v2.md)
- [bezier_nested_core_v4](bezier_nested_core_v4.md)

Combined report snapshot 2026-11-25T00:01:26+00:00 (November 24, 2026, 7:01 PM America/New_York); indexed November 25.

[Back to the project directory](README.md)
