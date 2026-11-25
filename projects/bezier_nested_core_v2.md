# Nested core v2 - fixed whole-core budget

`bezier_nested_core_v2` · Nested cores

Constrain the entire central region to 1,312 shared parameters and jointly train direct Boolean reasoning, initial routing, outer computation, and live redirection control.

## Saved status

Completed seven-arm feasibility pilot; integrity checks passed but the quality criterion failed.

## Work completed or attempted

- Replaced v1's shared tensor slice with a complete 64-to-16-to-16 core, fixed at the origin, with all four parameter tensors shared across roles.
- Added parameter-free syntax features, independently verified Boolean programs, and program/truth-table-disjoint train, validation, test, and longer-program splits.
- Added gain regression and ranking supervision whose gradients reach the live outer representation, plus a detached-control ablation.
- Completed a seven-arm, seed-7, 128-update pilot with 128 examples (eight complete program truth tables) evaluated per split.
- Recorded gradient interaction, routing coverage, core participation, counterfactual regret, CUDA overlap, latency, memory, and separate CPU learning diagnostics.

## Findings

**Measured.** The entire core contains exactly 1,312 learned parameters. Shared models contain 175,328 parameters; the private-core control contains 176,640. Whole-core identity, tensor participation, and control gradients into the encoder passed the saved audit.

[S120](../evidence/bezier_nested_core_v2/runs/pilot/report.md)

**Measured.** All seven pilot arms achieved 50.00% outer answer accuracy at 16 visits. The primary concurrent core reached 50.78% direct-core accuracy and 46.88% on longer programs; its outer NLL was 0.704778 versus clean 0.703922.

[S120](../evidence/bezier_nested_core_v2/runs/pilot/report.md)

**Measured.** The primary concurrent arm accepted no redirects at its validation-selected margin. Full, one-turn, and zero-turn predictions were identical; the combined reasoning/control/routing criterion failed.

[S120](../evidence/bezier_nested_core_v2/runs/pilot/report.md) · [S121](../evidence/bezier_nested_core_v2/runs/pilot/suite.json)

**Measured.** A one-batch gradient audit found direct-reasoning versus control cosine -0.095453 and outer-answer versus control cosine -0.030264. These local conflicts are diagnostic, not established causes of the poor scores.

[S120](../evidence/bezier_nested_core_v2/runs/pilot/report.md) · [S125](../evidence/bezier_nested_core_v2/runs/pilot/seed7/gradient_alignment.json)

**Measured.** Captured concurrent batch-one inference took 1.511 ms versus the same-policy reference's 1.842 ms, while clean inference took 0.965 ms. Eager and graph kernel overlap were observed, but the full controlled model was not faster than the clean baseline.

[S120](../evidence/bezier_nested_core_v2/runs/pilot/report.md)

**Measured.** A separate raw-input CPU probe at learning rate 0.003 selected step 1,000 with validation NLL 0.676549 and 54.10% accuracy; its final step 5,000 NLL worsened to 1.079140. These training/validation diagnostics did not evaluate test or OOD examples.

[S123](../evidence/bezier_nested_core_v2/runs/scaling_diagnostics/core_probes.json)

## Limits and open questions

- The completed run is a short, capped, one-seed pilot; full seven-arm, five-seed confirmation remains unexecuted.
- Computation concentrated on a small subset of regions; broad routing diversity was not demonstrated.
- Controller regret is audited at the first eligible decision, not every later checkpoint.
- Sharing all weights does not ensure that reasoning and control objectives are compatible.
- Outer and core results concern a constrained Boolean benchmark, not general language or logical reasoning.

## Recommended next steps

- Recommendation: retain v2 as the fixed 1,312-parameter baseline against the larger-core revisions.
- Recommendation: investigate the direct-core learning bottleneck and routing concentration before promoting a full study; v3 and v4 provide saved follow-ups.

## Sources

- [Whole-core architecture and Boolean protocol](../evidence/bezier_nested_core_v2/README.md)
- [Seven-arm pilot results](../evidence/bezier_nested_core_v2/runs/pilot/report.md)
- [Completion and quality gates](../evidence/bezier_nested_core_v2/runs/pilot/suite.json)
- [Saved artifact verification](../evidence/bezier_nested_core_v2/runs/pilot/verification.json)
- [Standalone core learning diagnostics](../evidence/bezier_nested_core_v2/runs/scaling_diagnostics/core_probes.json)

## Related projects

- [bezier_nested_core](bezier_nested_core.md)
- [bezier_nested_core_v3](bezier_nested_core_v3.md)
- [bezier_nested_core_v4](bezier_nested_core_v4.md)

Saved results inspected November 25, 2026. Scientific status remains pilot_only in suite.json.

[Back to the project directory](README.md)
