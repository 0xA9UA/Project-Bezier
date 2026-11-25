# Absolute versus relative shared-core scoring

`bezier_nested_llm_7b_v5` · Nested language models

Test corrected tie weighting and a genuine Stay-reference subtraction in two frozen controller arms.

## Saved status

Completed two-arm pilot; absolute remains frozen primary, neither arm demonstrates conditional benefit.

## Work completed or attempted

- Implemented fixed-denominator control loss and FP32 shared-head margins in both arms.
- Compared absolute candidate margins against relative candidate-minus-Stay margins using the same warm start and planned example order.
- Preregistered selection, evaluated both frozen arms, preserved the primary arm, and audited selected versus completed training exposure.

## Findings

**Measured.** Absolute completed 1,024 reasoning/341 replay updates and selected update 768; relative stopped at 256/85 after a retention guard and selected update 128 with Always Stay. Completed budgets and controller compute were unequal.

[S187](../evidence/bezier_nested_llm_7b_v5/runs/summary/results_summary.json) · [S186](../evidence/bezier_nested_llm_7b_v5/EXPERIMENT_REPORT.md)

**Measured.** Primary absolute learned test accuracy was 64.7461% versus 65.0391% disabled; OOD was 59.3750% versus 60.1563%. Learned-minus-disabled differences were -0.293 and -0.781 percentage points, with both paired intervals crossing zero.

[S187](../evidence/bezier_nested_llm_7b_v5/runs/summary/results_summary.json)

**Measured.** Relative chose Stay for every test/OOD/replay example: learned, disabled, fixed and shuffled results were identical (65.1367% test, 61.0352% OOD). The higher OOD point estimate cannot be attributed to conditional routing.

[S193](../evidence/bezier_nested_llm_7b_v5/runs/final/relative/evaluation.json) · [S186](../evidence/bezier_nested_llm_7b_v5/EXPERIMENT_REPORT.md)

**Measured.** Corrected all-tie loss scaling matched approximately 0.08, but first-128-update rescue/spoil AUC remained near chance: 0.489 absolute and 0.515 relative. These are changing-weight training diagnostics.

[S194](../evidence/bezier_nested_llm_7b_v5/runs/training_diagnostics_r128.json) · [S186](../evidence/bezier_nested_llm_7b_v5/EXPERIMENT_REPORT.md)

**Measured.** Both weight audits preserved every outside-core element; extra core-view CUDA allocation was zero. Controlled/disabled median latency was 1.305x absolute and 1.460x relative, without observed kernel overlap.

[S195](../evidence/bezier_nested_llm_7b_v5/runs/final/absolute_weight_sharing.json) · [S196](../evidence/bezier_nested_llm_7b_v5/runs/final/relative_weight_sharing.json) · [S197](../evidence/bezier_nested_llm_7b_v5/runs/final/absolute_resident/resident.json) · [S198](../evidence/bezier_nested_llm_7b_v5/runs/final/relative_resident/resident.json)

**Reported.** The report records 68 CPU checks and eight completed final tasks; the free-form generation smoke answered incorrectly.

[S186](../evidence/bezier_nested_llm_7b_v5/EXPERIMENT_REPORT.md) · [S199](../evidence/bezier_nested_llm_7b_v5/runs/final/suite.json)

## Limits and open questions

- Single seed and narrow program task; relative's early guard stop prevents a matched completed-budget comparison.
- Both arms share normalization and readout-precision changes, so the arm comparison does not isolate all changes from v4.
- Neither arm establishes useful learned steering; relative's failed pilot is not a general disproof of reference subtraction.

## Recommended next steps

- Recommended: preserve the registered primary and report actual exposure rather than describing completed arms as matched.
- Recommended: move beyond one whole-MLP substitution only with direct execution audits and strong fixed/shuffled controls; v6 implements node-level routing.

## Sources

- [Usage](../evidence/bezier_nested_llm_7b_v5/README.md)
- [Completed two-arm report](../evidence/bezier_nested_llm_7b_v5/EXPERIMENT_REPORT.md)
- [Frozen primary/cohort results](../evidence/bezier_nested_llm_7b_v5/runs/summary/results_summary.json)
- [Corrected cohort figure provenance](../evidence/bezier_nested_llm_7b_v5/runs/summary_final/render_provenance.json)

## Related projects

- [bezier_nested_llm_7b_v4](bezier_nested_llm_7b_v4.md)
- [bezier_nested_llm_7b_v6](bezier_nested_llm_7b_v6.md)

Machine summary created November 25, 2026 10:49 UTC; saved report updated November 25.

[Back to the project directory](README.md)
