# Shared-core distillation and scaffolded reasoning

`bezier_nested_llm_7b_v2` · Nested language models

Improve the shared core through dense intermediate supervision, recurrence and teacher distillation while retaining the earlier patch-review ability.

## Saved status

Completed; strong outer scaffolded-trace learning, but independent core reasoning and useful conditional control not established.

## Work completed or attempted

- Added verified multi-stage arithmetic programs, training-only teacher caches, gold intermediate-state supervision and four tied core passes.
- Ran a guarded main attempt and validation-guided recovery; selected recovery checkpoint best_u000512.
- Evaluated direct answers, fixed-format free-running traces, fixed-action/permuted routing controls, core restoration and recurrence.

## Findings

**Measured.** The main run stopped at update 384 after retention-guard failure; recovery completed 768 updates, with its local update 512 selected. The 1,152 attempted updates do not equal retained weight-lineage updates.

[S162](../evidence/bezier_nested_llm_7b_v2/runs/results_summary.json) · [S161](../evidence/bezier_nested_llm_7b_v2/EXPERIMENT_REPORT.md)

**Measured.** On the fixed 16-program/64-example scaffolded test panel, outer state accuracy rose from 18.18% to 95.45%, final answer from 51.56% to 95.31%, and whole-trace correctness reached 90.63%. The core achieved 10.91% state, 53.13% final and 0% whole-trace accuracy.

[S162](../evidence/bezier_nested_llm_7b_v2/runs/results_summary.json)

**Measured.** Full direct-answer test accuracy was 53.71% with control disabled, 53.32% with steering and 50.98% for the core across 512 examples. The 95% scaffolded-panel result is not a full direct-answer result.

[S162](../evidence/bezier_nested_llm_7b_v2/runs/results_summary.json)

**Measured.** Deployed steering reduced same-checkpoint full NLL by 0.045572 nats, but accuracy changed -0.391 percentage points with an interval crossing zero. The learned controller did not beat the same-candidate always-turn control; permutation p=0.249 provided no support for input-specific alignment.

[S162](../evidence/bezier_nested_llm_7b_v2/runs/results_summary.json) · [S161](../evidence/bezier_nested_llm_7b_v2/EXPERIMENT_REPORT.md)

**Measured.** Restoring the pre-v2 core improved direct-answer NLL on test/OOD; four core passes had worse NLL than one. These diagnostics do not attribute the scaffolded outer improvement to independent core reasoning.

[S162](../evidence/bezier_nested_llm_7b_v2/runs/results_summary.json)

## Limits and open questions

- Single seed, adaptive recovery and bundled training changes; no causal isolation of distillation, recurrence or controller training.
- Trace outputs are constrained digits/A/B with a supplied format, not unrestricted text generation or compute-matched direct answers.
- Only one delayed MLP substitution; cached counterfactual and deployed metrics differ numerically and must be kept separate.

## Recommended next steps

- Recommended: retain the scaffolded-trace result with its exact 64-example scope.
- Recommended: separate reasoning quality, weight adaptation and inference-time controller gain in future experiments; v3 pursues controller evaluation and streaming.

## Sources

- [Architecture and reproduction](../evidence/bezier_nested_llm_7b_v2/README.md)
- [Completed results and limits](../evidence/bezier_nested_llm_7b_v2/EXPERIMENT_REPORT.md)
- [Machine-readable comparisons and trace scores](../evidence/bezier_nested_llm_7b_v2/runs/results_summary.json)

## Related projects

- [bezier_nested_llm_7b](bezier_nested_llm_7b.md)
- [bezier_nested_llm_7b_v3](bezier_nested_llm_7b_v3.md)

Saved report dated November 24, 2026.

[Back to the project directory](README.md)
