# Resident shared-core controller

`bezier_nested_llm_7b_v4` · Nested language models

Test whether a separate execution context using the same resident core storage can choose beneficial outer-layer substitutions.

## Saved status

Completed; exact shared residency and update scope verified, useful conditional control not demonstrated.

## Work completed or attempted

- Optimized exact core slices while freezing the other 85% of the 7B model.
- Probed route candidates on training data; compared MLP 27/29 substitutions at application layer 28.
- Completed training, selected an active candidate under a disclosed validation-rule amendment, and ran fresh test/OOD, shuffled controls, weight audits and resident runtime checks.

## Findings

**Measured.** Training completed 1,365 total updates (1,024 reasoning plus 341 replay); the selected checkpoint retained 768 reasoning/256 replay updates. All 6,160,822,272 outside-core elements remained bitwise unchanged; all 48 core-bearing tensors changed.

[S177](../evidence/bezier_nested_llm_7b_v4/runs/summary/results_summary.json) · [S180](../evidence/bezier_nested_llm_7b_v4/runs/final/weight_sharing.json)

**Measured.** The original selector chose Always Stay. A secondary rule adopted after validation inspection selected the active update-768 candidate; this amendment must accompany the result.

[S178](../evidence/bezier_nested_llm_7b_v4/runs/control_selection.json) · [S176](../evidence/bezier_nested_llm_7b_v4/EXPERIMENT_REPORT.md)

**Measured.** On 1,024 fresh test examples, learned routing scored 66.6992% versus 65.8203% disabled (+0.879 percentage points, 95% interval -0.879 to +2.637) and 67.0898% shuffled. OOD learned accuracy was 57.7148% versus 58.0078% disabled.

[S177](../evidence/bezier_nested_llm_7b_v4/runs/summary/results_summary.json)

**Measured.** The controller made active decisions but never selected MLP 27 on test/OOD. Test rescues/spoils were 55/46 with strong answer-position asymmetry; this does not establish semantic routing benefit.

[S183](../evidence/bezier_nested_llm_7b_v4/runs/final/control_diagnostics.json) · [S176](../evidence/bezier_nested_llm_7b_v4/EXPERIMENT_REPORT.md)

**Measured.** Creating a core execution view allocated zero additional CUDA bytes. On four 141-token prompts, controlled inference took 90.377 ms versus 70.724 ms disabled (1.278x), with no observed simultaneous GPU-kernel overlap.

[S181](../evidence/bezier_nested_llm_7b_v4/runs/final/resident/resident.json)

**Reported.** All five final tasks completed and 49 CPU tests passed; the generation smoke ran but returned the wrong arithmetic answer.

[S176](../evidence/bezier_nested_llm_7b_v4/EXPERIMENT_REPORT.md) · [S184](../evidence/bezier_nested_llm_7b_v4/runs/final/suite.json)

## Limits and open questions

- One seed, one substitution, fixed geometry and a selection amendment.
- No matched outer-only-training ablation; exact storage and preservation do not imply useful control.
- No offloading, quantization or KV cache; short runtime samples are not a serving benchmark.

## Recommended next steps

- Recommended: compare candidate-relative scores and fixed-denominator tie weighting under a predeclared protocol; v5 implements these follow-ups.
- Recommended: require fresh holdouts for further checkpoint or policy selection.

## Sources

- [Usage](../evidence/bezier_nested_llm_7b_v4/README.md)
- [Completed results and selection amendment](../evidence/bezier_nested_llm_7b_v4/EXPERIMENT_REPORT.md)
- [Measured quality/runtime summary](../evidence/bezier_nested_llm_7b_v4/runs/summary/results_summary.json)
- [Selection amendment](../evidence/bezier_nested_llm_7b_v4/runs/control_selection.json)

## Related projects

- [bezier_nested_llm_7b_v3](bezier_nested_llm_7b_v3.md)
- [bezier_nested_llm_7b_v5](bezier_nested_llm_7b_v5.md)

Machine summary created November 25, 2026 04:05 UTC (November 24 in America/New_York).

[Back to the project directory](README.md)
