# Pretrained 7B shared-core pilot

`bezier_nested_llm_7b` · Nested language models

Test whether a central core sharing 15% of a pretrained Mistral model can reason independently and choose useful inference-time MLP substitutions.

## Saved status

Completed pilot; adaptation succeeds on its bounded task, independent core reasoning and useful rerouting not demonstrated.

## Work completed or attempted

- Adapted Mistral-7B-Instruct-v0.3 on bounded Python patch review using matched baseline and shared-controller arms.
- Implemented a logical central core with actual parameter aliases, single Bézier-endpoint MLP substitution, independent-core inference, held-out paired evaluation, and CUDA graph concurrency checks.
- Completed 512 joint updates per arm; saved frozen protocols, checkpoints, corpus identities and evaluation records.

## Findings

**Measured.** The full model has 7,248,023,552 parameters; 1,087,201,280 (14.9999689184%) belong to the shared core, with zero added learned parameters. This is full-weight adaptation of a pretrained 7B model, not training a 7B model from scratch.

[S159](../evidence/bezier_nested_llm_7b/parameter_layout.json) · [S152](../evidence/bezier_nested_llm_7b/EXPERIMENT_REPORT.md)

**Measured.** On 128 swapped presentations per split, the trained baseline and shared model both reached 100% test accuracy and 99.22% longer-chain OOD accuracy; the independently executed core stayed at 50% on both. The balanced task is bounded contract consistency, not general code repair.

[S153](../evidence/bezier_nested_llm_7b/runs/pilot_report.json)

**Measured.** Validation selected a threshold above the bounded controller score range, producing no turns and exactly zero same-checkpoint steering gain. The intended reasoning-and-control capability was not achieved.

[S153](../evidence/bezier_nested_llm_7b/runs/pilot_report.json) · [S152](../evidence/bezier_nested_llm_7b/EXPERIMENT_REPORT.md)

**Reported.** The report records 44 passing tests and finite nonzero gradients in every core parameter tensor for each audited objective; this establishes implementation participation rather than useful reasoning.

[S152](../evidence/bezier_nested_llm_7b/EXPERIMENT_REPORT.md)

**Measured.** A later CUDA-graph check observed 72.21875 microseconds of cross-stream kernel overlap over four profiled checker/lookahead calls. Four timing samples gave serial/parallel medians 37.16/37.02 ms, insufficient to establish speedup.

[S154](../evidence/bezier_nested_llm_7b/runs/graph_overlap_check_v2/graph_overlap.json) · [S152](../evidence/bezier_nested_llm_7b/EXPERIMENT_REPORT.md)

## Limits and open questions

- One training seed and a narrow task with an outer-model accuracy ceiling.
- One endpoint-addressed MLP substitution; original attention/suffix order remains, with no arbitrary repeated traversal.
- Core position is a logical address, not physical GPU placement; shared weights still have compute and storage costs.

## Recommended next steps

- Recommended: prioritize core capability and harder continuation decisions before more routing complexity; follow-up distillation is implemented in v2.
- Recommended: compare joint training with a matched outer-only training arm and use fresh held-out programs across multiple seeds.

## Sources

- [Architecture and usage](../evidence/bezier_nested_llm_7b/README.md)
- [Completed experiment report](../evidence/bezier_nested_llm_7b/EXPERIMENT_REPORT.md)
- [Machine-readable scores and comparisons](../evidence/bezier_nested_llm_7b/runs/pilot_report.json)
- [CUDA graph overlap measurements](../evidence/bezier_nested_llm_7b/runs/graph_overlap_check_v2/graph_overlap.json)

## Related projects

- [bezier_nested_core_v4](bezier_nested_core_v4.md)
- [bezier_nested_llm_7b_v2](bezier_nested_llm_7b_v2.md)

Saved report dated November 24, 2026; consolidation reads completed artifacts, not live process state.

[Back to the project directory](README.md)
