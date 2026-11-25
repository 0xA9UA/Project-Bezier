# Node-level spatial routing

`bezier_nested_llm_7b_v6` · Nested language models

Make Bézier coordinates select actual hidden computational nodes in a shared 7B model and test repeated route redirection.

## Saved status

Completed revised pilot and deployed verification; geometry causally changes computation, conditional advantage over fixed/shuffled routes unresolved.

## Work completed or attempted

- Built an atlas covering 987,136 nodes; 147,185 core nodes own the fixed 15% parameter budget.
- Implemented node-budgeted MLP and attention projection gathers, coordinate-sensitive dispatch, two controller checkpoints, and selected-path generation.
- Corrected a forward-distance degeneracy before held-out testing, completed the revised pilot, and independently reran all held-out prompts through actual deployed dispatch.

## Findings

**Measured.** Coordinates now demonstrably affect computation: swapping selected/unselected MLP node coordinates changed selected sets and logits by up to 0.125; restoring coordinates restored logits bitwise. Full-density greedy tokens matched dense inference in the audit, with maximum BF16 logit difference 0.0625.

[S203](../evidence/bezier_nested_llm_7b_v6/runs/runtime_selected.json) · [S201](../evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md)

**Measured.** The first sampler produced identical late route node sets; the revised forward window changed all 60 later stage groups after Stay. Revised training produced 255/256 nonzero second-checkpoint targets in its first 64 updates. The failing attempt is retained.

[S206](../evidence/bezier_nested_llm_7b_v6/runs/summary/route_set_audit.json) · [S208](../evidence/bezier_nested_llm_7b_v6/runs/summary/route_set_audit_initial.json) · [S209](../evidence/bezier_nested_llm_7b_v6/runs/pilot_v2/train_first64_diagnostics.json)

**Measured.** Revised training completed 128 reasoning plus 32 replay updates. Actual deployed test accuracy was 66.4063% learned, 59.3750% spatial Stay, 66.0156% validation-best fixed and 65.6250% shuffled, across 256 examples.

[S210](../evidence/bezier_nested_llm_7b_v6/runs/pilot_v2/training_summary.json) · [S202](../evidence/bezier_nested_llm_7b_v6/runs/deployed_verification.json) · [S201](../evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md)

**Measured.** Learned-minus-Stay test gain was +7.031 percentage points [95% CI +1.172, +12.891], but learned-minus-fixed was only +0.391 [-0.781, +1.953]. OOD learned and fixed both scored 59.3750%; useful input-specific control remains unproven.

[S202](../evidence/bezier_nested_llm_7b_v6/runs/deployed_verification.json) · [S201](../evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md)

**Measured.** All held-out examples took action 2 at the first checkpoint. Learned and fixed predictions differed on only three test examples and none on OOD, so a constant route reproduced most of the measured benefit.

[S207](../evidence/bezier_nested_llm_7b_v6/runs/summary/control_diagnostics.json) · [S201](../evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md)

**Measured.** Actual deployed verification supersedes cached-tree metrics: cached/output equality and NLL tolerance checks were false despite matched actions. Learned and shuffled dispatch had some prediction/NLL differences; no policy was reselected.

[S202](../evidence/bezier_nested_llm_7b_v6/runs/deployed_verification.json)

**Measured.** On two 31-token prompts, dense/Stay/learned median latency was 23.244/97.237/119.918 ms. Full weights remained resident, another core view used zero additional CUDA bytes, and all outside-core elements were preserved. The report records 33 CPU tests.

[S203](../evidence/bezier_nested_llm_7b_v6/runs/runtime_selected.json) · [S211](../evidence/bezier_nested_llm_7b_v6/runs/pilot_v2/weight_audit.json) · [S201](../evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md)

## Limits and open questions

- Density 0.98 with fixed coordinates; dense causal attention and transformer layer dependencies remain.
- Mandatory input/output vocabulary interfaces are scheduled spatially rather than sparsified.
- One small pilot, no KV cache, no measured total-VRAM advantage versus a single dense model; transient gathers add overhead.
- One temporary applications directory was inaccessible during the index scan; primary reports and result artifacts were readable.

## Recommended next steps

- Recommended: optimize grouping/selection overhead without weakening parity checks; v7 records that work.
- Recommended: retain deployed metrics as authoritative and require learned gains over fixed and shuffled paths, not only naive Stay.

## Sources

- [Usage and node-routing entry point](../evidence/bezier_nested_llm_7b_v6/README.md)
- [Architecture, corrected geometry and results](../evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md)
- [Authoritative deployed-path check](../evidence/bezier_nested_llm_7b_v6/runs/deployed_verification.json)
- [Selected-checkpoint runtime](../evidence/bezier_nested_llm_7b_v6/runs/runtime_selected.json)

## Related projects

- [bezier_nested_llm_7b_v5](bezier_nested_llm_7b_v5.md)
- [bezier_nested_llm_7b_v7](bezier_nested_llm_7b_v7.md)

Saved report updated November 25, 2026; deployed verification explicitly supersedes cached-tree measurements.

[Back to the project directory](README.md)
