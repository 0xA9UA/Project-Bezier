# Shared-core control with streamed GPU residency

`bezier_nested_llm_7b_v3` · Nested language models

Test useful conditional control and lower GPU residency while retaining the complete pretrained model.

## Saved status

Completed; lower GPU allocation and exact streamed parity demonstrated, with large latency costs; conditional control not useful.

## Work completed or attempted

- Relocated the logical shared core to layers 17-21 without adding learned weights.
- Completed guarded recovery training and fresh held-out program evaluation.
- Built synchronous CPU-backed outer-weight streaming with a persistent core and bounded block cache; measured exact output parity, allocation and latency.

## Findings

**Measured.** Recovery completed 512 updates after the earlier main attempt failed a retention guard at update 128. Validation selected Always Stay, so conditional control contributed zero accuracy/NLL gain.

[S169](../evidence/bezier_nested_llm_7b_v3/runs/final/summary/results_summary.json)

**Measured.** Fresh 512-example test accuracy improved from 53.52% to 63.67% with control disabled (+10.16 percentage points, paired 95% interval +5.66 to +14.65); OOD rose 53.71% to 58.59%. These are weight-adaptation gains, with no matched outer-only-training ablation.

[S169](../evidence/bezier_nested_llm_7b_v3/runs/final/summary/results_summary.json)

**Measured.** Streaming reduced measured peak CUDA allocation from 13.5974 to 3.4421 GiB (74.69%) for four 139-token prompts, while controlled-forward median increased from 79.661 to 1,208.119 ms (15.17x) and an eight-token generation probe became 29.16x slower.

[S170](../evidence/bezier_nested_llm_7b_v3/runs/final/engine/benchmark.json) · [S169](../evidence/bezier_nested_llm_7b_v3/runs/final/summary/results_summary.json)

**Measured.** Four resident/streamed routing cases matched logits, tokens and decisions bitwise; all eight generation tokens matched. The report records 42/42 CPU tests.

[S170](../evidence/bezier_nested_llm_7b_v3/runs/final/engine/benchmark.json) · [S168](../evidence/bezier_nested_llm_7b_v3/EXPERIMENT_REPORT.md)

**Measured.** Core-restoration accuracy and NLL intervals all included zero; fixed-panel whole-trace correctness declined from 60/64 to 58/64. The arithmetic generation smoke executed but answered incorrectly.

[S169](../evidence/bezier_nested_llm_7b_v3/runs/final/summary/results_summary.json) · [S168](../evidence/bezier_nested_llm_7b_v3/EXPERIMENT_REPORT.md)

## Limits and open questions

- Streaming is a general offloading mechanism, retains full CPU-backed model storage, and does not show a geometry-specific advantage.
- Synchronous single request; no KV cache or prefetch; every generated token recomputes context.
- One seed, fixed location and candidate pair; short runtime samples and desktop load limit timing generalization.

## Recommended next steps

- Recommended: treat low-residency streaming as a memory/latency tradeoff, not an inference-speed improvement.
- Recommended: evaluate resident shared execution separately; that is the v4 branch.

## Sources

- [Usage](../evidence/bezier_nested_llm_7b_v3/README.md)
- [Completed results](../evidence/bezier_nested_llm_7b_v3/EXPERIMENT_REPORT.md)
- [Consolidated measured results](../evidence/bezier_nested_llm_7b_v3/runs/final/summary/results_summary.json)
- [Final streaming benchmark](../evidence/bezier_nested_llm_7b_v3/runs/final/engine/benchmark.json)

## Related projects

- [bezier_nested_llm_7b_v2](bezier_nested_llm_7b_v2.md)
- [bezier_nested_llm_7b_v4](bezier_nested_llm_7b_v4.md)

Machine summary created November 25, 2026 03:06 UTC (November 24 in America/New_York).

[Back to the project directory](README.md)
