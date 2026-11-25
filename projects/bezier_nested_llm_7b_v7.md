# Optimized node routing, recurrence and reinforcement learning

`bezier_nested_llm_7b_v7` · Nested language models

Improve routing execution and input-specific decisions, then explore arbitrary recurrent node reuse, learned stopping and reward-gated coordinate adaptation.

## Saved status

Completed stage pilot, recurrent pilot, RL pilot and continuation to 512 updates; implementation and resume audits pass, useful recurrent reasoning/input-specific control not demonstrated.

## Work completed or attempted

- Optimized interval grouping and route-plan caching while sharing the original BF16 weight store.
- Completed a supervised stage-ordered controller pilot with constrained shared-head readout fitting; gradient-trained checkpoints were evaluated but not selected.
- Built arbitrary recurrent global MLP-node/attention-region execution with continue, redirect, revisit and stop; completed a 64-update pilot.
- Implemented policy-gradient routing and stopping with reward-gated successful-path coordinate contraction; completed a 256-update pilot and 256-update continuation.
- Added persisted optimizer/RNG state, exact restoration audits, clean runtime benchmarks and independent artifact audits.

## Findings

**Measured.** Warm forced-route execution improved over v6 by 1.48x to 9.37x on tested 2/4-prompt layouts; for four 140-token mixed-route prompts, median fell 1,451.79 to 154.96 ms. V7 dense inference still took 69.41 ms. These supplied-action benchmarks exclude learned decisions.

[S224](../evidence/bezier_nested_llm_7b_v7/runs/benchmark/comparison.json) · [S213](../evidence/bezier_nested_llm_7b_v7/REPORT.md)

**Measured.** Validation selected a constrained two-row readout fit on original v6 weights, discarding the 256 reasoning/64 replay gradient-updated candidates. The selected stage policy scored 59.24% test versus 60.42% fixed and 60.81% shuffled; OOD was 64.32% versus 64.58% fixed and 65.36% shuffled.

[S225](../evidence/bezier_nested_llm_7b_v7/runs/pilot/training_summary.json) · [S226](../evidence/bezier_nested_llm_7b_v7/runs/pilot/test_evaluation.json) · [S227](../evidence/bezier_nested_llm_7b_v7/runs/pilot/ood_evaluation.json) · [S213](../evidence/bezier_nested_llm_7b_v7/REPORT.md)

**Measured.** Stage-ordered learned routing missed its registered accuracy criterion. Its selected runtime was 147.74 ms versus 95.18 ms fixed and 69.63 ms dense on four 140-token prompts; 35 CPU tests passed, while the arithmetic generation smoke answered incorrectly.

[S228](../evidence/bezier_nested_llm_7b_v7/runs/selected_runtime.json) · [S213](../evidence/bezier_nested_llm_7b_v7/REPORT.md)

**Measured.** The 64-update arbitrary recurrent pilot scored 0/128 test and 0/64 OOD for learned/fixed/shuffled paths, while same-weight dense inference scored 61.72%/62.50%. Every learned prompt exhausted eight turns; explicit backward traversal and revisits worked but useful answer quality did not.

[S229](../evidence/bezier_nested_llm_7b_v7/runs/recurrent_pilot/REPORT.md) · [S230](../evidence/bezier_nested_llm_7b_v7/runs/recurrent_pilot/independent_audit.json)

**Measured.** The initial 256-update RL pilot learned a constant continue → continue → stop path and constant B answer: 50% on balanced test/OOD, identical to fixed/shuffled predictions. It moved 115,295 outer node positions through 1,905 eligible sampled paths; 113 tests and 40 independent checks passed.

[S231](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_pilot/REPORT.md) · [S232](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_pilot/independent_audit.json)

**Measured.** Latest saved continuation completed 256 further updates (512 total). Learned test/OOD accuracy stayed 50% (64/128 and 32/64), again constant B with the same two-turn path. Same-weight dense scores were 53.90625%/56.25%; parent and continuation holdouts differ, so cross-run changes are unpaired.

[S217](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/measured_summary.json)

**Measured.** The continuation changed 66,118 coordinates since its parent; 130,376 were displaced from the original atlas. Core coordinates and locked interfaces stayed fixed. It recorded 1,656 new successful paths and 4,417 accepted contraction events; 140 tests and 57 independent audit checks passed.

[S217](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/measured_summary.json) · [S233](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/independent_audit.json)

**Measured.** Clean continuation runtime for four 140-token prompts was 143.54 ms learned versus 70.09 ms dense, with peak allocations 13.673 versus 13.654 GiB. Lower parameter-application counts did not yield lower batched latency.

[S234](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/runtime.json) · [S217](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/measured_summary.json)

**Measured.** A fresh GPU process passed 18 restoration checks for 48 optimizer slices, 85 moment tensors and saved RNGs, without a training step. The first continuation used a fresh optimizer because the parent saved none; subsequent restoration is now supported.

[S223](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/resume_check.json) · [S216](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/REPORT.md)

## Limits and open questions

- The stage-ordered, recurrent and RL paths are distinct workstreams; their scores and compute budgets must not be pooled.
- Arbitrary pretrained-node reordering is a substantial representation change; execution correctness does not preserve pretrained quality.
- RL exploration and coordinate motion occurred during training, but latest greedy inference remains constant and chance-level.
- Recurrent prompts execute serially; dense benchmarks use native batching. Parameter-application proxies are not measured hardware FLOPs.
- No KV cache; one narrow task family and small recurrent holdouts.

## Recommended next steps

- Recommended: diagnose the recurrent representation/answer collapse before adding training duration or spatial complexity.
- Recommended: preserve stage-ordered results separately and require nonconstant, held-out gains versus budget-matched fixed/shuffled controls.
- Recommended: use saved optimizer/RNG state for further authorized continuation, with a fresh uninspected holdout.

## Sources

- [Project overview](../evidence/bezier_nested_llm_7b_v7/README.md)
- [Stage-ordered experiment](../evidence/bezier_nested_llm_7b_v7/REPORT.md)
- [Recurrent architecture](../evidence/bezier_nested_llm_7b_v7/RECURRENT.md)
- [Latest RL status and reproduction](../evidence/bezier_nested_llm_7b_v7/REINFORCEMENT.md)
- [Latest continuation report](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/REPORT.md)
- [Latest measured status](../evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/measured_summary.json)

## Related projects

- [bezier_nested_llm_7b_v6](bezier_nested_llm_7b_v6.md)
- [bezier_nested_llm_7b_v8](bezier_nested_llm_7b_v8.md)

Latest project documentation updated November 25, 2026. Latest saved RL continuation is 512 total updates; this is an artifact snapshot, not a live-process claim.

[Back to the project directory](README.md)
