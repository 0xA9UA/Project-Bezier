# Recurrent node pilot

**The recurrent implementation passes its correctness and storage checks, but this pilot fails task accuracy.** Learned, matched fixed and shuffled recurrent routing all scored 0/128 on fresh test examples and 0/64 on longer programs. The same trained weights under dense inference scored 79/128 (61.72%) and 40/64 (62.50%). Every learned run exhausted the eight-turn budget; no early stopping occurred. This checkpoint is an architecture experiment with failed answer quality.

Starting point: selected v7 core. Final trained checkpoint after 64 joint updates.
Core: fixed center (0,0,0), radius 0.25, 1,087,201,280 shared parameters (15% budget).
All learned weights stay on one GPU; the core aliases the outer parameter storage.

## Training and correctness

Maximum 8 node turns; minimum 1. Continuation comparisons use 2 turns (clipped to remaining budget) from identical activations.
Routing and halting are supervised jointly with recurrent answer cross-entropy and dense preservation cross-entropy.
Training truncates gradients at the captured exploration prefix; outside-core parameters remain frozen.
Stopping compares immediate output against equal-horizon continuations with an explicit per-turn compute price.
Ground-truth programs and labels are used only by training/evaluation, never by model inference.

Training peak allocated VRAM: 18.560 GiB; peak reserved: 21.910 GiB.
Changed core tensors: 48; outside-core bitwise preservation and storage aliases passed.

## Fresh evaluation

Fixed and shuffled routing retain each prompt’s learned turn count and exact controller-call count. Their node/parameter-use budgets are checked per prompt. Dense inference uses its native full pass and is not compute matched.
The table's latency is synchronized wall time for one prompt, including controller decisions, output readout and final execution-history/reuse diagnostics, excluding tokenization and scoring. Clean inference timings are reported separately below.
All modes use the same final trained weights. Validation selects the constant fixed action only.
Parameter reuse below refers to outer node execution only. Reports separately include controller, mandatory interface, and total parameter-application counts. The arithmetic estimate counts parameter-token products and dense attention matmuls; normalization, indexing, gathers, and kernel overhead are approximated or excluded. It is not measured hardware FLOPs.

| Split | Mode | Accuracy | Mean turns | Outer parameter reuse | Median ms | Peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|
| test | learned | 0.00% | 8.00 | 33.28% | 105.57 | 13.652 |
| test | fixed | 0.00% | 8.00 | 39.38% | 104.43 | 13.651 |
| test | shuffled | 0.00% | 8.00 | 33.28% | 99.77 | 13.652 |
| test | dense | 61.72% | 32 layers | n/a | 31.45 | 13.570 |
| ood | learned | 0.00% | 8.00 | 33.85% | 106.61 | 13.652 |
| ood | fixed | 0.00% | 8.00 | 39.38% | 105.37 | 13.652 |
| ood | shuffled | 0.00% | 8.00 | 33.85% | 105.51 | 13.652 |
| ood | dense | 62.50% | 32 layers | n/a | 32.75 | 13.576 |

## Stopping and interpretation

test: stopping reasons {'max_budget': 128}; turn histogram {'8': 128}; 128/128 prompts reuse parameters.
Learned minus fixed: +0.00 percentage points (95% complete-program bootstrap CI [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% complete-program bootstrap CI [+0.00, +0.00]).
Learned minus dense: -61.72 percentage points (95% complete-program bootstrap CI [-70.31, -53.12]).
ood: stopping reasons {'max_budget': 64}; turn histogram {'8': 64}; 64/64 prompts reuse parameters.
Learned minus fixed: +0.00 percentage points (95% complete-program bootstrap CI [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% complete-program bootstrap CI [+0.00, +0.00]).
Learned minus dense: -62.50 percentage points (95% complete-program bootstrap CI [-73.44, -51.56]).

This is a small architecture pilot. It does not establish general language-model quality or a speed advantage. Reusing pretrained layer weights out of their original order is a substantial distribution change; accuracy and overhead are reported even when the recurrent model underperforms.

Full evidence: protocol.json, training.jsonl, correctness.json, weight_audit.json, validation.json, test_evaluation.json, ood_evaluation.json and complete.json.

## Correctness and storage-sharing checks

All 75 CPU tests passed, including the legacy v7 suite and 40 new recurrent tests. Checks cover common-state counterfactuals, equal node budgets, causal attention, padding and example isolation, in-sphere curves, reversing and revisiting node sets, immutable source states, learned/minimum/maximum stopping behavior, finite joint gradients and exact replay. [CPU test log](../recurrent_checks/cpu_tests.log)

All 64 updates produced finite gradients in all 48 core-bearing tensors. Those tensors changed. All 6,160,822,272 outside-core parameters remained bitwise unchanged, all original parameter-storage pointers were preserved, and all registered weights remained on CUDA. The core's 1,087,201,280 parameters are true views of that same storage. Its center and radius stayed fixed. [Weight audit](weight_audit.json) · [GPU correctness](correctness.json)

An independent audit recomputed corpus truth tables on all 100 input pairs, confirmed fresh-function exclusions and no evaluation/training group overlap, checked live and snapshotted source hashes, and recomputed the evaluation metrics and matched budgets. Every check passed. [Independent audit](independent_audit.json)

## Actual routing and parameter reuse

The 128 test prompts executed 1,024 turns: 935 continue, 71 redirect and 18 revisit actions. Eighteen distinct prompts explicitly revisited a curve, producing 18 exact repeated complete node sets. Attention source indices moved backward 319 times, and every prompt had a nonmonotonic source sequence. The 64 longer-program prompts executed 459 continue, 33 redirect and 20 revisit actions, with 20 exact repeated node sets and 153 backward attention-source transitions. These are actual node executions, without a mandatory layer progression or sphere-exit destination. [Trace analysis](independent_audit.json)

The learned test path executed 285,278,208 outer parameter applications per prompt, of which 33.28% repeated a previously used outer parameter. Norms are included, so this percentage is reported alongside explicit full-node-set revisits. Its eight core decisions added 8,697,610,240 parameter applications; mandatory input/output interfaces added roughly 134.5 million. Total applications were 9.117 billion versus 7.114 billion for a native dense pass. These counts and the approximate arithmetic proxy are recorded separately from measured latency. The repeated large core evaluations dominate this pilot's compute budget.

## Clean inference latency and VRAM

A separate fresh process loaded the final checkpoint and used four fixed 126-token validation prompts. Seven synchronized repetitions followed two warmups. Timed calls used `return_details=False`; tokenization, verification and final diagnostic summaries were excluded. Recurrent calls process prompts serially, while dense execution uses its ordinary batch implementation. These are four-prompt call latencies, not autoregressive token-generation latencies. [Complete runtime record](runtime.json)

| Mode | Median four-prompt call | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|
| Learned recurrent | 396.50 ms | 13.640 GiB | 13.678 GiB |
| Matched fixed routing | 385.06 ms | 13.640 GiB | 13.678 GiB |
| Matched shuffled routing | 409.93 ms | 13.640 GiB | 13.678 GiB |
| Native dense inference | 55.92 ms | 13.613 GiB | 13.648 GiB |

Independent GPU instrumentation observed exactly 32 chosen outer prompt-turn executions and 32 core calls. It observed zero unchosen outer executions. Replaying chosen actions reproduced the learned logits bitwise; fixed and shuffled paths matched all per-prompt node and controller budgets. Storage aliases and GPU residency passed again after timing. Training peaked at 18.560 GiB allocated and 21.910 GiB reserved.

The implementation meets the recurrent execution contract. This 64-update pilot provides no evidence of useful answer quality, effective early stopping or lower inference cost. Its bounded common activation interface permits execution and learning but does not recover pretrained transformer behavior under the new node order. The held-out results were not used to retune or select another checkpoint.

[Implementation and reproduction commands](../../RECURRENT.md) · [Trained core checkpoint](selected_core.pt)
