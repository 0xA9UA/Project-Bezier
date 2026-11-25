# Reinforcement learning and reward-gated parameter positions

<!-- CONTINUATION_MEASURED_FINDINGS:START -->
## Measured continuation findings

Completed **256 additional updates**, from 256 to **512 total**. The parent’s trained weights and learned coordinates were restored exactly. The parent had no saved optimizer/RNG state, so this first continuation used fresh Adafactor moments and a fresh rollout RNG.

Held-out learned accuracy is **50.00%** (64/128) on test and **50.00%** (32/64) on longer OOD programs. The deployed policy remains constant: `continue → continue → stop`, always predicting B. This is chance-level accuracy on the balanced binary task, with no evidence of useful input-dependent routing.

| Split | Learned | Matched fixed | Matched shuffled | Native dense |
|---|---:|---:|---:|---:|
| test | 50.00% (64/128) | 50.00% (64/128) | 50.00% (64/128) | 53.91% (69/128) |
| ood | 50.00% (32/64) | 50.00% (32/64) | 50.00% (32/64) | 56.25% (36/64) |

Fixed and shuffled routing use each prompt’s exact learned turn, node, parameter and controller-call budgets. They test routing at that stopping budget; they do not isolate the causal benefit of learned halting. Dense inference uses a different native computation budget.

The parent and continued run use different held-out programs. Their accuracy differences are unpaired and cannot be presented as a paired improvement.

### Geometry: cumulative versus this continuation

**66,118 node coordinates differ from the parent checkpoint.** Relative to the original deterministic atlas, 15,081 previously unchanged nodes moved, and 0 returned to their original position. The final cumulative count away from original positions is **130,376** (parent: 115,295). These are different counts; cumulative movement is not all new movement.
This segment supplied 1,656 correct, positive-reward sampled trajectories (including repeated episodes and matched-route probes, not unique programs) and 1,656 path updates with movement. Accepted contractions increased by 4,417; the cumulative total is 9,930. Rejected contractions in this segment: 0.
Accepted events establish a local decrease in the selected-node dispersion at that event. They do not establish global final-layout clustering or an accuracy benefit from geometry changes.
Core and mandatory interface coordinates remain fixed. The core stays at (0,0,0), radius 0.25, with 1,087,201,280 shared parameters and unchanged ownership.

This training segment took 484.45 seconds and peaked at 20.972 GiB allocated / 24.219 GiB reserved VRAM.

**140 tests passed; 57 independent audit checks passed.** All weights remain on GPU and core views alias the outer storage. Outside-core neural weights remain bitwise unchanged.

The original evaluation tables below include final trace-summary work. The clean forward benchmark appended at the end measures ordinary `return_details=False` inference.

test: stop reasons `{'learned_stop': 128}`; turn counts `{'2': 128}`; outer parameter reuse 8.14%. Reuse includes mandatory normalization; explicit revisit actions and repeated full node sets are recorded separately.
Learned minus fixed: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).

ood: stop reasons `{'learned_stop': 64}`; turn counts `{'2': 64}`; outer parameter reuse 8.14%. Reuse includes mandatory normalization; explicit revisit actions and repeated full node sets are recorded separately.
Learned minus fixed: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).

[Independent audit](independent_audit.json) · [Measured evidence summary](measured_summary.json)
<!-- CONTINUATION_MEASURED_FINDINGS:END -->

This run continued the parent trained weights and layout for 256 updates: 512 total updates. Parent did not save optimizer state; new optimizer on resumed weights/layout.
Routing and stopping are trained with sampled returns and detached leave-one-out baselines; there are no supervised best-action targets.

The fixed core remains centered at (0,0,0), radius 0.25, with 1,087,201,280 shared parameters. All weights remain on GPU. Outside-core weights are frozen; successful outer node positions may move.

## Reinforcement learning

Each input has multiple sampled variable-length episodes from one identical starting state. The terminal reward combines full-vocabulary answer correctness, a small negative-log-likelihood signal and computation cost. Continue, redirect, revisit and stop are sampled from the policy. Leave-one-out returns form the baseline. One uniformly selected recorded decision, multiplied by trajectory length, estimates each trajectory score-gradient sum. Hidden transitions are detached; this is truncated policy-gradient training.
A second set of continuing-only sampled rollouts starts from the same captured state and consumes exactly the same turn/node budget. Its sampled rewards train routing with REINFORCE. Halting episodes are accounted separately with their actual computation cost. An entropy bonus encourages exploration. Selected outer-task and dense-task cross-entropy update the same core storage in the joint backward pass.

## Successful-path geometry

**Only a correct answer with positive total reward authorizes contraction.** Incorrect or nonpositive-reward paths do not move nodes. Co-executed outer Q/K/V/MLP nodes move closer together through bounded contraction. Core nodes, normalization and token-interface positions stay fixed. Each accepted move must reduce the independently recomputed selected-node dispersion. Per-update movement and total drift are capped; nodes cannot cross the core boundary or leave the sphere. The final layout is frozen for evaluation and saved with the weights.
Successful paths submitted for contraction: 1656. Layout updates reporting movement: 1656.

## Fresh held-out evaluation

Validation selects the best constant fixed routing action only. Fixed and shuffled paths retain each prompt’s learned turn and controller-call counts, using the identical final weights and layout. Dense inference is its native full pass, not compute matched. Latency includes final tracing diagnostics; scoring and tokenization are excluded.

| Split | Policy | Accuracy | Mean turns | Outer parameter reuse | Median ms | Peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|
| test | learned | 50.00% | 2.00 | 8.14% | 30.98 | 13.689 |
| test | fixed | 50.00% | 2.00 | 8.14% | 31.33 | 13.689 |
| test | shuffled | 50.00% | 2.00 | 8.14% | 30.06 | 13.689 |
| test | dense | 53.91% | 32 layers | n/a | 31.81 | 13.608 |
| ood | learned | 50.00% | 2.00 | 8.14% | 36.21 | 13.689 |
| ood | fixed | 50.00% | 2.00 | 8.14% | 36.61 | 13.689 |
| ood | shuffled | 50.00% | 2.00 | 8.14% | 30.45 | 13.689 |
| ood | dense | 56.25% | 32 layers | n/a | 33.01 | 13.613 |

## Stopping and reuse

test: {'learned_stop': 128}; turns {'2': 128}. Routes: {'action_counts': {'continue': 256, 'redirect': 0, 'revisit': 0, 'stop': 128}, 'prompts_with_explicit_revisit': 0, 'exact_repeated_node_sets': 0}.
Learned minus fixed: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus dense: -3.91 percentage points (95% program-bootstrap interval [-12.50, +3.91]).
ood: {'learned_stop': 64}; turns {'2': 64}. Routes: {'action_counts': {'continue': 128, 'redirect': 0, 'revisit': 0, 'stop': 64}, 'prompts_with_explicit_revisit': 0, 'exact_repeated_node_sets': 0}.
Learned minus fixed: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus dense: -6.25 percentage points (95% program-bootstrap interval [-15.62, +4.69]).

## Verification

Training peak: 20.972 GiB allocated, 24.219 GiB reserved. All weights stayed on CUDA.
Changed core tensors: 48. Outside-core weights passed bitwise preservation; storage pointers and core aliases passed.
Ground-truth labels and verification are confined to training/evaluation. Inference executes only the selected path and never changes the learned layout. A small pilot does not establish useful routing or early stopping; those claims require the reported held-out results to support them.

Artifacts: protocol.json, training.jsonl, layout_training.jsonl, selected_core.pt, selected_layout.pt, layout_audit.json, weight_audit.json, correctness.json, validation.json, test_evaluation.json, ood_evaluation.json and complete.json.

## Continuation and exploration

Parent: `WORKSPACE\bezier_nested_llm_7b_v7\runs\reinforcement_pilot`. The behavior policy mixes 15.0% uniform exploration over legal actions with the learned categorical policy. Sampling and gradient replay use exactly this same mixture. Greedy inference is unchanged. Illegal stops stay masked. The success gate remains correctness AND positive reward.

Optimizer and RNG states are saved, with resumable checkpoints every 64 updates. Batched geometry consolidation preserves sequential update order, movement caps and contraction checks.

<!-- CONTINUATION_CLEAN_RUNTIME_AND_AUDIT:START -->
## Clean inference benchmark

Measured on NVIDIA GeForce RTX 5090: **4 prompts**, padded context shape `[4, 140]`, 2 warmups and 7 timed calls per mode. Tokenization, scoring and final diagnostic summaries are excluded; recurrent history required by the policy remains active.
Per-prompt figures divide a four-prompt call by four. Recurrent prompts execute serially; dense inference uses its native batched pass. These figures are not isolated single-prompt latency. All modes use the same final weights and learned layout.

| Mode | Median 4-prompt call ms | Amortized ms/prompt | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| learned | 143.54 | 35.88 | 13.673 | 13.705 |
| fixed | 123.64 | 30.91 | 13.673 | 13.705 |
| shuffled | 137.19 | 34.30 | 13.673 | 13.705 |
| dense | 70.09 | 17.52 | 13.654 | 13.674 |

Learned, fixed and shuffled executed identical chosen paths in this measured fixture. Their timing differences are not evidence of a routing benefit.

Independent GPU instrumentation observed 8 chosen outer executions and 12 controller calls, matching recorded turns and decisions. Unchosen outer executions: 0. Forced replay of the chosen learned path is bitwise equal. Exact saved coordinates, parameter pointers/versions, core aliases and GPU residency remain unchanged throughout inference.

## Independent correctness and continuation audit

140 tests passed. The independent CPU audit passed all 57 checks, covering 2,048 sampled stopping episodes and 2,048 matched route probes. It checked 4,096 replayed mixture distributions, their legal supports and probability floors, reward arithmetic, leave-one-out baselines, geometry gates, movement bounds, saved optimizer/RNG provenance, source/checkpoint links and independently recomputed final metrics.
Correct, positive-reward paths comprised 825 full episodes and 831 route probes. The success gate was checked independently; the absence of a geometry change on failed or nonpositive-reward paths is part of that audit.
Variable-length sampled-episode actions only (excluding the separate matched-route probes): `{'continue': 4014, 'redirect': 543, 'revisit': 652, 'stop': 1942}`. Sampled exploration demonstrates training coverage; it does not establish conditional greedy inference.
The first continuation has fresh optimizer moments because its parent saved none. This run saves optimizer and RNG state for subsequent continuations. Resumption begins a new training segment with recorded ancestry exclusions; an exact interrupted batch-cursor replay is not claimed.

[Clean runtime JSON](runtime.json) · [Independent audit JSON](independent_audit.json) · [Test log](../continuation_checks/cpu_tests.log)
<!-- CONTINUATION_CLEAN_RUNTIME_AND_AUDIT:END -->

## GPU restoration check for the next continuation

All 18 additional restoration checks passed in a fresh GPU process. The 48 optimizer slices and 85 moment tensors restored exactly, with floating moments in FP32 on CUDA and optimizer views aliasing the original outer storage. Saved CPU/CUDA Torch, rollout-generator, Python and NumPy random states reproduced their draws exactly. No training step occurred; weights, geometry and storage pointers remained unchanged. The optimizer records 256 steps from this continuation, while the weight lineage records 512 total training updates. [GPU restoration evidence](resume_check.json)
