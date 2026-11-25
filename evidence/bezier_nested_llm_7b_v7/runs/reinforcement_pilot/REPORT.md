# Reinforcement learning and reward-gated parameter positions

**Reinforcement learning and successful-path contraction are implemented and trained, but this pilot did not learn useful conditional routing.** Every held-out prompt followed `continue → continue → stop` and answered B. Learned, matched fixed and shuffled routing all scored 50% on the balanced task. The same trained weights under dense inference scored 61.72% on 128 test examples and 48.44% on 64 longer-program examples. The learned rule stopped after two turns, rather than exhausting the eight-turn cap, but its route and answer were constant across inputs.

The pilot completed 256 on-policy joint updates from selected v7 weights.
Routing and stopping are trained with sampled returns and detached leave-one-out baselines; there are no supervised best-action targets.

The fixed core remains centered at (0,0,0), radius 0.25, with 1,087,201,280 shared parameters. All weights remain on GPU. Outside-core weights are frozen; successful outer node positions may move.

## Reinforcement learning

Each input has multiple sampled variable-length episodes from one identical starting state. The terminal reward combines full-vocabulary answer correctness, a small negative-log-likelihood signal and computation cost. Continue, redirect, revisit and stop are sampled from the policy. Leave-one-out returns form the baseline. One uniformly selected recorded decision, multiplied by trajectory length, estimates each trajectory score-gradient sum. Hidden transitions are detached; this is truncated policy-gradient training.
A second set of continuing-only sampled rollouts starts from the same captured state and consumes exactly the same turn/node budget. Its sampled rewards train routing with REINFORCE. Halting episodes are accounted separately with their actual computation cost. An entropy bonus encourages exploration. Selected outer-task and dense-task cross-entropy update the same core storage in the joint backward pass.

## Successful-path geometry

**Only a correct answer with positive total reward authorizes contraction.** Incorrect or nonpositive-reward paths do not move nodes. Co-executed outer Q/K/V/MLP nodes move closer together through bounded contraction. Core nodes, normalization and token-interface positions stay fixed. Each accepted move must reduce the independently recomputed selected-node dispersion. Per-update movement and total drift are capped; nodes cannot cross the core boundary or leave the sphere. The final layout is frozen for evaluation and saved with the weights.
Successful paths submitted for contraction: 1905. Layout updates reporting movement: 1905.

These were 1,905 qualifying **sampled** paths, including repetitions: 949 variable-length episodes and 956 equal-budget routing probes. They produced 5,513 accepted turn-level contractions; 156 proposals were rejected. Accepted contractions reduced the logged mean squared pairwise separation by a median 0.44785% per event. Across the final saved layout, 115,295 distinct outer node positions changed: 45,316 Q nodes, 11,321 K nodes, 11,334 V nodes and 47,324 MLP neurons. Core, normalization and token-interface positions did not move. The maximum change per update was bounded at 0.005; total drift was bounded at 0.05, allowing FP32 rounding tolerance. [Contraction audit](independent_audit.json) · [Final layout audit](layout_audit.json)

The accepted-event separation checks use the stored before/after measurements; the independent audit additionally regenerated the original full atlas and checked every final checkpoint coordinate. It confirmed core locking, shell membership, sphere bounds and the exact moved-node census. The new coordinates are routing metadata. Neural weight ownership, storage addresses and the 15% core budget did not change.

## Fresh held-out evaluation

Validation selects the best constant fixed routing action only. Fixed and shuffled paths retain each prompt’s learned turn and controller-call counts, using the identical final weights and layout. Dense inference is its native full pass, not compute matched. Latency includes final tracing diagnostics; scoring and tokenization are excluded.

| Split | Policy | Accuracy | Mean turns | Outer parameter reuse | Median ms | Peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|
| test | learned | 50.00% | 2.00 | 6.43% | 35.89 | 13.676 |
| test | fixed | 50.00% | 2.00 | 6.43% | 29.46 | 13.676 |
| test | shuffled | 50.00% | 2.00 | 6.43% | 30.09 | 13.676 |
| test | dense | 61.72% | 32 layers | n/a | 31.55 | 13.595 |
| ood | learned | 50.00% | 2.00 | 6.43% | 31.10 | 13.679 |
| ood | fixed | 50.00% | 2.00 | 6.43% | 35.45 | 13.679 |
| ood | shuffled | 50.00% | 2.00 | 6.43% | 35.43 | 13.679 |
| ood | dense | 48.44% | 32 layers | n/a | 33.13 | 13.601 |

## Stopping and reuse

Test: all 128 prompts stopped after two turns. The decisions were 256 continue actions and 128 stop actions, with no redirects, explicit revisits or repeated complete node sets. Stop decisions are taken from `stop_reason`; the raw execution-history histogram omits stop because it executes no outer nodes.
Learned minus fixed: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus dense: -11.72 percentage points (95% program-bootstrap interval [-20.31, -3.91]).
Longer programs: all 64 prompts stopped after two turns. There were 128 continue and 64 stop decisions, with no redirects, explicit revisits or repeated complete node sets.
Learned minus fixed: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus shuffled: +0.00 percentage points (95% program-bootstrap interval [+0.00, +0.00]).
Learned minus dense: +1.56 percentage points (95% program-bootstrap interval [-7.81, +12.50]).

## Verification

Training peak: 20.976 GiB allocated, 24.885 GiB reserved. All weights stayed on CUDA.
Changed core tensors: 48. Outside-core weights passed bitwise preservation; storage pointers and core aliases passed.
Ground-truth labels and verification are confined to training/evaluation. Inference executes only the selected path and never changes the learned layout. A small pilot does not establish useful routing or early stopping; those claims require the reported held-out results to support them.

Artifacts: protocol.json, training.jsonl, layout_training.jsonl, selected_core.pt, selected_layout.pt, layout_audit.json, weight_audit.json, correctness.json, validation.json, test_evaluation.json, ood_evaluation.json and complete.json.

All **113 CPU tests** passed. They include reward-to-policy-gradient sign, detached leave-one-out baselines, the trajectory-length score estimator, stochastic sampling, exact probability replay, equal-budget probes, stale-state rejection, correctness-and-positive-reward gating, contraction, transactional geometry loading, core locking and cache invalidation. [Test log](../reinforcement_checks/cpu_tests.log)

All **40 independent artifact checks** passed. Every training reward, baseline, advantage and sampled score coefficient matched its defining equation. Replayed action log probabilities had zero recorded error. Exactly 1,044 of the 4,096 sampled decision-score coefficients were nonzero; many paired trajectories earned equal returns. The audit matched every geometry update to a correct, positive-reward sampled path and verified corpus separation, frozen sources, checkpoints and held-out arithmetic. All 48 core-bearing tensors changed; all 6,160,822,272 outside-core neural weights remained bitwise unchanged. [Independent audit](independent_audit.json) · [Weight audit](weight_audit.json)

## Clean inference benchmark

A separate fresh process restored both final neural weights and the exact saved coordinates. Four 139-token validation prompts were timed with two warmups and seven synchronized repetitions using `return_details=False`. Tokenization, verification and final diagnostic summaries were outside the timer. Recurrent prompts execute serially; dense inference uses its ordinary batched implementation. These are whole four-prompt call latencies, not generated-token latency. [Runtime evidence](runtime.json)

| Policy | Median four-prompt call | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|
| Learned recurrent | 120.81 ms | 13.661 GiB | 13.701 GiB |
| Matched fixed | 139.18 ms | 13.661 GiB | 13.701 GiB |
| Matched shuffled | 138.06 ms | 13.661 GiB | 13.701 GiB |
| Native dense | 69.94 ms | 13.642 GiB | 13.674 GiB |

GPU instrumentation counted exactly eight outer prompt-turn executions and twelve core decisions, including the four stop decisions. It found zero unchosen outer executions. Forced replay reproduced learned logits bitwise. Layout coordinates, weight versions, storage pointers and GPU residency remained unchanged throughout inference. The core still shared the outer model's original parameter storage.

The test policy used approximately 3.467 billion parameter applications per prompt, including 3.262 billion from core decisions, versus 7.114 billion for native dense inference. Outer-parameter reuse was 6.43%, including normalization nodes, without any explicit revisits in greedy inference. Lower counted arithmetic did not translate into lower batched latency in this implementation. The small timing differences between identical learned/fixed/shuffled paths should not be interpreted as routing benefits.

The pilot establishes actual RL credit assignment and reward-gated geometry changes. It does **not** establish input-specific parameter selection or useful reasoning. Its 50% accuracy comes from a constant B answer on a balanced task, and fixed/shuffled routing reproduces its predictions exactly. The longer-program difference from dense has a confidence interval spanning zero. There was no policy, weight or layout retuning using held-out results.

[Implementation and commands](../../REINFORCEMENT.md) · [Trained core](selected_core.pt) · [Learned layout](selected_layout.pt)
