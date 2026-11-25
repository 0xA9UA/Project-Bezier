<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Reinforcement learning of parameter selection and layout

<!-- CONTINUATION_LATEST_SUMMARY:START -->
The latest run completed **256 more reinforcement updates (512 total)** from the parent’s learned weights and layout. Held-out learned accuracy is **50.00%** (64/128) on test and **50.00%** (32/64) on longer OOD programs. The deployed policy remains constant: `continue → continue → stop`, always predicting B. This is chance-level accuracy on the balanced binary task, with no evidence of useful input-dependent routing.

66,118 node coordinates changed since the parent; 130,376 are displaced cumulatively from the original atlas. All 140 tests and 57 independent audit checks passed. The central core, parameter ownership and GPU storage sharing remain fixed.

[Continued run: measured results](runs/reinforcement_continued/REPORT.md) · [Clean inference benchmark](runs/reinforcement_continued/runtime.json) · [Independent continuation audit](runs/reinforcement_continued/independent_audit.json)

The first continuation used fresh optimizer state because the original pilot had not saved it. The parent and continued evaluations use different holdouts; cross-run accuracy changes are unpaired.

## Original pilot and implementation
<!-- CONTINUATION_LATEST_SUMMARY:END -->

The completed 256-update pilot moved 115,295 outer node coordinates through 1,905 correct, positive-reward sampled paths. Core coordinates and shared GPU storage passed all checks. However, the final policy always chose `continue → continue → stop` and answered B: 50% accuracy, identical to matched fixed/shuffled routing. This is not evidence of useful input-specific parameter selection. [Full measured report](runs/reinforcement_pilot/REPORT.md)

This variant trains the recurrent core with on-policy REINFORCE. It samples continue, redirect, revisit and stop actions, executes the resulting paths, and adjusts their probabilities using measured task reward. It replaces the earlier supervised routing-target experiment; that experiment and its frozen artifacts remain available for comparison.

For each training input, independent rollouts start from the same activation. Each rollout's reward uses the other rollouts' average reward as its leave-one-out baseline. The reward is:

`correct_answer - 0.05 * full_vocabulary_NLL - 0.02 * normalized_computation`

Computation includes selected outer nodes, actual controller calls and the final readout. An entropy bonus encourages exploration. Separate continuing-only rollouts compare routes from an identical captured activation under exactly equal turn and parameter budgets. Variable-length stopping episodes receive their actual computation penalty. No best-action labels or supervised routing targets enter the policy loss.

The cost is a parameter-token computation proxy, not measured hardware FLOPs.

To fit the resident 7B model on one GPU, the policy-gradient estimator replays one uniformly sampled decision per trajectory, multiplied by that trajectory's decision count. It detaches the captured recurrent state. This estimates the trajectory score-gradient sum while truncating differentiation through earlier hidden transitions. Selected recurrent-task and dense-task cross-entropy also train the same shared core in one backward pass.

## Moving successful paths closer together

The movement rule is **correct answer AND positive total reward**, as requested. A failed path or a nonpositive reward cannot move any node.

For an eligible path, the outer Q/K/V and MLP nodes used together on each turn contract toward their common centroid. A proposal is accepted only if its recomputed mean squared pairwise separation decreases. Maximum displacement is 0.005 per update and 0.05 from the original position. These limits and a weak pull toward the original layout prevent unconstrained collapse. Nodes stay within the sphere and outside the central core.

All core coordinates, token-interface positions and normalization positions remain bitwise fixed. The core stays centered at `(0,0,0)`, radius `0.25`, with the original 1,087,201,280 shared weights. The model's neural weights remain on GPU and the core still references the outer model's exact storage. The learned positions are geometry metadata saved separately in `selected_layout.pt`; weight ownership and the core's 15% budget do not change.

Geometry changes happen after the joint weight update, using the collected successful-path reward. They invalidate cached node selections. Inference loads the frozen layout, executes only its selected path, and does not receive correctness labels or change positions.

## Train and evaluate

```powershell
Set-Location 'WORKSPACE'
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v7/scripts/reinforcement_pilot.py --output bezier_nested_llm_7b_v7/runs/reinforcement_pilot_new --steps 256
```

Use a new output directory. The pilot saves both weights and geometry, exact training rewards and sampled action probabilities, contraction audits, a source snapshot, and fresh held-out evaluation. Prior recurrent evaluation functions are excluded. Validation chooses the fixed routing baseline only; the final RL checkpoint and geometry are frozen before test evaluation.

Fixed and shuffled routing use the same weights and learned geometry, with exact per-prompt node and controller budgets matched to the learned policy. Dense inference uses its native full pass and is reported with its own computation cost. Learned stopping effectiveness, explicit revisits, task accuracy, memory and latency must be judged from the measured results.

## Continue from a trained checkpoint

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v7/scripts/continue_reinforcement.py --parent bezier_nested_llm_7b_v7/runs/reinforcement_continued --output bezier_nested_llm_7b_v7/runs/reinforcement_next --steps 256
```

The continuation restores both learned weights and geometry. Its default training behavior mixes 15% uniform exploration over legal actions with the learned policy; sampling and gradient replay use exactly the same mixture. Greedy inference remains unchanged. Geometry updates preserve the original success gate, original anchor positions, fixed core and displacement limits.

Each continuation saves optimizer and RNG state, with intermediate checkpoints every 64 updates. A completed run or an intermediate checkpoint directory may be passed as `--parent`. Continuation selects training groups excluded from its recorded ancestry where enough groups remain; this is a new training segment, not an exact replay of an interrupted data cursor. The first 256-update pilot did not save optimizer or RNG state, so its first continuation restores weights/layout with a fresh optimizer. Subsequent continuations restore the saved optimizer and RNG.

Use `applications/continuation_generate.py` and `scripts/continuation_benchmark.py` with `--run runs/reinforcement_continued` (relative to this project directory) for checkpoints created by this runner. Both validate the frozen sources and saved layout. Choose fresh output filenames.

## Use the original RL pilot weights and layout

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v7/applications/reinforcement_generate.py --run bezier_nested_llm_7b_v7/runs/reinforcement_pilot --prompt 'Evaluate (12 + 5) * 2. A: 34 B: 29 Answer:' --max-new-tokens 8 --output bezier_nested_llm_7b_v7/runs/reinforcement_generation.json
```

The application verifies the checkpoint and layout checksums, restores the exact saved coordinates, and checks that inference leaves weights, positions and storage unchanged. It uses no verifier or answer labels. Generation is bounded to eight new tokens and recomputes the context without a KV cache.

To measure clean inference calls separately from evaluation diagnostics:

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v7/scripts/reinforcement_benchmark.py --run bezier_nested_llm_7b_v7/runs/reinforcement_pilot --output bezier_nested_llm_7b_v7/runs/reinforcement_runtime_new.json
```

- [RL pilot report](runs/reinforcement_pilot/REPORT.md)
- [Training rewards and policy diagnostics](runs/reinforcement_pilot/training.jsonl)
- [Successful-path contractions](runs/reinforcement_pilot/layout_training.jsonl)
- [Layout audit](runs/reinforcement_pilot/layout_audit.json)
- [Correctness test log](runs/reinforcement_checks/cpu_tests.log)
- [Independent RL and layout audit](runs/reinforcement_pilot/independent_audit.json)
- [Clean inference benchmark](runs/reinforcement_pilot/runtime.json)
- [Recurrent execution architecture](RECURRENT.md)
