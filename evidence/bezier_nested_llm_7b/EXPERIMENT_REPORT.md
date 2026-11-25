<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Pretrained 7B shared-core pilot

Validation selected a policy that makes no turns on the test set. This pilot does not demonstrate a benefit from inference-time rerouting. The independent core scored 50.00%, at chance on this balanced task. The intended reasoning-and-control capability of the core has not been achieved.

## Architecture and task

The base is [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), pinned to `c170c708c41dac9275d15a8fff4eca08d52bab71`. It contains **7,248,023,552** learned parameters. The fixed central core contains **1,087,201,280** of those same parameters (**14.9999689184%**); **zero learned parameters are added**. Its token embedding, language head and norms are included in the budget. Center `(0,0,0)` and radius `0.25` are logical addresses, not physical GPU placement.

The task is bounded Python patch review: choose A or B to satisfy an ordered contract. Each label is checked by independent contract/AST interpreters over all 100 legal input pairs. Test and OOD each contain 64 complete programs / 128 swapped presentations, with candidate truth functions disjoint from training and validation. OOD uses longer operation chains. This measures bounded contract consistency, not general code repair.

The pilot supports one endpoint-addressed Bézier MLP substitution, with position, tangent, bend and live activation supplied to the core. The two candidates are preset spatial neighbors. A chosen endpoint resolves the executed MLP; a continuous tail shares the outer exit. Attention order and suffix order remain fixed; the alternative MLP is also used again at its ordinary later position. There is no repeated steering, arbitrary attention traversal or dynamic core entry yet.

## Held-out results

Accuracy is forced A/B choice. Raw accuracy uses the unrestricted full-vocabulary greedy token. NLL is conditional on A/B. Turns are per prediction, with at most one.

| Model/policy | Test accuracy | Longer-chain OOD | Test raw accuracy | Test A/B NLL | Test turns |
|---|---:|---:|---:|---:|---:|
| Pretrained Mistral | 67.19% | 54.69% | 67.19% | 2.044 | 0.000 |
| Trained baseline | 100.00% | 99.22% | 100.00% | 4.07e-07 | 0.000 |
| Shared model, steering off | 100.00% | 99.22% | 100.00% | 9.313e-09 | 0.000 |
| Shared model, steering on | 100.00% | 99.22% | 100.00% | 9.313e-09 | 0.000 |
| Independent shared core | 50.00% | 50.00% | 50.00% | 0.6976 | 0.000 |

The fixed validation-selected gain threshold is **1.00179 nats**. The controller's estimate is bounded to ±0.1 nats; fitting targets are clipped, while measured true gains/regret remain unclipped. Validation uses the same constrained A/B probability objective. An explicit never-turn option is available.

## Paired comparisons

Positive gains favor the shared model with steering. Confidence intervals resample complete programs (10,000 draws), preserving swapped pairs. They do not estimate variation across training seeds. Improvement over a separately trained baseline can result from control training; only the same-checkpoint comparison isolates the effect of switching steering on.

| Split | Reference | Accuracy gain | Paired 95% interval | A/B NLL gain, nats |
|---|---|---:|---:|---:|
| test | trained_baseline | +0.00 pp | [+0.00, +0.00] pp | +0.00000 |
| test | same_checkpoint_disabled | +0.00 pp | [+0.00, +0.00] pp | +0.00000 |
| test | pretrained_stock | +32.81 pp | [+27.34, +38.28] pp | +2.04374 |
| ood | trained_baseline | +0.00 pp | [-2.34, +2.34] pp | -0.07538 |
| ood | same_checkpoint_disabled | +0.00 pp | [+0.00, +0.00] pp | +0.00000 |
| ood | pretrained_stock | +44.53 pp | [+40.62, +47.66] pp | +2.53414 |

## Training actually completed

Both arms start from the same immutable pretrained weights and use seed 7, the same balanced batches, and 512 joint updates. Each batch contains two whole swapped program pairs. Learning rate 2e-6 ramps over 32 updates. Core reasoning weight is 0.25 and control weight 0.1. There is no isolated core warmup. The control arm cycles decision layers 8/16/24; primary evaluation uses layer 8. Examples are sampled with replacement. This is full-weight adaptation, not a new 7B pretraining run.

| Arm | Updates | Outer/core examples | Counterfactual labels | Training wall time | Peak allocated GPU |
|---|---:|---:|---:|---:|---:|
| baseline | 512 | 2048 | 0 | 476.0 s | 22.81 GiB |
| shared | 512 | 2048 | 2048 | 525.3 s | 23.12 GiB |

Alternative training continuations share the exact activation and remaining compute budget. Total training compute differs between arms because the shared arm additionally runs the controller and two counterfactual suffixes. The optimizer uses factored state and per-leaf in-backward updates; one combined backward accumulates all shared uses before an update. BF16 stochastic rounding is enabled. No adapter weights, second teacher, full FP32 master model, gradient accumulation or activation checkpointing are used.

## Checks

- 44 tests passed, including CUDA serial/concurrent parity, true parameter aliases, exact core budget, selected-only execution, endpoint-dependent MLP dispatch, sphere containment, tangent continuity, detached-checker gradients, paired supervision, optimizer updates, checkpoint resume, corpus isolation and metric correctness.
- The full 7B preflight completed 16 joint updates at batch 4, peaking at 21.84 GiB. The initial discarded preflight exposed a control-scale problem and weak one-token attention; both were corrected before freezing the pilot.
- Full-size serial/concurrent maximum logit difference was `0.0`. Native full-sequence HF head versus last-token-only head differed by at most `0.03125` in BF16 because matrix shapes can select different kernels.
- Each full-size gradient audit found nonzero gradients in every core parameter tensor for outer prediction, independent core prediction and control. This is tensor-level participation, not a claim that every scalar changes in every batch.
- Trained checkpoint files are checksummed, package/corpus hashes match training, and reload probes require exactly equal core/outer/steered logits before held-out evaluation.
- The final test/OOD results were not used to revise hyperparameters. This is a one-seed bounded pilot; broad reasoning, longer text generation and arbitrary sphere traversal remain unvalidated.

## Concurrent inference check

The original eager CUDA-stream implementation produced equal predictions but no observed kernel overlap. Independent CUDA graph replay subsequently produced **72.21875 microseconds of actual cross-stream kernel overlap across four profiled checker/lookahead calls**, with zero overlap in the serial graph negative control. Original and changed-input logits, scores and choices matched exactly. This demonstrates limited simultaneous GPU execution, not a general speedup or effective routing.

Four timing samples gave median full-forward latencies of 37.16 ms for serial graph execution and 37.02 ms for parallel graph execution. This is too small a sample to support a speedup claim. Graph capture adds setup time and fixed-shape buffers. No learned weights are added; frozen training sources and accuracy results remain unchanged. See `runs/graph_overlap_check_v2/graph_overlap.json` and the optional `--graph-concurrent` CLI mode.

## Artifacts and reproduction

The next scientific bottleneck is core capability, not fitting the 7B model into memory. Both outer models reached the task's accuracy ceiling while core performance stayed at chance. A follow-up should test core-specific distillation from the full model and harder continuation comparisons, with fresh held-out cases and multiple seeds. These are proposed next experiments, not benefits demonstrated by this pilot.

- `runs/pilot_report.json`: complete scores, paired comparisons and training summaries.
- `runs/pilot_protocol.json`: frozen settings, source/base/corpus identities and comparison definitions.
- `runs/pilot_shared/checkpoint_u000512`: trained shared model and optimizer/RNG state.
- `runs/pilot_baseline/checkpoint_u000512`: independently trained baseline.
- `runs/eval_shared/evaluation.json`: complete per-program measurements.
- `runs/eval_shared/validation_threshold.json`: validation-only margin selection.
- `runs/pilot_results.png`: training curves and accuracy comparisons.
- `README.md` and `docs/use_case.md`: architecture limits, commands and single-example usage.

Original pretrained weights and all earlier toy experiments remain available. This experiment adds no learned model parameters, but inference compute, training memory and checkpoint storage still have costs.
