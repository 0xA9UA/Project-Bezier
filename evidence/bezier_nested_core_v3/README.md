<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Revision 3: larger models with a 5% shared core

This revision scales the whole central core along with the outer model. The core stays at the origin inside radius 0.25. Its complete weights and biases are shared by outer computation, initial-curve control, redirect ranking, and direct reasoning. There is no additional hidden central expert or learned routing head.

| Configuration | Core dimensions | Core parameters | Total shared-model parameters | Core fraction |
| --- | --- | ---: | ---: | ---: |
| Small comparison | 64 → 71 → 64 | 9,223 | 183,239 | 5.0333% |
| Larger model | 256 → 458 → 256 | 235,210 | 4,704,970 | 4.9992% |

The small difference from exactly 5% comes from whole-number layer widths. The private-controller control deliberately adds another complete core. All shared arms add zero checker weights relative to their corresponding clean outer model. Extra computation, buffers, and runtime memory are measured separately.

Outer execution blends the two nearest region centers. The core has a fixed central parameter address, but can be selected while the curve is outside radius 0.25. That radius is not a hard execution gate; expert selection and geometric sphere entry are separate measurements.

## Run and resume

The verified Python environment is reused from `WORKSPACE\bezier_nested_core\.venv`. This revision does not modify either earlier experiment or its results.

```powershell
# Verify environment and tests, then execute the fixed seven-run study.
& "WORKSPACE\bezier_nested_core_v3\scripts\run_experiment.ps1"

# Resume the same protocol after interruption.
& "WORKSPACE\bezier_nested_core_v3\scripts\run_experiment.ps1" -Resume
```

The study uses the local RTX 5090 and does not launch cloud jobs or upload data. Its protocol is defined in `scripts/run_scaleup.py` and both JSON configurations. The initial bounded matrix contains small clean/shared models on seed 7, large clean/shared models on seeds 7 and 19, and the large private-controller control on seed 7: seven training runs total.

## Training changes

Each run performs 2,256 optimizer updates:

1. **500 core-only updates:** direct reasoning, batch 256, peak learning rate 0.003. Outer parameters receive no gradients during this phase.
2. **256 outer warmup updates:** outer answer prediction plus core reasoning, batch 64, routing checks disabled.
3. **1,500 joint updates:** outer prediction, core reasoning, and supervised continuation comparisons, batch 64. Outer execution alternates 8 and 16 region visits.

Both the clean and controlled arms receive the same reasoning and outer warmups. Controlled arms label eight training examples every second joint update, with one uniformly sampled eligible checkpoint per labelled minibatch. The depth schedule rotates so both depths receive the same number of supervised updates. This gives 750 control updates and 6,000 labelled activations per controlled run. Each run receives 240,384 direct-reasoning example presentations and 112,384 outer-answer presentations.

Teacher snapshots refresh every 25 joint updates and at the transition to joint training. Stay and four redirect continuations start from exactly the same detached application activation, with the same remaining expert-call budget. Continuations use FP32 arithmetic to preserve small loss differences. Inference evaluates only the chosen continuation.

Raw gains remain measured in nats. Training divides predicted and true gains by 0.01 before a Huber regression and soft ranking loss. The control objective has weight 0.1, ramped in over 128 joint updates. The total joint loss is outer answer CE + direct-core reasoning CE + weighted control loss. Gradients train both the shared core and the live outer representation; labels remain detached.

Learning rates use phase-relative warmup and cosine decay. Their horizons are fixed independently of the requested stopping step, preserving exact resume semantics. Atomic checkpoints include model, teacher, optimizer, RNG states, source hashes, and exposure counts.

## Data and selection

The expanded frozen corpus contains **1,024 training programs covering 256 distinct four-input Boolean functions**: four syntactic variants and all sixteen input assignments per function. It has 16,384 training examples. Equal numbers of variants prevent frequently generated functions from dominating sampling. Short one-instruction programs supplement the original 2–4 instruction examples. Commuting operands and double negation generate equivalent syntax; every serialized answer is independently verified.

The original 32 validation, 32 test, and 32 longer-program test files are preserved byte-for-byte. Each split contains 512 balanced examples. No training truth table overlaps any heldout truth table. These are reused benchmark holdouts, not a newly sequestered confirmation set.

Initial reasoning-only probes compared learning rates on training and validation data. Longer direct-core training began to overfit, motivating a bounded warmup. The probe files are in `runs/scaling_diagnostics/`. Test and OOD results did not choose the recipe.

During joint training, full validation runs select `best.pt` by the sum of outer prediction NLL and direct-core reasoning NLL. The first joint update and each 250th joint update are eligible. `final.pt` always preserves all requested training updates even if validation selects an earlier state. Validation also selects a routing margin from a fixed nine-value grid. Test and OOD labels never choose checkpoints or margins.

Direct reasoning reads only parameter-free syntax features and the complete core. It cannot borrow the outer embedding, recurrent encoder, or shell experts. The syntax adapter does not execute Boolean operators or supply intermediate answers.

## Evaluation and evidence

Every study run evaluates all 512 examples in each heldout split, at 8 and 16 visits. Reports distinguish direct-core accuracy, complete-model accuracy, NLL, local controller regret, actual turns, and region coverage. Paired confidence intervals resample whole heldout programs. They describe program uncertainty within a seed, not uncertainty across training seeds.

Controller regret evaluates all test examples at the first eligible checkpoint only; it does not establish control quality throughout every route. CUDA overlap, latency, and memory are audited separately using the larger seed-7 model. A same-checkpoint, same-policy single-stream reference isolates scheduling effects from retraining differences.

Useful artifacts:

- `runs/scaleup/study.json`: exact matrix and completion status.
- `runs/scaleup/report.md`: combined measured results and limitations.
- `runs/scaleup/training_curves.png`: validation trajectories, when available.
- `runs/scaleup/*/suite.json`: complete per-group evidence.
- `runs/scaleup/*/seed*/ARM/best.pt`: validation-selected inference checkpoint.
- `runs/scaleup/*/seed*/ARM/final.pt`: fully completed training state.
- `runs/scaleup/*/seed*/ARM/validation.jsonl`: checkpoint selection evidence.
- `runs/scaleup/*/seed*/ARM/source/`: exact package sources used for that run.

This bounded study does not execute the full original seven-arm, multi-seed research protocol. The previous pilot changed in capacity, core fraction, data, duration, optimization, and selection, so a comparison against that pilot does not isolate scaling alone. The matched small/large comparisons in this revision hold the training recipe and corpus constant while scaling both the core and outer model.

Related primary research motivating staged learning and careful routing measurements includes [Automated Curriculum Learning](https://proceedings.mlr.press/v70/graves17a.html) and [Switch Transformers](https://arxiv.org/abs/2101.03961). This experiment does not implement either paper's complete algorithm, and those papers do not validate this architecture.
