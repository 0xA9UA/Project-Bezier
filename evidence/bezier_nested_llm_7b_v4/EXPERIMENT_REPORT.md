# Resident shared-core controller: completed pilot

**The shared parameter and resident-runtime implementation passes its checks. Useful conditional control is not established.** Learned routing reaches 66.70% test accuracy versus 65.82% with the same weights and control disabled, but its paired confidence interval crosses zero; shuffled routing scores higher and longer-program OOD accuracy falls slightly. Training, the five-task final suite, and secondary diagnostics are complete.

The goal is to improve the full outer model through activation-dependent choices made by a smaller shared core. Standalone core language-model accuracy is not the objective. This revision keeps all weights on the GPU and gives the core a separate execution context that references the same parameter storage.

## Model and exact parameter contract

The starting point is the completed [v3 recovery checkpoint](WORKSPACE/bezier_nested_llm_7b_v3/runs/controller_recovery/best_u000512). Its [local configuration](WORKSPACE/bezier_nested_llm_7b_v3/runs/controller_recovery/best_u000512/config.json) identifies a standard Mistral decoder: 32 layers, hidden width 4,096, MLP width 14,336, grouped-query attention, and untied vocabulary embeddings/head. This experiment uses BF16 weights and the original vocabulary.

| Quantity | Exact value |
|---|---:|
| Full model | 7,248,023,552 parameters |
| Shared core | 1,087,201,280 parameters |
| Core fraction | 14.999968918% |
| Outside-core parameters | 6,160,822,272 |
| Added learned parameters | 0 |
| Logical core center / radius | `(0, 0, 0)` / `0.25` |

The fixed core comprises layers 17–21, with five attention/norm blocks, four complete MLPs, and 14,061 neurons of the fifth MLP. Its interfaces are two existing embedding rows, two existing output-head rows, and the shared final norm. The ordinary outer forward still uses every full block and vocabulary row.

`CoreExecutionView` is a separate execution object that references these exact slices. It is not another registered model or a copy of the parameters. Geometry and core membership are fixed; parameter placement is not learned.

## What the controller actually does

The [TRAIN-only probe](WORKSPACE/bezier_nested_llm_7b_v4/runs/initial_probe/probe.json) compared four candidate locations on 64 programs/256 examples. It selected decision layer 26: continue through normal MLP 28, or substitute nearby outer MLP 27 or 29. All three lie outside the core. The selected pair's training oracle accuracy was 77.34%, versus 64.84% for its best constant action: 12.50 percentage points of diagnostic headroom, not an achieved controller gain.

After layer 26, the core receives the full detached activation sequence and two shared query/geometry tokens. It scores both candidate destinations using the same core parameters. The highest score exceeding the frozen validation threshold selects an alternate; otherwise the model stays on its normal path. One selected MLP runs after the common lookahead. Its input comes from the ordinary application layer's normalization; the chosen destination supplies the MLP weights. Attention and all remaining layers retain their original order.

Each proposal uses a cubic Bézier segment with position, direction, bend, and destination features. The endpoint resolves to a concrete MLP address. This is **one fork and one MLP substitution per forward**, not unrestricted traversal, repeated steering, or learned spatial placement. Both candidates are evaluated during training to obtain supervision; inference runs only its chosen outer continuation.

## Training and preservation

The [main run](WORKSPACE/bezier_nested_llm_7b_v4/runs/accuracy_main/training_summary.json) completed 1,365 optimizer updates: 1,024 reasoning/control updates and 341 replay updates. It took 790.08 seconds and peaked at 18.97 GiB allocated GPU memory. The [configuration](WORKSPACE/bezier_nested_llm_7b_v4/configs/accuracy_core.json) records learning rate `5e-7`, warmup, decay, loss weights, threshold grid, and retention guards.

Only the exact core slices are optimized; the remaining approximately 85% is frozen. Outcome supervision rewards rescuing a wrong full-vocabulary answer and penalizes spoiling a correct answer. A bounded probability-loss term handles correctness ties. Ordinary outer answer loss and initial-logit preservation train those same shared core weights for continued use by the outer model. Old-task replay occurs after every three reasoning updates. BF16 stochastic rounding and optimizer handles that alias the permitted slices support these updates without an independent trainable controller.

The frozen [teacher cache](WORKSPACE/bezier_nested_llm_7b_v4/runs/initial_probe/teacher_manifest.json) stores initial-model full-vocabulary BF16 logits for 4,096 reasoning TRAIN examples and 512 replay TRAIN examples on CPU. Small target batches are transferred as needed. There is no second teacher model resident on the GPU; CPU target storage is distinct from CPU weight offloading.

The [gradient audit](WORKSPACE/bezier_nested_llm_7b_v4/runs/accuracy_main/gradient_audit.json) found finite, nonzero gradients in all 48 core slice groups. The final [checkpoint comparison](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/weight_sharing.json) found changed values in all 48 core-bearing tensors and verified **all 6,160,822,272 outside-core elements bitwise unchanged**, including interface-row and partial-MLP complements. This establishes update scope, not usefulness of every scalar. Frozen outside weights do not imply fixed outer outputs: the outer path also uses the trained core.

Complete shuffled program passes replace sampling with replacement. The completed run covered all 1,024 reasoning groups exactly once: 4,096 unique examples. Replay used 682 example exposures across 341 group draws, covering all 256 groups/512 unique examples and beginning a second pass.

## Selection amendment: preserve the original result

The original validation rule selected reasoning update 1,024 / total update 1,365. Its 64.0625% validation accuracy tied the saved update-768 checkpoint, and its lower full-vocabulary NLL won the tie-break. Its threshold 1.0 produces **Always Stay**, with zero conditional gain.

A secondary rule was adopted after inspecting validation outcomes and before final test/OOD inference. It restricts selection to saved checkpoints that pass all guards and whose active policy has positive validation accuracy gain over both disabled control and the best validation-selected fixed action. This was **not preplanned** and must not be presented as the original selection protocol. The [frozen derived summary](WORKSPACE/bezier_nested_llm_7b_v4/runs/control_selection.json) retains the unchanged original summary's hash, the original/default selection, the secondary rule, and the candidate records.

The sole eligible saved active candidate, [reasoning update 768 / total update 1,024](WORKSPACE/bezier_nested_llm_7b_v4/runs/accuracy_main/best_r000768_u001024), is selected for final evaluation. Its threshold is `−0.005`; validation accuracy is 64.0625%, a gain of 1.7578 percentage points over both controls, with alternate actions on 50.195% of examples. These are selection-set observations, not final evidence of conditional generalization.

| Exposure scope | Reasoning/replay updates | Reasoning examples | Replay examples |
|---|---:|---:|---:|
| Completed training run | 1,024 / 341 | 4,096 | 682 |
| Selected checkpoint at update 768 | 768 / 256 | 3,072 | 512 |

The selected checkpoint has seen 768 distinct reasoning groups, not the full 1,024-group reasoning pass. Its replay exposure completes one 256-group pass. The derived artifact verifies these counts against the saved checkpoint state and records its hashes separately from completed-run totals.

## Data and final comparison definitions

The [new corpus](WORKSPACE/bezier_nested_llm_7b_v4/corpus_eval/manifest.json) freezes 128 validation programs, 256 test programs of 3–4 stages, and 256 OOD programs of 5–6 stages. Every program has four counterbalanced input/answer-order variants. Independent opcode and bounded AST interpreters verify all 100 inputs. Prior v2/v3 final and intermediate functions, old patch correct/foil functions, and all one/two-stage final functions are excluded. No new training programs were added. A separate inspection ledger records final-test use while leaving the creation manifest immutable.

Accuracy means the unmodified full-vocabulary top token equals the correct answer. Negative log likelihood is secondary. Controls use the same selected weights: disabled control, every fixed action, the validation-selected best fixed action, and groupwise shuffled scores executed through actual mixed dispatch. The shuffle matches answer labels/variant slots and preserves the exact action histogram. Initial-versus-selected comparisons also include weight adaptation.

Paired uncertainty uses complete programs, keeping their four variants together, with 10,000 bootstrap draws and seed 2026. These intervals are not adjusted for multiple comparisons. Previously inspected patch examples measure retention rather than fresh generalization.

## Final quality

The completed [initial evaluation](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/initial/evaluation.json), [selected evaluation](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/selected/evaluation.json), and [computed summary](WORKSPACE/bezier_nested_llm_7b_v4/runs/summary/results_summary.json) agree:

| Full-vocabulary accuracy | Test, 1,024 examples | OOD, 1,024 examples |
|---|---:|---:|
| Initial model, disabled | 65.9180% | 58.1055% |
| Selected model, disabled | 65.8203% | 58.0078% |
| Selected model, learned control | 66.6992% | 57.7148% |
| Validation-best fixed action: Stay | 65.8203% | 58.0078% |
| Matched shuffled scores | 67.0898% | 57.8125% |
| Always use MLP 27 | 65.5273% | 55.6641% |
| Always use MLP 29 | 67.5781% | 57.1289% |

| Paired accuracy gain, percentage points [95% CI] | Test | OOD |
|---|---:|---:|
| Learned minus disabled / validation-best fixed | +0.879 [−0.879, +2.637] | −0.293 [−1.270, +0.586] |
| Learned minus shuffled | −0.391 [−2.051, +1.270] | −0.098 [−1.465, +1.270] |
| Selected disabled minus initial disabled | −0.098 [−1.660, +1.367] | −0.098 [−1.855, +1.660] |
| Learned minus initial disabled | +0.781 [−1.367, +2.930] | −0.391 [−2.344, +1.660] |

The controller is active, choosing Stay/27/29 **535/0/489** times on test and **878/0/146** on OOD. It never selects MLP 27 on these splits. Active decisions alone do not demonstrate useful conditional reasoning: learned routing does not outperform shuffled scores, and the main intervals cross zero. Fixed MLP 29 has the highest test point estimate, but it was not selected on validation and cannot retrospectively replace that control.

Test full-vocabulary NLL worsens from 0.598506 disabled to 0.604065 learned; OOD worsens from 0.652257 to 0.653170. Their paired intervals also cross zero. Initial-versus-selected accuracy is essentially flat. All arms retain 100% accuracy on the 128 previously inspected replay examples; this is retention evidence, not fresh generalization.

The [secondary diagnostics](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/control_diagnostics.json) count 55 test rescues and 46 spoils versus disabled control. On A-labelled answers there are 3 rescues/46 spoils; on B-labelled answers, 52/0. Turn rates are similar across labels (48.05% A, 47.46% B), so this is not evidence of a router that simply follows gold labels, but the outcome strongly tilts toward B. All-four-variants-correct falls from 25.39% to 24.61%, a −0.781 percentage-point change [−4.297, +2.734]. OOD has 17 rescues/20 spoils. Routing may rebalance answer preference; that is an observational interpretation, not a causal finding or demonstrated semantic-reasoning gain. Shuffle counts match globally, by label, and by variant slot.

Diagnostic definitions were frozen before results. A disclosed compatibility adapter restored verified raw answer choices omitted by the model-facing corpus projection; it changed no metric, policy, or permutation. The adapter passed [a real-corpus check](WORKSPACE/bezier_nested_llm_7b_v4/runs/cpu-diagnostic-compat-tests.log) in addition to the 49 main CPU tests.

![Completed shared-core quality, actions, and resident memory](WORKSPACE/bezier_nested_llm_7b_v4/runs/summary/summary.png)

## Final resident runtime and memory

The [selected-checkpoint resident check](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/resident/resident.json) uses the frozen policy, an RTX 5090, BF16, and four validation prompts padded to 141 tokens. These are final measurements, distinct from the earlier preflight.

| Measurement | Result |
|---|---:|
| Unique full-model weight storage | 13.50050 GiB |
| Logical core views within that storage | 2.02507 GiB |
| Additional CUDA bytes creating a core execution view | **0** |
| Peak allocated / reserved CUDA memory | 13.6964 / 14.2070 GiB |
| Disabled / serial controlled median | 70.724 / 90.377 ms |
| Separate-stream controlled median | 90.411 ms |
| Controlled / disabled latency | 1.278× |
| Serial/separate-stream outputs and actions | Bitwise equal |
| Observed simultaneous GPU kernel overlap | None |

An independently loaded copy of these core weights would add 2.02507 GiB for weights alone. Sharing avoids that duplicate allocation; this is an arithmetic comparison, not a measured independent-controller baseline. The complete outer model still occupies GPU memory, and controller activations cost additional memory. There is no CPU weight offloading, quantization, or KV cache. Separate execution contexts and CUDA streams exist, but measured kernel overlap does not; no parallel-kernel speedup is established. Three short repetitions are not a serving benchmark, and allocator figures differ from driver/display VRAM.

The [final suite](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/suite.json) completed five tasks; [49 CPU tests passed](WORKSPACE/bezier_nested_llm_7b_v4/runs/cpu-final-tests.log). The [eight-token generation smoke](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/generation.json) completed but answered `B: 29` to `(12 + 5) × 2`, whose correct choice was `A: 34`. Execution passed; arithmetic correctness did not. General prompt ability remains unvalidated.

## Interpretation and next hypotheses

This pilot establishes exact shared storage, enforced update scope, and working conditional dispatch. It does **not** establish a smarter outer model through conditional control. One seed, fixed geometry/organization, one intervention, narrow task coverage, the selection amendment, and no matched outer-only-training ablation limit causal claims. The original update-1,024 selection was not included in final evaluation, so these results do not compare the amended checkpoint against it.

Two source-level hypotheses merit separate future ablations, **neither implemented nor proven here**. First, weighted BCE divides by `weights.sum().clamp_min(1)`: eight tied candidate targets weighted 0.1 produce 0.8 times mean CE, not a 0.1 multiplier. Fixed-element-count normalization or skipping all-tie batches could isolate that effect. Second, the same A/B head rows serve answer prediction and control; a candidate-independent answer preference may dominate the score. Candidate-minus-Stay readout at the same context might cancel that offset using existing weights, with additional control compute. Further model or data selection now requires a new holdout: these final results have been inspected.

## Reproduction

Run from `WORKSPACE/bezier_nested_llm_7b_v4` with the existing verified probe/cache and new output directories. The first command reproduces the completed training configuration; the second repeats evaluation of the already frozen selected policy under the original selection identity, not a new blind test. Do not rerun the selection amendment or `finalize.py` on this inspected corpus: the inspection ledger requires the existing selection hash.

```powershell
& WORKSPACE/Bezier-Think/.venv/Scripts/python.exe WORKSPACE/bezier_nested_llm_7b_v4/scripts/train.py --config WORKSPACE/bezier_nested_llm_7b_v4/configs/accuracy_core.json --output WORKSPACE/bezier_nested_llm_7b_v4/runs/accuracy_reproduction_new
& WORKSPACE/Bezier-Think/.venv/Scripts/python.exe WORKSPACE/bezier_nested_llm_7b_v4/scripts/evaluate.py --model WORKSPACE/bezier_nested_llm_7b_v4/runs/accuracy_main/best_r000768_u001024 --selection-json WORKSPACE/bezier_nested_llm_7b_v4/runs/final/selection.json --policy-json WORKSPACE/bezier_nested_llm_7b_v4/runs/control_selection.json --output WORKSPACE/bezier_nested_llm_7b_v4/runs/selected_reproduction_new
```
