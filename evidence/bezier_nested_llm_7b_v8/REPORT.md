<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Conditional routing on fresh arithmetic programs

**The trained core recovered five more answers than fixed routing on the 768-prompt test and 13 more on the 384 longer prompts. The uncertainty intervals include zero.** On the longer programs, shuffling the learned decisions among programs recovered one more answer than matching each decision to its own prompt. These results show opportunities for redirection while leaving its input-specific accuracy benefit unresolved. The application defaults to fixed (2, 2), the policy selected before either test set was scored.

| Policy | Test, 768 prompts | Longer programs, 384 prompts |
|---|---:|---:|
| V8 learned core | 490 · 63.80% | 237 · 61.72% |
| Validation-selected fixed (2, 2), same weights | 485 · 63.15% | 224 · 58.33% |
| Learned actions shuffled across programs | 481 · 62.63% | 238 · 61.98% |
| Previous v7 controller on its original weights | 484 · 63.02% | 229 · 59.64% |
| Dense outer model on fitted weights | 495 · 64.45% | 238 · 61.98% |
| Dense outer model on original v7 weights | 495 · 64.45% | 243 · 63.28% |

| Learned accuracy difference in percentage points, with paired 95% interval | Test | Longer programs |
|---|---:|---:|
| Versus fixed | +0.65 [−0.65, +1.95] | +3.39 [−0.52, +7.29] |
| Versus shuffled | +1.17 [−0.65, +2.99] | −0.26 [−4.43, +3.91] |
| Versus previous v7 | +0.78 [−1.04, +2.60] | +2.08 [−2.08, +6.25] |

The primary test required both lower bounds against fixed and shuffled routing to exceed zero. Neither cleared that threshold. The full dense predictor ranked above the learned policy on both splits. [Test predictions and paired intervals](runs/pilot/test_evaluation.json) · [Longer-program predictions and paired intervals](runs/pilot/ood_evaluation.json)

## Which redirects helped

On the test set, the learned controller deviated from the fixed pair on 131 of 768 prompts (17.1%) and used all nine action pairs. It rescued 17 answers and spoiled 12. On longer programs it deviated on 191 of 384 prompts (49.7%), used seven action pairs, rescued 35 and spoiled 22. These totals count answers whose full-vocabulary correctness changed under live dispatch.

| Comparison with fixed routing | Test | Longer programs |
|---|---:|---:|
| Correct answer A, rescues / spoils | 5 / 9 | 23 / 3 |
| Correct answer B, rescues / spoils | 12 / 3 | 12 / 19 |
| Inputs correct in both A/B display orders, learned | 161 / 384 | 72 / 192 |
| Inputs correct in both A/B display orders, fixed | 156 / 384 | 64 / 192 |

The dominant answer-position effect reversed across the two splits. The same-input, both-orders measure improved on both. The independent audit joined results by program and variant, checked each greedy prediction against its gold token, and reproduced the rescue and spoil counts. [Independent paired audit](runs/checks/paired_audit.json)

## Where the controller loses available gains

On the 256 validation prompts, enabling only the first checkpoint reduced the correct count from 164 to 161. Enabling only the second checkpoint left all decisions at the fixed route and answered 164 correctly. With both enabled, the first stage chose 41 alternatives and the second chose eight, bringing the count back to 164. With access to the gold answer, choosing among the recorded branches could reach 213 correct by changing only the first action, 175 by changing only the second, and 232 with both available. The largest opportunity is selecting the first action reliably. The 232 figure is a diagnostic ceiling that requires the correct answers. [Checkpoint ablation and route ceilings](runs/checks/stage_diagnostics.json)

The prediction application also ran the plain-language arithmetic example in the README. The released fixed policy replied B. The expected response was A because `(7 + 2) × 3 mod 10 = 7`. This off-template example was separate from the pilot estimates. [Generated response](runs/checks/generation.json)

## Experimental question

Can the central 15% of this Mistral model recognize when changing the outer model's Bézier path improves its next-token answer? The primary comparison gives learned and fixed routing the same weights and the same node count per outer stage. A second control takes all the learned action pairs from one program and applies them to another, matching each of the four prompt variants. A gain against this control tests whether a decision helps the particular prompt that produced it. The learned controller also incurs the cost of two five-layer core passes per prompt.

## Paired training

Every training program contributes two queries. They use the same pair of possible result digits and have opposite correct answers under a fixed ordering of those digits. Each query appears twice with A and B exchanged. Comparing the two queries at the same option order tests sensitivity to the problem. Comparing the two orders of the same query tests sensitivity to the answer presentation. The fit rewards the observed differences in route utility for both comparisons. The gold answer, its position and other examples from the program are absent from the inference interface.

For each of the three first actions, training tries all three second actions from exactly the same intervening activation. The target adds full-vocabulary greedy correctness to 0.35 times the probability assigned to the correct next token. The second-stage fit estimates the utility of choosing each continuation. First-stage training splits whole programs in two, fits a second-stage policy on each half, and uses its decisions on the other half to score the available first actions. These out-of-fold decisions use zero cutoffs. The final controller's two cutoffs can differ after calibration. [Trainer](scripts/pilot.py) · [Value targets and calibration](scripts/action_values.py)

The inherited language-model weights supply the learned core features. This pilot fits six 1,024-element row slices of the first central MLP's existing down projection, totaling 6,144 scalars. The direct core and the outer model read the same physical slices. The rest of the 1,087,201,280-parameter core supplies its existing computation. A dense-path accuracy and loss check measures the impact of the writeback on the outer model. [Physical readout](model.py)

The protocol allocates 768 programs to the fit and 192 separate training programs to calibrate the cutoffs. The initial 3,072 fit prompts supply 27,648 completed route outcomes. After the first writeback, the protocol refreshes 192 of those programs, or 6,912 outcomes, to capture the shared rows' effects on the features and branch predictions. The second fit uses these refreshed programs and the original measurements on the other 576. Its second-stage folds exclude the target program from the corresponding regression. The shared weights used for refreshed measurements have already seen all fit programs. Validation and both test sets use the final stored rows.

## Frozen evaluation

The validation, test and longer-program sets contain 64, 192 and 96 programs. The corpus builder independently verifies each program's outputs for all 100 possible digit inputs. It excludes 5,350 earlier final and intermediate truth functions when selecting fresh final programs. Reusing an arithmetic operation or a prefix remains possible. Reported test differences have paired percentile intervals using 10,000 bootstrap draws of complete programs, with four prompt variants kept together. The shuffle comparison uses one seeded permutation and its interval conditions on that assignment. [Corpus verification](corpus_eval/verification.json)

The fit chooses its default actions and two margins using separate portions of the training corpus. Validation selects the best of all nine fixed routes, with negative log likelihood breaking accuracy ties. Releasing the learned policy requires that its actual validation accuracy exceed both fixed and shuffled decisions and that the dense predictor stay within its accuracy and loss tolerances. The registered test criterion asks whether both lower bounds for the learned policy's improvements exceed zero. Results for the learned candidate are reported regardless of the released policy.

The strongest fixed pair on validation was (2, 2), also the learned policy's default. Both answered 164 of 256 prompts correctly (64.06%). The learned controller redirected 41 prompts, rescuing six answers and spoiling six. Shuffling the decisions answered 162 correctly. All cached validation actions agreed with live dispatch. Validation therefore selected **fixed (2, 2)** for the prediction application's default. Dense accuracy on the fitted rows was 163/256, down five answers from 168/256 on the warm weights. Its mean negative log likelihood improved from 0.6172 to 0.6161. Both changes fell within the registered tolerances. [Frozen selection](runs/pilot/selection.json) · [Validation predictions](runs/pilot/validation.json)

Re-evaluating the 6,912 training routes after the first writeback changed 120 greedy correctness outcomes (1.74%). The second fit's stored score scales were 0.02949 and 0.07993. Calibration selected margins of 0.003448 and 0.009643 in those scaled units. The final weight audit found changes in one central tensor, verified all 6,160,822,272 outside-core scalars bitwise and confirmed that all GPU weight storage addresses remained stable. [Fit measurements](runs/pilot/fit.json) · [Final weight audit](runs/pilot/weight_audit.json)

## Node coordinates and GPU storage

The atlas accounts for all 7,248,023,552 parameters exactly once across 987,136 spatial nodes, with the central 1,087,201,280 parameters inside radius 0.25. The BF16 model occupied 14,496,047,104 unique weight bytes on the GPU. The controller aliases those tensors and adds zero persistent weight bytes. Compared with storing a second copy of the same core in BF16, sharing avoids 2.03 GiB of weights. The engine also reported 19.77 MiB of metadata at initialization and uses transient weight gathers and activations during execution. [Atlas audit](runs/pilot/atlas_audit.json) · [GPU storage audit](runs/pilot/storage_audit.json)

## Regression checks

The v8 CPU suite passed 12 tests, and the inherited suite passed 35. They cover shared readout storage, prompt isolation, geometry-sensitive decisions, deployed path parity, same-state routing, program-level fitting and per-variant scoring. A reduced GPU preflight exercised the entire protocol on separate subsets of train programs. [V8 tests](runs/cpu_tests_final.log) · [Inherited tests](runs/legacy_cpu_tests.log) · [GPU preflight](runs/preflight/complete.json)

## Coordinate cache check

When a batch shares a path, the engine carries that fact forward across decoder stages and reuses the curve view. This avoids grouping identical geometry again after subdividing a path. A CPU regression check compared all nine continuations with the previous implementation and found bitwise-identical logits. The same test observed grouping for a genuinely mixed batch. [Dispatch checks](tests/test_shared_controller.py)

On four 140-token prompts using the same fitted weights and policy, the optimized GPU dispatch reduced median time for a full nine-route tree from 398.81 to 390.59 ms across eight repetitions. Curve regroupings fell from 21 to zero. The same learned inference pass took 148.61 and 146.64 ms respectively, with three genuinely mixed groupings in each implementation. Both workloads returned bitwise-identical outputs. [Matched dispatch benchmark](runs/checks/dispatch_benchmark.json)

Seven alternating timed repetitions on an RTX 5090 compared capacities of 20 and 22 index plans on the same training batch of four 140-token prompts. Both caches held all 14 distinct plans and recorded zero misses after warm-up. Median full-tree labeling time was 397.13 ms at capacity 20 and 398.29 ms at 22. Both produced bitwise-identical logits. The 20-entry capacity already covers this workload. [Benchmark measurements](runs/cache_benchmark.json)

In a separate end-to-end benchmark on that batch, the learned policy took 151.17 ms versus 122.32 ms for its exact actions supplied in advance, 95.50 ms for fixed (2, 2), and 69.79 ms for dense inference. Peak allocated VRAM was 13.714 GiB for learned routing and 13.625 GiB for dense. Full-vocabulary logits matched when the learned actions were replayed, with zero unchosen continuations. These are warm full-context forward timings without a KV cache. [Inference runtime and storage](runs/checks/selected_runtime.json)

The registered pilot, selected row checkpoint and warm-start identity are available alongside the original per-prompt predictions. The prediction application restores the selected rows into the same GPU model. [Completion record](runs/pilot/complete.json) · [Selected BF16 rows](runs/pilot/selected_rows.pt) · [Frozen policy and checkpoint hashes](runs/pilot/selection.json)
