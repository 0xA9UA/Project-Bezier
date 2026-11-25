<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Shared-core Mistral v2 experiment

This follow-up did not establish the combined goal of a reasoning core that usefully controls the outer model.

There was a substantial **outer-model trace improvement** on the fixed 16-program/64-example test panel: intermediate-state accuracy rose from 18.1818% to 95.4545%, final-answer accuracy from 51.5625% to 95.3125%, and complete-trace accuracy reached 90.6250%. The core achieved 10.9091% state accuracy, 53.1250% final accuracy, and 0.0000% complete traces. This is a real improvement under scaffolded multi-step decoding, not a 95% result on the full direct-answer test or proof of effective core control.

The model contains **7,248,023,552** learned parameters. Its fixed central core contains **1,087,201,280 (14.999968918%)**, all shared with the outer model; added learned parameters: **0**. The core center is (0,0,0), radius 0.25. Four tied passes execute 16 core blocks per direct prediction without increasing stored model size.

## Training and selection

| Attempt | Completed / configured updates | Reasoning / replay examples | Control examples | Outcome |
|---|---:|---:|---:|---|
| main | 384 / 1024 | 1152 / 384 | 224 | validation_guard_stopped |
| recovery | 768 / 768 | 2304 / 768 | 1120 | training_complete |

The main attempt was rejected at update **384**: old-validation outer raw accuracy was 93.7500%, and full-vocabulary NLL was 0.30519082. Reasons: replay_outer_accuracy_drop, replay_outer_nll_increase. These rejected weights were not selected.

Recovery restarted from eligible main update 128 with a fresh optimizer. Adaptive recovery based on validation; this bundle does not isolate which change causes an effect.

- Restart from eligible update 128 with a fresh optimizer; do not resume the rejected update-384 weights.
- Reduce both learning rates by a factor of four, double the core gold-target CE weight, and multiply old-task outer KL preservation by four.
- Train the gain estimator from matched counterfactuals, while using threshold 1 during training, above its 0.1 gain bound. Outer training always follows Stay.
- Reduce the control loss weight by a factor of five and ramp it more slowly. Select the inference policy on validation after training.

There were 1152 attempted updates across both runs; this is not the number of updates retained in the selected weight lineage.

Selected checkpoint: [best_u000512](WORKSPACE/bezier_nested_llm_7b_v2/runs/distill_recovery/best_u000512). Its local step is 512.

Validation composite: .25 replay direct raw accuracy +.50 reasoning direct raw accuracy +.25 free-running scaffolded trace final raw accuracy; direct NLL tie-break.

Stop/select previous eligible checkpoint if old-validation raw accuracy falls>2percentage points or full-vocabulary NLL rises>.10nats.

All subsequent test, oracle, zero-core and original-core restoration results are diagnostics after selection. They do not revise the checkpoint or routing policy.

## Data and teacher

The teacher is the adapted pre-v2 baseline, cached on TRAIN only. All 8,704 cache entries were checked against TRAIN program membership; the views are {'replay_final': 512, 'reasoning_final': 4096, 'reasoning_trace': 4096}. Dense supervision uses verified intermediate digit states and final answers. Gold earlier states are used during teacher-forced training, never in final-answer evaluation or the free-running trace probes.

| Split | Program groups | Examples |
|---|---:|---:|
| train | 1024 | 4096 |
| validation | 64 | 256 |
| test | 128 | 512 |
| ood | 128 | 512 |

Every reasoning program has two inputs with different results and both A/B orderings: four examples, exactly balanced. Value-only or position-only guessing is 50%. Full 100-input truth functions are disjoint across splits, new final functions exclude old correct/foil functions, and TRAIN trace-prefix functions exclude held-out final functions. OOD is written program length 5–6 versus 1–4 stages, not a claim about minimum equivalent program complexity.

## Full held-out answers

Forced A/B accuracy considers only the two answer logits. Raw accuracy uses the unmodified full-vocabulary greedy token; raw validity measures whether that token is A or B. Full NLL includes probability spent on other tokens; conditional NLL renormalizes over A/B. All-four/all-variants requires every example in a program group to be correct.

| Model | Dataset | Mode | Forced A/B | Raw | Raw validity | Full NLL | A/B NLL | All variants | Turns |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| initial | reasoning_test | outer_disabled | 51.7578% | 51.7578% | 100.0000% | 0.84376128 | 0.82618099 | 1.5625% | 0.0000% |
| initial | reasoning_test | core | 50.0000% | 50.0000% | 100.0000% | 0.70466747 | 0.70331648 | 0.0000% | 0.0000% |
| initial | reasoning_ood | outer_disabled | 50.0000% | 50.0000% | 100.0000% | 0.89192804 | 0.87700017 | 3.9062% | 0.0000% |
| initial | reasoning_ood | core | 50.0000% | 50.0000% | 100.0000% | 0.70858189 | 0.70722557 | 0.0000% | 0.0000% |
| initial | replay_test | outer_disabled | 100.0000% | 100.0000% | 100.0000% | 6.5937561e-07 | 4.0698753e-07 | 100.0000% | 0.0000% |
| initial | replay_test | core | 50.0000% | 50.0000% | 100.0000% | 0.69694021 | 0.69535642 | 0.0000% | 0.0000% |
| selected | reasoning_test | outer_disabled | 53.7109% | 53.7109% | 100.0000% | 0.73905023 | 0.73873455 | 4.6875% | 0.0000% |
| selected | reasoning_test | core | 50.9766% | 50.9766% | 100.0000% | 0.69812434 | 0.69427778 | 0.7812% | 0.0000% |
| selected | reasoning_test | steered | 53.3203% | 53.3203% | 100.0000% | 0.69347827 | 0.69308128 | 5.4688% | 90.4297% |
| selected | reasoning_ood | outer_disabled | 54.2969% | 54.2969% | 100.0000% | 0.74764764 | 0.7473452 | 9.3750% | 0.0000% |
| selected | reasoning_ood | core | 49.0234% | 49.0234% | 100.0000% | 0.69917122 | 0.69541943 | 0.0000% | 0.0000% |
| selected | reasoning_ood | steered | 54.1016% | 54.1016% | 100.0000% | 0.70146081 | 0.70106138 | 9.3750% | 93.7500% |
| selected | replay_test | outer_disabled | 100.0000% | 100.0000% | 100.0000% | 4.2375155e-07 | 2.7660263e-07 | 100.0000% | 0.0000% |
| selected | replay_test | core | 44.5312% | 44.5312% | 100.0000% | 0.69801304 | 0.69798327 | 6.2500% | 0.0000% |
| selected | replay_test | steered | 100.0000% | 100.0000% | 100.0000% | 1.745286e-06 | 1.5338762e-06 | 100.0000% | 64.0625% |

Paired uncertainty resamples complete programs, preserving all within-program variants:

- reasoning_test, selected core versus initial core raw accuracy: +0.976562 percentage points (95% CI -0.976562 to +2.92969); full-NLL gain: +0.00654313 nats (95% CI +0.00479264 to +0.00832028).
- reasoning_ood, selected core versus initial core raw accuracy: -0.976562 percentage points (95% CI -3.125 to +1.17188); full-NLL gain: +0.00941067 nats (95% CI +0.00755615 to +0.0112408).
- replay_test, selected core versus initial core raw accuracy: -5.46875 percentage points (95% CI -10.9375 to +0); full-NLL gain: -0.00107283 nats (95% CI -0.00522045 to +0.0031269).

Historical context: on the old patch-review test, stock forced accuracy was 67.1875%, while **both** the earlier ordinary baseline and shared-core training reached 100.0000% and 100.0000%. That beneficial adaptation is not evidence of a steering advantage. This v2 experiment starts from the already adapted baseline; the old test is a retention regression, not a fresh discovery set.

## Routing and control attribution

One policy was selected on validation: decision layer 16, alternate MLP 17, threshold 0. Selection compared 72 policies using full-vocabulary gold NLL, with fewer turns preferred on ties. Held-out deployment-style inference executes only the chosen continuation.

Route choices vary, which is necessary but insufficient evidence of useful activation-aware control. Compare fixed-action and score-permutation diagnostics.

| Same selected candidate | Forced A/B | Raw | Full NLL | Turns |
|---|---:|---:|---:|---:|
| always_stay | 53.7109% | 53.7109% | 0.73905023 | 0.0000% |
| always_turn | 53.7109% | 53.7109% | 0.6911899 | 100.0000% |
| learned | 53.3203% | 53.3203% | 0.69411692 | 90.4297% |

The fixed-action table is a **cached counterfactual diagnostic**, not the deployed result table above.

The deployed learned-policy NLL is 0.69347827; recombining cached branch metrics gives 0.69411692, a difference of 0.00063864484 nats. Accuracy is 53.3203% deployed versus 53.3203% cached; turn rates are 90.4297% and 90.4297%. Deployment-style inference dispatches only the selected MLP per example; diagnostic branches execute homogeneous candidate batches and recombine scalar metrics. Grouped BF16 dispatch is a plausible numerical source, not isolated experimentally. Deployed metrics are primary; oracle/fixed-action/permutation metrics remain separately labelled diagnostics. No policy or weights were revised from this difference.

Same-checkpoint **deployed** steering versus disabled full-NLL gain: **+0.045572 nats (95% CI +0.0356455 to +0.0558995)**; raw-accuracy gain: -0.390625 percentage points (95% CI -1.5625 to +0.585938).

Within the cached diagnostic, learned versus always-turn full-NLL gain is -0.00292701 nats (95% CI -0.00630179 to -1.0233e-05). A negative gain favors the fixed always-turn action; varying choices alone therefore does not establish useful activation-aware control.

The separate test oracle found 235 positive, 277 negative and 0 tied branch gains. Mean oracle headroom: 0.098836173 nats; learned achieved gain: 0.044933308; regret: 0.053902865; harmful turns: 246/463.

Score permutations preserve the exact turn count. Learned minus permuted mean gain: 0.0017507593 nats; exploratory one-sided permutation p=0.24902724. These recombinations use cached branch metrics after selection; they are not additional deployed inference or a confirmatory test.

## Free-running scaffolded traces

The format is supplied, each intermediate output is restricted to digit tokens, and the final answer to A/B. Predicted digits—not gold earlier values—are fed back. Raw metrics measure the unconstrained greedy token at those same prediction positions. This is not unrestricted text generation.

| Model / mode | Examples | State accuracy | Raw state accuracy | Final A/B | Raw final | Whole trace |
|---|---:|---:|---:|---:|---:|---:|
| initial / reasoning_test_outer_disabled | 64 | 18.1818% | 18.1818% | 51.5625% | 51.5625% | 1.5625% |
| initial / reasoning_test_core | 64 | 10.0000% | 0.0000% | 50.0000% | 0.0000% | 0.0000% |
| selected / reasoning_test_outer_disabled | 64 | 95.4545% | 95.4545% | 95.3125% | 95.3125% | 90.6250% |
| selected / reasoning_test_core | 64 | 10.9091% | 10.9091% | 53.1250% | 53.1250% | 0.0000% |

Outer raw-state gain on the paired trace panel: +77.2727 percentage points (95% CI +65 to +86.7925); outer final raw-answer gain: +43.75 percentage points (95% CI +34.375 to +53.125).

Core constrained-state gain: +0.909091 percentage points (95% CI -7.69231 to +9); core raw-state gain: +10.9091 percentage points (95% CI +5.66038 to +16.6667). The initial core often produced invalid raw tokens, so improved raw trace accuracy includes output-format learning; it does not by itself demonstrate improved program execution. The trace probe supplies extra output steps and a fixed format, so it is not compute-matched to direct A/B inference.

## Core relevance

Zero-core ablation preserves embeddings, language head, final norm, and the partial-core MLP shell suffix. Original-core restoration, when present, also preserves those trained interfaces and the shell; it substitutes pre-v2 core transformer slices, including their internal norms. Both test relevance/coadaptation under the current outer model, not independent reasoning ability.

| Diagnostic | Dataset | Trained-core full-NLL gain, 95% CI |
|---|---|---|
| zero core | reasoning_test | +10.4789 nats (95% CI +10.4381 to +10.521) |
| zero core | replay_test | +11.4649 nats (95% CI +11.4482 to +11.4812) |
| original core | reasoning_test | -0.0283638 nats (95% CI -0.0462763 to -0.0122627) |
| original core | reasoning_ood | -0.0318563 nats (95% CI -0.0540679 to -0.0122068) |
| original core | replay_test | -1.51805e-07 nats (95% CI -2.56113e-07 to -8.3819e-08) |

Restoring the original pre-v2 core lowered direct-answer NLL on reasoning_test, reasoning_ood. The new core weights therefore did not establish a benefit in this direct-answer restoration probe. Restoration did not evaluate scaffolded traces, so it cannot attribute their improvement. Zero-core degradation alone cannot attribute useful reasoning to the newly trained core.

## Core recurrence compute check

These are the same selected weights evaluated with different compute budgets; neither a separately trained comparison nor a latency claim.

| Core passes | Forced A/B | Raw | Full NLL |
|---|---:|---:|---:|
| 1 | 49.8047% | 49.8047% | 0.69393496 |
| 4 | 50.9766% | 50.9766% | 0.69812434 |

Four-pass versus one-pass full-NLL gain: -0.00418938 nats (95% CI -0.00601718 to -0.00223987).

## Frozen membership screen

Frozen adapted initial weights, validation only, no updates or architecture selection.

The screen changed logical direct-core membership only. It did not modify the trained architecture or outer routing map. Comparisons across pass counts use different compute budgets; no relocated core was trained.

| Membership | Passes / schedule | Attention calls | New validation forced / raw | Old validation forced / raw | Trace raw state / final |
|---|---|---:|---:|---:|---:|
| [0, 1, 2, 3] | 1 / repeated_stack | 4 | 50.0000% / 50.0000% | 50.0000% / 50.0000% | 0.0000% / 0.0000% |
| [0, 1, 2, 3] | 2 / repeated_stack | 8 | 50.0000% / 50.0000% | 50.0000% / 50.0000% | 0.0000% / 0.0000% |
| [0, 1, 2, 3] | 4 / repeated_stack | 16 | 50.0000% / 50.0000% | 50.0000% / 50.0000% | 0.0000% / 0.0000% |
| [0, 10, 20, 31] | 1 / repeated_stack | 4 | 50.0000% / 0.0000% | 50.0000% / 0.0000% | 0.0000% / 0.0000% |
| [0, 10, 20, 31] | 2 / repeated_stack | 8 | 50.0000% / 0.0000% | 50.0000% / 0.0000% | 0.0000% / 0.0000% |
| [0, 10, 20, 31] | 4 / repeated_stack | 16 | 50.0000% / 0.0000% | 50.0000% / 0.0000% | 0.0000% / 0.0000% |
| [0, 1, 30, 31] | 1 / repeated_stack | 4 | 50.0000% / 0.0000% | 50.0000% / 0.0000% | 0.0000% / 0.0000% |
| [0, 1, 30, 31] | 2 / repeated_stack | 8 | 50.0000% / 0.0000% | 48.4375% / 0.0000% | 0.0000% / 0.0000% |
| [0, 1, 30, 31] | 4 / repeated_stack | 16 | 50.0000% / 0.0000% | 50.0000% / 0.0000% | 0.0000% / 0.0000% |
| [0, 10, 20, 31] | 2 / input_middle_output | 6 | 50.0000% / 0.0000% | 50.0000% / 0.0000% | 0.0000% / 0.0000% |
| [0, 10, 20, 31] | 4 / input_middle_output | 10 | 50.0000% / 0.0000% | 50.0000% / 0.0000% | 0.0000% / 0.0000% |

## Evidence and limits

The following conservative descriptive checks are not a new training selector or preregistered confirmation test:

| Check | Observed support |
|---|---|
| fixed budget central all shared | yes |
| selected checkpoint passed validation retention guard | yes |
| fresh test core forced accuracy ci above 50pct | no |
| fresh test core raw accuracy improves over initial | no |
| free running core constrained state accuracy improves | no |
| same checkpoint steering improves full nll | yes |
| controller makes nonconstant choices | yes |
| controller beats same candidate always turn | no |
| score permutation diagnostic supports alignment | no |
| zero core ablation shows relevance | yes |

- Single training seed; adaptive validation-guided recovery.
- Training changes are a bundle, not a causal ablation of distillation, recurrence, learning rates, or routing.
- One delayed MLP substitution per prediction, not repeated unconstrained sphere traversal.
- Four tied core passes reuse weights but increase computation; no latency improvement is established here.
- Scaffolded traces restrict digit/answer outputs and feed back predicted values; not unrestricted text generation.
- Old test was previously inspected; it is a retention check. New test/OOD were first scored after selection.
- Zero/restoration ablations measure relevance and coadaptation, not independent reasoning ability.

Training peak memory and elapsed time are recorded in the machine summary; they are not latency comparisons. Teacher caches and optimizer state add training storage, despite zero additional learned model parameters.

## Artifacts

[Machine summary](runs/results_summary.json)

- [protocol](WORKSPACE/bezier_nested_llm_7b_v2/runs/protocol.json) — SHA256 `5600fa338a68cac638299f7b1dd4fc857d2d88fe1238c1868bd1076c7b96d27e`
- [recovery_protocol](WORKSPACE/bezier_nested_llm_7b_v2/runs/recovery_protocol.json) — SHA256 `0aa73049dcffc3a4a4ea358cd29169b5ef52b44b9c26488e25c4fc4fce2fdbb8`
- [main_summary](WORKSPACE/bezier_nested_llm_7b_v2/runs/distill_main/training_summary.json) — SHA256 `81eca47097ac23219d728f39af66e5e1eb64ab41512b70acc665befac432a4d2`
- [recovery_summary](WORKSPACE/bezier_nested_llm_7b_v2/runs/distill_recovery/training_summary.json) — SHA256 `1449b0a1038c7b93d247f4e7b52e5c41317ea629acd3a72daffe6e20fd8afb90`
- [initial_evaluation](WORKSPACE/bezier_nested_llm_7b_v2/runs/eval_initial/evaluation.json) — SHA256 `d294d92ed29528227510fcc7749392980b52401e6a838b489a41da710b8a0af8`
- [selected_evaluation](WORKSPACE/bezier_nested_llm_7b_v2/runs/eval_selected/evaluation.json) — SHA256 `def6a945d0d432c91d52cf4f521cfe452ea4a9f90613cc3f26e8f8d86f4df72a`
- [reasoning_manifest](WORKSPACE/bezier_nested_llm_7b_v2/corpus_reasoning/manifest.json) — SHA256 `37cecf3d3119329f7d29e1d37e19cc668ceefd861de9f66a9b02aa2292d822b3`
- [replay_manifest](WORKSPACE/bezier_nested_llm_7b/corpus/manifest.json) — SHA256 `257969c1b134abccf0a890c92a67ae0e711087692842af10718c9610169038af`
- [teacher_cache](WORKSPACE/bezier_nested_llm_7b_v2/teacher_cache/manifest.json) — SHA256 `a70d08b64e135e0edd8fbb414cd8a640ff933ec5d32f14622a8ea7df5b9febae`
- [membership_screen](WORKSPACE/bezier_nested_llm_7b_v2/runs/core_membership_screen/screen.json) — SHA256 `a204e1113441f54a2968b1b431d6385cc1f2c7a0c71d7701be788771c3ac737e`
- [prior_report](WORKSPACE/bezier_nested_llm_7b/runs/pilot_report.json) — SHA256 `7ee2b74d91290f32635eb50a330e2ca5d5670b2adfaa3d2f984a16c9c5cc9ad8`
- [core_restore](WORKSPACE/bezier_nested_llm_7b_v2/runs/eval_core_restore/core_restore.json) — SHA256 `2366e25c012ae419181a44029d3c59975f96e5e07dd057df8dc559d734a41670`
