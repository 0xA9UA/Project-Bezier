<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# V5 shared-core controller: completed two-arm pilot

**Neither arm demonstrates useful learned steering on fresh test/OOD programs.** Training, evaluations, diagnostics, preservation audits, and runtime checks are complete. Exact shared storage and update scope passed independently of the accuracy result. Absolute remains the preregistered primary; selected checkpoints and policies stay frozen.

## Fixed model contract

The experiment warm-starts [v4 reasoning update 768](WORKSPACE/bezier_nested_llm_7b_v4/runs/accuracy_main/best_r000768_u001024), using a standard 32-layer Mistral decoder. Both arms independently load those weights and create fresh optimizers; this is not an exact optimizer-state resume.

| Quantity | Value |
|---|---:|
| Full model | 7,248,023,552 parameters |
| Shared core | 1,087,201,280 parameters |
| Core fraction | 14.999968918% |
| Outside core | 6,160,822,272 parameters |
| Added learned parameters | 0 |
| Logical core center / radius | `(0, 0, 0)` / `0.25` |

Core membership remains layers 17-21: five attention/norm blocks, four full MLPs, and 14,061 neurons of the fifth MLP, plus two embedding rows, two output-head rows, and the final norm. The full outer forward uses those same weights and all outside-core weights. Only exact core slices are optimized; the outside 85% is frozen.

`CoreExecutionView` supplies a separate execution context with shared parameter views. All full-model weights remain on GPU. There is no independent core model copy, CPU weight offloading, quantization, or KV cache. Transient activations and FP32 readout casts still consume memory. Sharing can avoid a duplicate core weight store; it cannot remove the full model's resident weight requirement.

## Controlled changes

Both arms receive the full detached live activation sequence and shared query/geometry tokens. The existing TRAIN-only route selection is retained: decision layer 26, canonical application MLP 28, and alternatives 27/29. Training evaluates equal-budget outer continuations from the same activation. Correct-answer rescues and spoils provide the primary control signal; probability loss supplies a bounded tie target. Inference scores candidate routes and runs only the chosen outer continuation.

Both arms share two changes from v4. Weighted control BCE is divided by the fixed number of batch-by-candidate elements, rather than the sum of target weights. Thus tie weights retain their intended downweighting; `tie_weight=0.01` and `control_weight=8`. The existing two-row shared-head margin is also computed with FP32 elementwise multiplication/reduction. This retains the mathematical margin formula but does not reproduce the old BF16 head-output rounding bitwise.

| Arm | Raw readout used for training and threshold scoring | Queries per example |
|---|---|---:|
| Absolute | Candidate margin `(turn - stay) / 4` | 2 |
| Relative | Candidate margin minus a genuine same-context Stay-proposal margin | 3 |

The subtraction occurs before the bounded `0.1 * tanh` score and loss. Stay has its actual canonical destination and Bézier geometry. A query-independent additive readout offset cancels mathematically, but useful learned decisions are not guaranteed. The arms differ in this reference subtraction; both receive the normalization and precision changes. Example order and planned updates match, **controller compute does not**. The relative arm's guard stop additionally makes actual completed budgets unequal. Cross-arm outcomes cannot isolate inference-time subtraction or a matched completed training budget.

This is one fork and one MLP substitution per forward, with original attention order. Geometry and placement are fixed. It is not unrestricted traversal or repeated trajectory correction.

## Training and preservation

The [configuration](WORKSPACE/bezier_nested_llm_7b_v5/configs/accuracy_core.json) planned 1,024 reasoning/control updates and 341 replay updates per arm: batches of four and two examples, respectively. Seed **83**, initialization, example order, optimizer settings, and retention guards are shared. Learning rate is `5e-7`; validation occurs every 128 reasoning updates.

The planned full run exposes all 4,096 existing reasoning TRAIN examples once and 682 replay example exposures. No new training programs are added. Ordinary outer answer loss, initial-logit preservation, and replay keep the same optimized core parameters useful to the outer model. The [frozen teacher cache](WORKSPACE/bezier_nested_llm_7b_v5/runs/initial_teacher/teacher_manifest.json) holds warm-start BF16 full-vocabulary logits for 4,096 reasoning and 512 replay TRAIN examples on CPU; a second teacher model is not resident on GPU.

The completed [relative preflight](WORKSPACE/bezier_nested_llm_7b_v5/runs/preflight_relative/training_summary.json) ran 8 reasoning plus 2 replay updates, with 32 and 4 example exposures, and peaked at **19.6124 GiB allocated**. Its [gradient audit](WORKSPACE/bezier_nested_llm_7b_v5/runs/preflight_relative/gradient_audit.json) reached all 48 core slice groups with finite nonzero norms. [68 CPU checks passed](WORKSPACE/bezier_nested_llm_7b_v5/runs/cpu-tests.log), including exact core partitioning, shared aliases, relative score semantics, mixed dispatch, and cohort integrity. These are implementation checks, not final quality evidence.

| Main training outcome | Absolute | Relative |
|---|---:|---:|
| Completed reasoning / replay updates | 1,024 / 341 | 256 / 85 |
| Completed reasoning / replay exposures | 4,096 / 682 | 1,024 / 170 |
| Selected reasoning / replay updates | 768 / 256 | 128 / 42 |
| Selected reasoning / replay exposures | 3,072 / 512 | 512 / 84 |
| Elapsed training | 818.46 s | 264.45 s |
| Peak allocated GPU memory | 18.9554 GiB | 19.6797 GiB |

The [absolute run](WORKSPACE/bezier_nested_llm_7b_v5/runs/absolute/training_summary.json) completed. Its selected update 768/global 1,024 has validation accuracy 65.4297%, a +0.9766-point gain over disabled and validation-best fixed control, at threshold 0.01. It saw 768 unique reasoning groups and one complete 256-group replay pass.

The [relative run](WORKSPACE/bezier_nested_llm_7b_v5/runs/relative/training_summary.json) stopped at reasoning update 256/global 341: disabled validation accuracy was 62.8906% versus initial 64.0625%, a six-of-512-example drop (1.1719 points) exceeding the one-point limit. Its guarded fallback is update 128/global 170, threshold 1.0, **Always Stay**, validation accuracy 64.2578%, and zero control gain. This selected checkpoint saw 128 reasoning groups and 42 replay groups. Completed-run exposure must not be attributed to either earlier selected checkpoint.

The [first-128-update TRAIN diagnostic](WORKSPACE/bezier_nested_llm_7b_v5/runs/training_diagnostics_r128.json) confirms the corrected all-tie loss multiplier is approximately 0.08 (`8 * 0.01`) in both arms. Relative scores have lower global dispersion, but rescue-versus-spoil AUC remains close to chance: 0.489 absolute, 0.515 relative. These use changing-weight training records with hashed provenance, not fixed-model held-out predictions; their analytic logit-gradient summaries are not shared-parameter gradient attribution.

## Frozen selection and evaluation

The [pilot protocol](WORKSPACE/bezier_nested_llm_7b_v5/runs/pilot_protocol.json) was registered before main training. It binds source packages, scripts, configuration, teacher cache, and corpus hashes; training and finalization enforce those identities.

Within each arm and across arms, the preregistered selector first prefers guarded policies with positive validation accuracy gains over both same-weight disabled control and the validation-best fixed route. It maximizes the smaller of those two observed gains, then steered accuracy, then minimizes full-vocabulary NLL. If none qualifies, it uses a guarded accuracy/NLL fallback explicitly labelled as no evidence of conditional benefit. Exact arm ties prefer absolute. Observed validation gains are not confidence bounds or held-out success.

The [new corpus](WORKSPACE/bezier_nested_llm_7b_v5/corpus_eval/manifest.json), seed 2026112505, reserves 128 validation groups and 256 test groups of 3-4 stages, plus 256 OOD groups of 5-6 stages. Each has four counterbalanced variants. Independent interpreters verify all 100 inputs. New final-function signatures exclude all prior v2/v3/v4 final and intermediate functions, old patch correct/foil functions, and one/two-stage final functions. Primitive operations and intermediate prefixes may still be shared across programs. Maximum untruncated prompt lengths are 142/142/170 tokens.

The registered ranking selected **absolute as primary before final testing**, with no protocol amendment. Both selected arm checkpoints/policies belong to one frozen cohort. One inspection-ledger hash permits initial-disabled and both arm evaluations. Final scores cannot select a replacement arm. Controls include disabled, every fixed action, validation-best fixed, and one label/slot-matched whole-program score shuffle executed through actual dispatch. Accuracy uses the full-vocabulary top token. Paired intervals resample complete programs with 10,000 draws and seed 2026; they are not multiplicity-adjusted. [Preregistered diagnostics](WORKSPACE/bezier_nested_llm_7b_v5/runs/control_diagnostics_protocol.json) cover rescues/spoils, answer labels, swap consistency, and all-four-correct outcomes.

## Completed primary quality

The [initial warm-start evaluation](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/initial/evaluation.json) and [primary absolute evaluation](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/absolute/evaluation.json) are complete, with 1,024 examples per fresh split:

| Full-vocabulary accuracy | Test | OOD |
|---|---:|---:|
| Initial warm-start, disabled | 65.4297% | 60.2539% |
| Absolute selected weights, disabled | 65.0391% | 60.1563% |
| Absolute learned control | 64.7461% | 59.3750% |
| Validation-best fixed: Stay | 65.0391% | 60.1563% |
| Matched shuffled scores | 64.3555% | 60.1563% |
| Always MLP 27 | 64.2578% | 58.5938% |
| Always MLP 29 | 64.3555% | 59.7656% |

Primary control versus the same weights disabled changes accuracy by **−0.293 percentage points [95% CI −2.148, +1.563]** on test and **−0.781 [−2.441, +0.881]** on OOD. The validation-best fixed action is Stay, so its comparison is identical. Versus shuffled scores, gains are +0.391 [−1.270, +2.051] and −0.781 [−2.832, +1.270]. All these intervals include zero; a positive test point estimate versus shuffle is not evidence of useful control.

Weight adaptation and inference policy have distinct effects. With control disabled, NLL falls from 0.613171 to 0.610760 on test and from 0.656023 to 0.652184 on OOD, while accuracy changes by −0.391 and −0.098 points. These adaptation comparisons' paired intervals include zero. Enabling routing then raises NLL to 0.614393/0.657353 and lowers accuracy further. Learned-versus-warm-start accuracy is −0.684 points [−2.832, +1.367] on test and −0.879 [−2.832, +1.074] on OOD. Thus neither conditional value nor an accuracy gain from the complete adaptation-plus-routing system is established. All primary arms retain 100% on the 128 old replay examples, a retention diagnostic.

The primary chooses Stay/27/29 **484/0/540** times on test and **599/0/425** on OOD. The [preregistered diagnostics](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/absolute_diagnostics.json) find 60 test rescues/63 spoils and 50 OOD rescues/58 spoils. Test A-labelled examples have 2 rescues/61 spoils; B-labelled examples have 58/2. Turn rates are similar by label (54.10% A, 51.37% B), while outcomes shift strongly toward B. This is an observed answer-preference effect, not proof of a causal mechanism or a gold-label routing shortcut.

All-four-correct changes from 22.27% disabled to 22.66% learned on test: +0.391 points [−3.125, +3.906]. OOD falls from 16.02% to 14.45%. Swap-consistency intervals also include zero. Global, answer-label, and variant-slot shuffle action counts match. Diagnostics ran through the registered v5 script and verified raw metadata, with no late schema adapter or new policy selection.

## Relative fallback and arm comparison

The completed [relative evaluation](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/relative/evaluation.json) selects Stay on **all 1,024 test, 1,024 OOD, and 128 replay examples**. Its learned, disabled, validation-best fixed, and shuffled results are identical: test 65.1367% (NLL 0.616362), OOD 61.0352% (NLL 0.656141), and replay 100%. Its [diagnostics](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/relative_diagnostics.json) therefore have zero rescues, zero spoils, and exactly zero conditional gains. This is the registered no-proof fallback, not a useful learned controller.

Relative-minus-initial accuracy is −0.293 points [95% CI −1.855, +1.270] on test and +0.781 [−1.074, +2.637] on OOD. Relative-minus-absolute learned accuracy is +0.391 [−1.855, +2.734] and +1.660 [−0.488, +3.809]. These intervals include zero. The higher relative OOD point estimate reflects its selected weights and disabled routing, not evidence for Stay-reference steering. Actual exposure differs sharply between arms because relative stopped early; the comparison is not a matched completed training budget. Absolute remains primary, with no post-test arm promotion. All evaluated replay arms retain 100% accuracy.

![Both frozen arms with actual completed and selected training budgets](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary_final/cohort_comparison.png)

The [frozen summary](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary/results_summary.json) and [primary figure](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary/summary.png) retain all metrics. The original cohort chart called the arms matched; the displayed chart corrects that wording and shows actual completed/selected updates. A separate [renderer](WORKSPACE/bezier_nested_llm_7b_v5/analysis/render_cohort_budget.py) and [provenance record](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary_final/render_provenance.json) record unchanged input metrics and input/source/render hashes. Frozen experiment scripts remain unchanged.

## Preservation and resident runtime

Both [absolute](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/absolute_weight_sharing.json) and [relative](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/relative_weight_sharing.json) checkpoint audits passed: all **6,160,822,272 outside-core elements are bitwise unchanged**, including unused interface rows and partial-MLP complements. All 48 core-bearing tensors contain changed values. This proves update scope, not usefulness of every parameter.

The final [absolute](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/absolute_resident/resident.json) and [relative](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/relative_resident/resident.json) checks used BF16, an RTX 5090, and four validation prompts padded to 140 tokens, with three timing repetitions:

| Measurement | Absolute | Relative |
|---|---:|---:|
| Extra CUDA bytes creating another core view | **0** | **0** |
| Peak allocated / reserved memory | 13.6959 / 14.2070 GiB | 13.7823 / 14.5781 GiB |
| Disabled median | 68.971 ms | 68.368 ms |
| Serial controlled median | 90.017 ms | 99.841 ms |
| Controlled / disabled latency | 1.305× | 1.460× |
| Separate-stream controlled median | 91.425 ms | 100.267 ms |
| Serial/separate-stream logits and actions | Bitwise equal | Bitwise equal |
| Observed simultaneous GPU kernel overlap | None | None |

Each model has one 13.50050-GiB weight store; the core views cover 2.02507 GiB already within it. Sharing avoids that hypothetical duplicate core weight allocation, not the full outer model's resident footprint. Controller activations still use memory. Relative still evaluates three core queries even when its frozen threshold chooses Stay. Separate streams do not establish parallel-kernel speedup. These workload-specific allocator figures differ from driver/display VRAM, and short timing checks are not serving benchmarks. No weights are offloaded; every generated token recomputes context without a KV cache.

All eight [final execution tasks](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/suite.json) completed. The [eight-token generation smoke](WORKSPACE/bezier_nested_llm_7b_v5/runs/final/generation.json) produced `B: B\n\nThe correct answer` for `(12 + 5) × 2`, whose correct choice was `A: 34`. Runtime succeeded; the answer was wrong. General free-form reasoning remains unvalidated.

## Reproduction

The holdout is now inspected. Further model/data selection needs a fresh corpus; training or cohort re-registration against this holdout is rejected. To repeat the frozen evaluations, reuse the original selection identity and choose new output directories:

```powershell
$pilotRoot = 'WORKSPACE/bezier_nested_llm_7b_v5'
$python = 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe'
$cohort = Get-Content "$pilotRoot/runs/final/selection.json" -Raw | ConvertFrom-Json
foreach ($armName in @('absolute', 'relative')) {
    $member = $cohort.cohort.$armName
    & $python "$pilotRoot/scripts/evaluate.py" --arm $armName --model $member.selected_checkpoint --selection-json "$pilotRoot/runs/final/selection.json" --policy-json $member.training_summary --output "$pilotRoot/runs/recheck_$armName"
}
```

This one-seed, narrow-task pilot has fixed organization, unequal completed arm budgets, and no matched outer-only-training ablation. The implemented fixes pass engineering checks; neither arm proves useful learned control. Relative's early stop does not conclusively disprove reference subtraction as a broader technique.
