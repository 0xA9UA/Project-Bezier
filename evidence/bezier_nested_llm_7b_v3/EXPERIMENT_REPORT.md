<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Shared-core controller experiment: completed results

**The adapted outer model improved on the fresh task, and streaming reduced GPU allocation by 74.69%. Conditional control did not improve the selected model: validation selected Always Stay.** These measurements do not establish the proposed combination of smarter routing and efficient residency.

All final checks completed. The [summary](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/summary/results_summary.json) records input hashes, paired intervals, training history, and runtime. The [figure](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/summary/summary.png) separates quality from memory/latency. The [final checks](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/final_checks.json) distinguish execution correctness from scientific success. Results below use the final selected checkpoint, not the engine preflight.

## Exact shared budget and routing scope

| Quantity | Value |
|---|---:|
| Full model | 7,248,023,552 parameters |
| Logical central core | 1,087,201,280 parameters |
| Core fraction | 14.999968918% |
| Added learned controller parameters | 0 |
| Logical center / radius | `(0, 0, 0)` / `0.25` |
| Shared core blocks, zero-based | 17–21 |
| Core interfaces | Two embedding rows, two output-head rows, final norm |

The core contains five attention/norm blocks, four complete MLPs, and the first 14,061 neurons of the fifth MLP. Its weights are actual views into outer parameters. The ordinary outer path retains access to every full block and vocabulary row.

Placement and geometry are fixed, not learned. The controller reads activation/geometric features at decision layer 16. With this fixed pair, geometry features do not vary across examples, so their added control value is untested. After lookahead, one intervention substitutes MLP 17 for normal MLP 18; **both choices lie inside the core**. Attention order is preserved. This tests one candidate pair, not arbitrary navigation among nearby outer regions. Core and outer computations run sequentially; two concurrent inferences are not implemented.

## Training and selection

The [recovery run](WORKSPACE/bezier_nested_llm_7b_v3/runs/controller_recovery/training_summary.json) restarted from the unchanged [v2 adapted checkpoint](WORKSPACE/bezier_nested_llm_7b_v2/runs/distill_recovery/best_u000512). It completed **512 updates**: 256 reasoning/control and 256 replay. Exposures were 1,024 reasoning/controller examples, including 777 material routing comparisons, and 512 replay examples. These are sampled exposures, not distinct-example counts.

Training combines outer gold-token loss, shared-core continuation-value supervision, and old-task replay cross-entropy/KL. Both continuations start from the same detached activation with equal remaining budgets. Their correct-token NLL difference supervises control; deployment runs only the chosen continuation. Gain/ranking supervision, observed sign balancing, a control ramp, and BF16 stochastic rounding support the shared objective. Standalone core accuracy is not a selector.

The earlier [main attempt](WORKSPACE/bezier_nested_llm_7b_v3/runs/controller_main/training_summary.json) stopped at update 128: old-task validation NLL increased approximately 0.219 nats, exceeding its 0.1 guard. Recovery increased replay frequency from one in four updates to one in two, doubled replay KL weight from 4 to 8, and lowered outer/core learning rates to `1e-7`/`2.5e-7`. All scheduled retention guards passed. Training took 677.94 seconds and peaked at 19.58 GiB allocated GPU memory.

Validation selected [recovery update 512](WORKSPACE/bezier_nested_llm_7b_v3/runs/controller_recovery/best_u000512) by steered outer accuracy, then full-vocabulary NLL. Its threshold **1.0** exceeds the bounded controller score range: deployment is **Always Stay**. Fresh holdout results were not used for selection.

## Full outer-model quality

The holdout excludes earlier final/intermediate reasoning functions, patch functions, and all one/two-stage functions. Each split contains 128 independently verified programs and four counterbalanced examples per program. Accuracy uses the full-vocabulary top token. The [initial](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/initial/evaluation.json) and [selected](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/selected/evaluation.json) evaluations used identical examples/settings.

| Full outer-model mode | Fresh 3–4-stage test | Fresh 5–6-stage OOD |
|---|---:|---:|
| Initial, control disabled | 53.52% | 53.71% |
| Selected, control disabled | 63.67% | 58.59% |
| Selected, frozen controller | 63.67% | 58.59% |
| Selected, Always Turn | 60.94% | 57.42% |

Selected-disabled versus initial-disabled accuracy gains were **+10.16 percentage points**, paired 95% interval **[+5.66, +14.65]**, on test and **+4.88 points [0.00, +9.96]** on OOD. Full-vocabulary NLL decreased 0.12560 nats [0.06311, 0.18714] and 0.06569 [0.01549, 0.11593], respectively. Intervals use 10,000 complete-program paired bootstrap draws, seed 2026, without multiplicity correction.

The controller adds **zero** accuracy/NLL gain over the selected disabled model. Its accuracy difference from Always Turn is +2.73 points [−0.39, +5.86] on test and +1.17 [−2.15, +4.49] on OOD; both intervals include zero. Oracle analysis finds approximately 0.104/0.106 nats of branch-selection headroom, but the constant policy captures none. Score permutation cannot establish activation-specific routing under Always Stay.

The gains therefore belong to **weight adaptation under joint training**, not conditional inference control. No matched outer-only training ablation was run, so the controller loss's contribution remains unresolved. Previously inspected patch retention stayed at 128/128 correct; it is a regression diagnostic, not fresh generalization evidence.

## Shared-core relevance and trace checks

The [restoration diagnostic](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/core_reuse/core_reuse.json) temporarily restored 1,087,180,800 core-transformer parameters to initial values, leaving interfaces and other outer weights trained. Forty-five core transformer tensors had changed. Parameters were restored exactly afterward; checkpoint hashes remained unchanged.

| Learned core contribution versus restoration | Test gain, 95% interval | OOD gain, 95% interval |
|---|---:|---:|
| Accuracy, percentage points | +2.34 [−1.17, +5.86] | +0.20 [−3.13, +3.52] |
| Full NLL reduction, nats | +0.02249 [−0.02263, +0.07228] | +0.00920 [−0.02481, +0.04444] |

Point estimates favor learned core changes, but **all restoration intervals include zero**. This conditional ablation neither proves their benefit nor isolates the control objective.

In the small fixed-format trace panel (16 programs/64 examples), whole-trace correctness decreased **60/64 → 58/64**; final answers decreased 62/64 → 61/64. Predicted intermediate digits were fed back, not gold states. Selected steered/disabled results matched. This panel does not demonstrate improved multistep trace reasoning.

## Final runtime and checks

The [final benchmark](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/engine/benchmark.json) used identical BF16 weights and four 139-token validation prompts on an RTX 5090, three repeats, streamed mode first, and the frozen policy.

| Measurement | Fully resident | Streamed |
|---|---:|---:|
| Peak CUDA allocation | 13.5974 GiB | 3.4421 GiB |
| Peak CUDA reservation | 13.7480 GiB | 3.4922 GiB |
| Process RSS, point measurement | 14.08 GiB | 13.61 GiB |
| Controlled forward median | 79.661 ms | 1,208.119 ms |
| Disabled forward median | 68.968 ms | 1,177.673 ms |
| Eight-token generation probe | 0.32459 s | 9.46457 s |

Streaming reduced peak allocation **74.69%**, while controlled forward took **15.17×** as long and generation **29.16×**. Logical core weights occupy 2.02507 GiB; extra hot outer weights add 0.50626 GiB. The bounded two-block cache adds at most 0.81253 GiB. Streamed shell weights retain 10.96916 GiB of CPU backing, with approximately 11.78 GB transferred per full forward.

Four cases—disabled, selected control, forced outer alternate, and mixed outer alternates—matched logits, predicted tokens, and decisions **bitwise**. All eight generated token IDs matched; core GPU addresses stayed stable. [CPU checks](WORKSPACE/bezier_nested_llm_7b_v3/runs/cpu-tests.log) passed **42/42**.

The separate [generation smoke](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/generation_smoke-console.log) executed but answered incorrectly: `(12+5)*2`, A=34/B=29, produced truncated `B: 2` within four tokens. Runtime success is not arithmetic success; arbitrary-prompt ability remains unvalidated.

The engine is synchronous, single-request, without KV cache/prefetch. Every token recomputes context. “Cold” means GPU cache, not cold OS pages; mode order/desktop load affect timings. Peaks include initialization/all cases, exclude driver/display allocations, and depend on workload. Streaming is a general offloading mechanism, not a unique consequence of shared-core geometry. Full model storage remains required; training still uses resident GPU weights. This is one seed, one fixed placement/pair, and a narrow task.

## Reproduction

Repeat the actual recovery configuration and final suite using **new directories**. Existing local v1/v2 checkpoints/corpora/cache and the verified [holdout](WORKSPACE/bezier_nested_llm_7b_v3/corpus_holdout/manifest.json) are required. Reusing this now-inspected holdout is replication, not another blind test.

```powershell
& WORKSPACE/Bezier-Think/.venv/Scripts/python.exe WORKSPACE/bezier_nested_llm_7b_v3/scripts/run_controller_training.py --config WORKSPACE/bezier_nested_llm_7b_v3/configs/controller_recovery.json --output WORKSPACE/bezier_nested_llm_7b_v3/runs/reproduce_recovery
& WORKSPACE/Bezier-Think/.venv/Scripts/python.exe WORKSPACE/bezier_nested_llm_7b_v3/scripts/run_final_checks.py --training-summary WORKSPACE/bezier_nested_llm_7b_v3/runs/reproduce_recovery/training_summary.json --output WORKSPACE/bezier_nested_llm_7b_v3/runs/reproduce_final
```

The completed [suite record](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/suite.json) retains every command/exit status. The [summary script](WORKSPACE/bezier_nested_llm_7b_v3/scripts/summarize_experiment.py) regenerates JSON/PNG from frozen results without loading a model.
