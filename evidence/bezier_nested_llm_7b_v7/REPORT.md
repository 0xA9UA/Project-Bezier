# Can the shared core choose a better route for each input?

## Prior experiment

The earlier controller chose the same first move on all 384 held-out examples. Its predictions differed from the validation-selected fixed route on three test examples and zero longer-program examples. This made the fixed route the relevant baseline for training and evaluation. [Previous pilot](WORKSPACE/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md)

The nine branches of the previous selected checkpoint's validation tree show why both decisions matter. With both moves fixed to 2, the model answered 86 of 128 examples correctly. An oracle that knew the answer could rescue 21 more cases by changing only the first move, five by changing only the second, and 29 with both moves available. These counts measure opportunity on the old validation set. They also show that turning away from the incumbent can be costly. For example, changing the first move from 2 to 1 rescued 12 answers and spoiled 29. [Validation tree analysis](WORKSPACE/bezier_nested_llm_7b_v7/analysis/v6_validation_headroom.json)

## Corpus and measurements

The new evaluation corpus contains 64 validation programs, 192 test programs, and 96 longer programs. Each program has four counterbalanced prompt variants. Final function signatures exclude earlier v2 through v6 final and intermediate functions and the earlier patch functions. Programs can share primitive operations and prefixes. A second interpreter checked every program on all 100 input pairs. [Corpus verification](WORKSPACE/bezier_nested_llm_7b_v7/corpus_eval/verification.json)

The quality measurement is full-vocabulary greedy accuracy of the outer model. Comparing the trained core's decisions with a fixed route on the same weights measures the effect of steering. Comparing those decisions with shuffled decisions tests whether matching the action to its own activation helps. Program-level resampling keeps the four variants together when computing paired accuracy intervals.

## Model

The 32-layer Mistral decoder has 7,248,023,552 parameters. Each of the 987,136 spatial nodes owns defined projection rows, columns, normalization scalars or token-interface rows. The 147,185 nodes inside the sphere of radius 0.25 own 1,087,201,280 parameters, or 14.99997% of the total. At an inference density of 0.98, each stage selects its hidden nodes by their distance to 33 points on the forward portion of the current Bézier curve. The execution schedule preserves the transformer's layer dependencies. Required input and output rows execute in coordinate order.

## Engine measurements

The new engine groups prompts that share a curve once for an entire interval between controller decisions. It prepares selections for the layers in that interval and reuses the mandatory vocabulary traversal order. Its cache stores coordinates and indices. Weight gathers are temporary and continue to read the shared BF16 storage.

Both versions ran in separate processes on the same RTX 5090 with the same v6 checkpoint, token IDs, density and forced actions. Timing used synchronized wall time after warm-up, with nine measured repetitions. The full-model plan fingerprint matched for the checked reference curve. Both engines predicted the same greedy tokens in all four routed cases. The largest absolute difference across output logits was 0.0625. The table measures routing execution with the action decisions supplied in advance. [Benchmark comparison](WORKSPACE/bezier_nested_llm_7b_v7/runs/benchmark/comparison.json)

| Workload | V6 median | V7 median | Speedup |
|---|---:|---:|---:|
| Two 31-token prompts on the same route | 96.80 ms | 45.67 ms | 2.12× |
| Two 31-token prompts on different routes | 140.27 ms | 70.83 ms | 1.98× |
| Four 140-token prompts on the same route | 141.14 ms | 95.45 ms | 1.48× |
| Four 140-token prompts on different routes | 1,451.79 ms | 154.96 ms | 9.37× |

For the four-prompt workload, dense inference took 69.41 ms in the v7 process. Peak allocated GPU memory was 13.63 GiB dense, 13.72 GiB on the common route, and 13.73 GiB on mixed routes. Creating a separate direct core execution view allocated zero additional CUDA bytes. The checkpoint weights remained on the GPU throughout. These measurements apply to the two tested batch layouts with warm route plans and full-context recomputation. [V7 runtime records](WORKSPACE/bezier_nested_llm_7b_v7/runs/benchmark/v7.json)

## Training and selection

The model starts with the v6 pilot's selected core restored into the outer model. At layers 10 and 21, the baseline is action 2. The controller estimates whether actions 0 or 1 will improve the next-token prediction. One direct core pass sees the live activations and the geometry for all three paths. The two reused output rows predict a separate advantage for each challenger. No answer label enters that pass. In a CPU check, perturbing the padded activations and the other example in a batch left the target example's scores unchanged. [Controller implementation](WORKSPACE/bezier_nested_llm_7b_v7/spatial_core/model.py) · [Independent isolation check](WORKSPACE/bezier_nested_llm_7b_v7/tests/test_independent_control.py)

During training, the engine evaluates three counterfactual routes from the same captured activation. First-checkpoint comparisons hold the later action at 2. Second-checkpoint comparisons rotate the earlier action through 0, 1 and 2. Each comparison has the same per-stage node budget. Targets combine the change in full-vocabulary greedy correctness with the difference in correct-token log probability.

For each four-prompt counterbalanced program, training subtracts the program mean separately for each challenger from the predicted and target advantages. This part of the loss rewards learning which presentations benefit from a redirect. An additional absolute-advantage loss, weighted by 0.35, supplies a common reference across programs. The per-program correction is only used during training. Each inference decision uses its own prompt. Ordinary dense prediction, distillation against the warm-start model, and replay train the same core parameters for answer quality.

A second candidate fits a correction directly to the two existing output rows using 96 additional reasoning groups. The fit projects the correction away from the dense final activations of those prompts and 32 replay groups, then solves a regularized least-squares problem on CPU. Each correction has a norm cap of 20% of its original row. The solve can preserve the measured dense logits in float64, while conversion back to BF16 introduces small changes. Validation determines whether the fitted candidate is retained. The two rows remain ordinary shared language-model weights during inference. [Constrained readout fit](WORKSPACE/bezier_nested_llm_7b_v7/scripts/readout_fit.py)

Validation determines one cutoff per checkpoint and chooses the strongest of nine fixed action pairs by accuracy, with negative log likelihood breaking ties. Checkpoint ranking prefers policies whose validation accuracy exceeds both that fixed route and a whole-program action shuffle. Shuffling preserves the variant and answer label while moving the action pair between programs. The dense-path guard allows a maximum two-percentage-point accuracy drop and 0.10 increase in mean negative log likelihood relative to the warm start. The registered primary test criterion requires the lower bounds of both paired 95% bootstrap intervals to exceed zero. [Pilot driver](WORKSPACE/bezier_nested_llm_7b_v7/scripts/pilot.py)

The pilot completed 256 reasoning and 64 replay updates with finite gradients. All 48 core-bearing tensors received a nonzero gradient on each reasoning update. Among 1,024 challenger comparisons at the first decision, 125 changed an incorrect answer to a correct answer and 145 spoiled a correct answer. At the second decision, 58 rescued an answer and 43 spoiled one. The other comparisons could still carry probability differences. These training records were collected as model weights changed. [Training log](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/training.jsonl)

**Validation selected the constrained readout fit on the original v6 weights.** The two checkpoints trained by gradient updates each reached 62.11% validation accuracy after cutoff selection. The warm-start and fitted candidates both reached 65.63%. The fitted candidate had the lower negative log likelihood. The deployed checkpoint therefore incorporates the two fitted output rows from the additional training examples. The earlier 256 reasoning and 64 replay updates were evaluated and discarded. Peak allocated training memory was 18.32 GiB. [Training history](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/training_summary.json) · [Frozen selection](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/selection.json)

The float64 solve limited each row correction to 20% of its original norm, applying roughly 3.1% of each unconstrained correction. On the 448 dense final activations used for protection, the largest stored BF16 A/B logit drift was 0.00516. Recomputing the counterfactuals on six training programs changed two correctness outcomes. The fit's within-program score correlations with the targets increased from 0.020 and −0.017 to 0.058 and 0.016 on its training examples. [Readout measurements](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/readout_fit.json)

The selected v7 checkpoint changed one parameter tensor relative to the warm start. Both fitted rows belong to the shared language-model head. All 6,160,822,272 outside-core parameters were bitwise identical before and after selection. The full model remained on the GPU and every original weight-storage pointer was preserved. [Weight audit](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/weight_audit.json)

The chosen first and second cutoffs were 1.5681 and 2.1727. On validation, the policy answered 168/256 correctly, while the strongest fixed pair (1, 1) answered 171/256 correctly. A gold-aware choice from the nine branches could reach 231/256. Validation cutoff search first evaluated cached branches, then reran the winning policy directly. On this checkpoint the two methods differed on two action pairs and four predicted tokens, with identical total accuracy. Final measurements below use direct execution for every held-out prompt. [Selected validation record](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/validation_0000_readout_fit.json)

## Held-out evaluation

**The v7 core exercised conditional control but missed the registered accuracy criterion.** It answered 455 of 768 fresh test examples correctly. The validation-selected fixed route answered 464 correctly, and shuffled action pairs answered 467 correctly. Compared with keeping the earlier v6 controller on the same test set, v7 lost 19 correct answers. [Test records](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/test_evaluation.json)

The test difference was −1.17 percentage points against the selected fixed route, with a 95% paired interval of [−4.17, +1.69]. Against shuffled actions, the difference was −1.56 points [−3.91, +0.78]. Against the incumbent pair (2, 2), it was −2.60 points [−4.82, −0.39]. Against the prior v6 learned policy, it was −2.47 points [−4.82, −0.13]. These are complete-program bootstrap intervals with 10,000 draws.

At the first decision, 206 test examples took action 0 and the other 562 took action 2. None took action 1, the first move of the validation-selected fixed route. The controller used six distinct action pairs and followed the incumbent pair on only 74 of 768 test examples. Relative to (1, 1), the controller rescued 88 predictions and spoiled 97. Of the A-labelled examples it rescued 5 and spoiled 91. Of the B-labelled examples it rescued 83 and spoiled 6. This asymmetry is consistent with a change in answer preference. The benchmark measures answer outcomes and leaves the underlying mechanism open.

The nine-route sweep on the frozen test outputs found a gold-aware upper bound of 658/768 correct. This describes conditional headroom for the route family. Test labels influenced neither the deployed route selection nor its comparison policy.

| Full-vocabulary answer accuracy | Fresh test, 768 examples | Longer programs, 384 examples |
|---|---:|---:|
| V7 selected conditional policy | 59.24% | 64.32% |
| V7 validation-selected fixed pair (1, 1) | 60.42% | 64.58% |
| V7 incumbent pair (2, 2) | 61.85% | 63.54% |
| V7 shuffled action pairs | 60.81% | 65.36% |
| V7 same-weight dense model | 64.06% | 65.89% |
| V6 warm-start learned policy | 61.72% | 63.28% |
| V6 warm-start dense model | 64.32% | 66.15% |

On longer programs the v7 controller answered 247/384 correctly, one fewer than the selected fixed route. Its paired differences were −0.26 points [−4.69, +3.91] against fixed routing, −1.04 points [−4.43, +2.34] against shuffled actions, and +1.04 points [−1.82, +4.17] against the prior v6 controller. Relative to fixed routing it rescued 46 answers and spoiled 47, including 4 rescues and 43 spoils among A-labelled examples. Six action pairs appeared again and action 1 was never selected at the first checkpoint. A gold-aware choice among the nine paths could have reached 347/384 correct. [Longer-program results](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/ood_evaluation.json) · [Action and outcome diagnostics](WORKSPACE/bezier_nested_llm_7b_v7/runs/analysis/control_diagnostics.json)

## Selected-model runtime and generation

On four 140-token prompts, a complete learned forward took 147.74 ms. Forcing the same four action pairs took 118.99 ms and reproduced the learned logits bitwise. Applying the validation-selected fixed pair to all prompts took 95.18 ms. The dense pass took 69.63 ms. Peak allocated memory in the same process was 13.72 GiB for the learned route and 13.63 GiB for the dense pass. The forced-route benchmark above quantifies the engine change, while these end-to-end timings include the actual controller overhead. [Selected checkpoint timings](WORKSPACE/bezier_nested_llm_7b_v7/runs/selected_runtime.json)

All 35 CPU tests passed. The generation application then used the selected model and the same chat template as training to evaluate `(12 + 5) × 2`. The options were A for 34 and B for 29. It generated eight tokens, answered B, and repeated that B was correct. The engine executed the requested per-token routing, and the arithmetic answer was wrong. [Test log](WORKSPACE/bezier_nested_llm_7b_v7/runs/cpu-tests-final.log) · [Generation record](WORKSPACE/bezier_nested_llm_7b_v7/runs/generation.json) · [Run inference](WORKSPACE/bezier_nested_llm_7b_v7/README.md)
