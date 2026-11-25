# V6 node-level spatial routing experiment

**Node-level routing, shared GPU storage, training, and the new pilot are complete.** Coordinates demonstrably change hidden computation. The learned policy improves this test set over spatial Stay, but an advantage over fixed routing or shuffled control remains unproven. The implementation currently adds latency; it does not establish reduced total VRAM relative to one dense model.

## Architecture implemented

V6 gives every parameter-owning computational node an explicit fixed `(x, y, z)` coordinate. A node is an MLP gated neuron, an attention projection channel, a vocabulary row, or a normalization scale; individual scalar weights are the connections owned by these nodes. The atlas covers the full Mistral model once, without an independent core weight store.

| Quantity | Exact value |
|---|---:|
| Full parameters | 7,248,023,552 |
| Core parameters | 1,087,201,280 |
| Outside-core parameters | 6,160,822,272 |
| Core parameter fraction | 14.999968918% |
| Total spatial nodes | 987,136 |
| Core spatial nodes | 147,185 |
| Coordinate storage, FP32 xyz only | 11,845,632 bytes |
| Added learned parameters | 0 |

The core center is exactly `(0, 0, 0)` and its radius is `0.25`. Core membership is the radial coordinate predicate, validated against the fixed shared slices. A coordinate edit that crosses the core boundary is rejected. The retained pretrained subset is five attention/norm blocks, four full MLPs and 14,061 neurons of a fifth, two embedding rows, two head rows, and final normalization. The 15% restriction is a parameter budget; it is not a sphere volume fraction or the fraction of nodes.

`NodeAtlas` maps each parameter to one owning node axis. Each MLP neuron owns its gate/up rows and down-projection column. A query/output channel owns a query row and output-projection column. Key and value channels each own their projection rows. Embedding/head nodes own vocabulary rows; normalization nodes own scale scalars. The independent meta-model test checks all names, dimensions, totals, unique coordinates and IDs without allocating 7B weights.

These coordinates are logical computation addresses resolved to tensor row/column indices. They do not describe physical three-dimensional GPU memory, and the underlying pretrained tensor layout remains intact.

## How geometry affects inference

The engine measures each hidden node's squared distance to 33 samples in the forward window `t ∈ [0.125, 1]` of a cubic Bézier path. Excluding the common current point allows changed future directions to affect the selection. Under a fixed node budget per compatible stage, nearest nodes are selected with deterministic tie handling. It gathers the selected shared projection rows and columns for computation; it does not compute all MLP neurons and discard their outputs afterward.

Selected Q/K/V projection results are scattered into the required attention shapes, after which ordinary rotary and dense causal attention execute. The attention output projection uses selected shared columns. Selected normalization scales apply their learned values; other scales use identity while the required RMS statistics remain. Every hidden node family participates in coordinate selection.

Embedding and vocabulary output are mandatory semantic interfaces. Input tokens select their required embedding rows, which execute in coordinate order and return to token order. All vocabulary rows execute in coordinate-ranked chunks and return to vocabulary order. Their positions determine scheduling, not which tokens may exist. These interfaces are not sparse vocabulary prediction.

The pretrained causal and layer dependency order remains. This is a node-routed transformer, not an arbitrary cyclic graph or unrestricted layer-free traversal. Within each interval between controller checks, each stage uses the current curve; the curve is subdivided at the next checkpoint. The model does not execute individual nodes in order of their nearest curve parameter. Token position in rotary attention is separate from node xyz.

The core follows its direct shared-weight path. Two control checkpoints, after zero-based layers 10 and 21, can retain the remaining curve or change its second interior control point toward either of two alternatives. A redirect preserves current position, tangent, and final endpoint. The core reads the current activation and geometry and scores alternatives relative to Stay. In ordinary routed inference, only chosen outer continuations run. Counterfactual evaluation in the pilot enumerates the small tree for measurement; that is not the deployed inference cost.

The engine's per-layer dispatch context is sequential and restores original module forwards before another context or core call. This version does not demonstrate simultaneous outer/core GPU kernel overlap. Shared GPU weight storage is verified independently of concurrency.

The supported node-engine API is `SpatialController.route_forward(...)`. The generation application invokes it explicitly for learned and spatial-Stay modes; its dense mode is a comparison using the same weights. The inherited generic `model(...)` interface remains the legacy dense forward and must not be treated as a spatial inference entry point.

![Actual node atlas and geometric node selection](WORKSPACE/bezier_nested_llm_7b_v6/runs/summary/node_geometry.png)

The figure uses a deterministic sample of actual atlas coordinates and a complete geometric selection for one MLP stage. Its paths illustrate static route plans; they are not a trace of pilot activations. The 15% label is a parameter fraction, not a geometric volume fraction. The [CPU-only renderer](WORKSPACE/bezier_nested_llm_7b_v6/analysis/render_geometry.py) and [provenance record](WORKSPACE/bezier_nested_llm_7b_v6/runs/summary/node_geometry_provenance.json) bind the image to its source and coordinate definitions without allocating pretrained weights.

## Training and evaluation design

The pilot warm-starts the selected v5 absolute checkpoint. Placement is deterministic and fixed; this pilot does not learn a new spatial organization. Both the outer and core paths reference the same GPU parameter store. Only exact core slices are updated. The whole dense outer model continues to use those parameters.

The planned pilot uses density `0.98`, 128 reasoning updates, and 32 replay updates. Each reasoning update contains one complete four-example counterbalanced program. Training alternates the two decision checkpoints. From the exact same captured activation it executes Stay and both redirect suffixes under equal node budgets. The controller regresses a clipped correct-token log-probability advantage with a weighted Huber loss. Ties receive weight 0.1 using a fixed denominator. Dense outer answer loss, initial-logit KL targets, and replay preserve the usefulness of the same core parameters to the full model. No second teacher model remains on GPU.

Training at the second checkpoint cycles through balanced exploratory earlier prefixes: Stay, positive redirect, and negative redirect. This covers all three earlier choices but is not fully on-policy sampling from the evolving learned controller. The core is assessed through its effect on full-model predictions, not through standalone core answer accuracy.

Fresh validation/test/OOD splits have 32/64/32 complete programs, or 128/256/128 examples. New final-function signatures exclude previous final and intermediate functions; reusable primitives and intermediate prefixes may still overlap. The held-out tasks remain bounded program execution, with longer programs in OOD. They do not establish general free-form language or reasoning performance.

Checkpoint and threshold selection use validation only, with dense-outer retention guards. The primary ranking favors positive accuracy gain against both spatial Stay and the best fixed pair of redirects; a guarded accuracy/NLL fallback is explicitly not proof of learned control. Test compares the frozen learned policy with same-weight spatial Stay, validation-best fixed routes, shuffled scores, all nine fixed action pairs, and the dense full model. Paired uncertainty resamples complete programs.

## Training-diagnostic correction before final testing

The first attempt included the common current point in route-distance sampling and trained the second checkpoint only from a Stay prefix. All 256 second-checkpoint comparisons in its first 64 reasoning updates had exactly zero target advantage. An independent CPU audit of the actual atlas established why: from that prefix, both second redirects selected exactly the same nodes in all 60 remaining stage groups, so no meaningful preference target could exist. The first attempt was stopped through its source-integrity guard before test/OOD access, and its artifacts were preserved.

The revised sampler uses one fixed forward window, `t ∈ [0.125, 1]`, chosen to exclude the common start rather than through a held-out search. Second-checkpoint training also cycles all three earlier prefixes. A [CPU audit of the complete finite geometry tree](WORKSPACE/bezier_nested_llm_7b_v6/runs/summary/route_set_audit.json) now shows both late redirects change all 60 stage groups after a Stay prefix. After earlier turns, the late alternatives change 59 or 60 of those groups. Both first-checkpoint alternatives change all 66 intervening stage groups. The [initial failing geometry audit](WORKSPACE/bezier_nested_llm_7b_v6/runs/summary/route_set_audit_initial.json) is retained for comparison. These are selection-set checks; they are not accuracy evidence.

The revised [first-64-update TRAIN diagnostic](WORKSPACE/bezier_nested_llm_7b_v6/runs/pilot_v2/train_first64_diagnostics.json) confirms that 255 of 256 second-checkpoint log-probability targets are now nonzero, including all 88 comparisons following Stay. The three preceding actions occur in 11/11/10 training updates. At this checkpoint there are 16 correct-answer rescues, 17 spoils, and 223 ties; prediction/advantage correlation is only 0.079. First-checkpoint comparisons give 20 rescues, 29 spoils, and 207 ties. Every reasoning update reaches 48 core-bearing tensor groups with finite positive aggregate gradient norms. These are changing-weight TRAIN observations and combined-objective gradients, not held-out accuracy or proof that the control loss independently trained every weight.

## Architecture and runtime checks

All [33 CPU tests passed](WORKSPACE/bezier_nested_llm_7b_v6/runs/cpu-tests-v2.log). These include the independent full-size coordinate/ownership census, fixed core membership, exact node budgets, coordinate-sensitive dispatch, rejected stale plans, selected gradient scope, mixed-path batching, interface ordering, forward-window degeneracy regression, exploratory-prefix training, and preservation of the deployed curve subdivisions in counterfactual captures.

The [selected-checkpoint GPU audit](WORKSPACE/bezier_nested_llm_7b_v6/runs/runtime_selected.json) passed on two prompts padded to 31 tokens. The complete model remained on GPU, and every parameter storage pointer stayed stable. Creating another direct core execution view allocated **zero extra CUDA bytes**. The atlas plus node metadata occupies **20,729,856 bytes**; xyz alone is included in the census above. There are still temporary selected-weight gathers and activations. The [corrected-geometry initial audit](WORKSPACE/bezier_nested_llm_7b_v6/runs/runtime_initial_v2.json) and [earlier sampler audit](WORKSPACE/bezier_nested_llm_7b_v6/runs/runtime_initial.json) remain available as history.

At full node density, the selected-checkpoint spatial engine and dense model chose identical greedy tokens on the audit prompts, with maximum BF16 logit difference **0.0625**; chunked output GEMMs are checked numerically, not claimed bitwise identical. Swapping the coordinates of one selected and one unselected MLP node changed the actual selected set and the output logits by up to **0.125**. Restoring coordinates restored logits bitwise. This demonstrates that hidden-node coordinates affect computation.

The two learned action sequences in that audit were both `[2, 2]`. Running those same forced paths reproduced the learned inference logits bitwise. The inference routine evaluated zero unchosen outer continuations.

| Selected-checkpoint runtime measurement | Value |
|---|---:|
| Dense full-model median | 23.244 ms |
| Spatial Stay median | 97.237 ms |
| Spatial learned-control median | 119.918 ms |
| Peak allocated GPU memory across audit timing | 13.7171 GiB |
| Peak reserved GPU memory | 13.7773 GiB |
| Unique full-model BF16 weight store | 14,496,047,104 bytes |

These short three-repetition measurements show substantial overhead in this implementation, not a speed improvement. The aggregate peak does not provide separate matched dense and routed peak-memory baselines. Shared storage avoids a separate core weight copy but does not establish lower total VRAM than a single dense model.

## Pilot results

The revised [training run](WORKSPACE/bezier_nested_llm_7b_v6/runs/pilot_v2/training_summary.json) completed all 128 reasoning and 32 replay updates. The selected checkpoint is the final update, with 512 reasoning and 64 replay example exposures. Main training and validation took 312.12 seconds, with peak allocated GPU memory **19.7642 GiB**. The earlier six-reasoning/one-replay preflight passed separately; it does not contribute updates to this checkpoint.

The frozen validation threshold is `-0.5`. Selected validation accuracy is 65.625%, compared with 55.469% for spatial Stay and 67.188% for the best fixed pair `(2, 2)`. This is the registered guarded fallback: it improves on Stay but does not beat the best fixed route, so the selector explicitly records **no positive validation evidence of conditional control**. These are selection observations, not held-out results.

The [pilot completed](WORKSPACE/bezier_nested_llm_7b_v6/runs/pilot_v2/complete.json). The subsequent [full deployed-path verification](WORKSPACE/bezier_nested_llm_7b_v6/runs/deployed_verification.json) reran every held-out program through actual selected-path inference with the same frozen checkpoint and policy. Its deployed metrics supersede cached-tree metrics below; no selection changed:

| Full-vocabulary accuracy | Test, 256 examples | Longer-program OOD, 128 examples |
|---|---:|---:|
| Initial dense warm-start | 65.2344% | 55.4688% |
| Initial spatial Stay | 60.1563% | 54.6875% |
| Selected weights, dense | 64.0625% | 57.0313% |
| Selected weights, spatial Stay | 59.3750% | 57.0313% |
| Selected weights, learned steering | **66.4063%** | **59.3750%** |
| Validation-best fixed pair `(2, 2)` | 66.0156% | 59.3750% |
| Matched shuffled scores, actual dispatch | 65.6250% | 59.3750% |

On test, learned control improves on spatial Stay by **7.031 percentage points [95% CI +1.172, +12.891]**. On OOD the corresponding gain is **2.344 points [−5.469, +10.156]**. This is evidence for a useful path change over the naive Stay path on this test set, but does not isolate context-dependent control.

The required stronger comparison remains unresolved: learned minus validation-best fixed is only **+0.391 points [−0.781, +1.953]** on test and exactly zero on OOD, where predictions are identical. Versus shuffled scores, test gives **+0.781 [−0.781, +2.344]** and OOD gives **0 [−2.344, +2.344]**. Against same-weight dense inference, learned routing gives **+2.344 [−1.953, +6.641]** test and **+2.344 [−3.906, +9.375]** OOD. Against the initial dense warm-start, the complete adaptation-plus-steering system improves point estimates by **1.172 [−3.125, +5.469]** test and **3.906 [−2.344, +10.156]** OOD. These stronger comparison intervals include zero. Intervals use complete-program bootstrap resampling and are not adjusted for multiple comparisons. No replacement policy is selected after seeing these results.

The [descriptive action audit](WORKSPACE/bezier_nested_llm_7b_v6/runs/summary/control_diagnostics.json) shows that the controller takes action 2 at its first checkpoint for **every test/OOD example**. On test it then takes action 2 in 201 examples and Stay in 55; on OOD, 115 and 13. The learned policy differs from the best fixed pair's predictions on only three test examples (two rescues, one spoil) and zero OOD examples. Thus the measured benefit is largely reproduced by a constant route; improved conditional steering is not established.

Relative to spatial Stay, actual test dispatch gives 49 rescues and 31 spoils. A-labelled examples have 43 rescues/2 spoils; B-labelled examples have 6/29. OOD gives 20/17 overall, split A 15/2 and B 5/15. This answer-preference shift is an observed outcome, not proof of a specific causal shortcut. These post-selection diagnostics do not change the policy. The [earlier cached-tree diagnostic](WORKSPACE/bezier_nested_llm_7b_v6/runs/summary/control_diagnostics_cached_tree.json) is preserved separately.

The [weight preservation audit](WORKSPACE/bezier_nested_llm_7b_v6/runs/pilot_v2/weight_audit.json) verified all **6,160,822,272 outside-core elements bitwise unchanged**, with changes in **all 48 core-bearing tensors**. All model weights stayed on GPU and their storage pointers remained stable.

## Full deployed-path verification

The full rerun found output differences between cached whole-branch batches and actual batches grouped by selected path. Numerical BF16 batch-grouping effects are a plausible explanation; no controlled dtype/kernel ablation isolated the cause. All actions matched the cached tree, but some logits/NLLs and three test predictions differed. The learned accuracy remained 66.4063%; shuffled test accuracy became 65.6250% rather than cached 66.0156%. Actual learned NLL is **0.652998 test / 0.665374 OOD**. Final diagnostics and confidence intervals above use the actual records.

| Policy | Test prediction mismatches / NLL tolerance mismatches | OOD prediction mismatches / NLL tolerance mismatches |
|---|---:|---:|
| Learned | 2 / 20 | 0 / 5 |
| Spatial Stay | 0 / 0 | 0 / 0 |
| Best fixed `(2, 2)` | 0 / 0 | 0 / 0 |
| Shuffled scores | 1 / 19 | 0 / 6 |

The NLL comparison used `atol=rtol=1e-5`; the largest absolute NLL discrepancy was 0.08319. The audit therefore records both cached-output equality and NLL tolerance checks as false. These numerical mismatches are retained, not described as passing equivalence. Validation selection used its original cached-tree measurements and was not redone using test outputs. The actual forward ran zero unchosen outer continuations. Its full held-out verification peaked at **13.9125 GiB allocated**, a different workload from the short runtime audit.

## Generation smoke and reproduction

The [generation smoke](WORKSPACE/bezier_nested_llm_7b_v6/runs/generation_learned.json) completed eight greedy tokens using the selected shared GPU weights and explicit node-routing path. It tokenized a raw prompt rather than applying the pilot's chat template, so it is an execution smoke rather than a matched-format quality benchmark. It produced `B\n\nHere's the reasoning` for `(12 + 5) × 2`, whose correct option was `A: 34`. Runtime succeeded; the answer was wrong, and no post hoc prompt tuning was used. General free-form reasoning is not established by this pilot.

To run a new inference output without changing the frozen experiment:

```powershell
Set-Location 'WORKSPACE/bezier_nested_llm_7b_v6'
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' applications/generate.py --run runs/pilot_v2 --prompt 'Evaluate (12 + 5) * 2. A: 34 B: 29 Answer:' --max-new-tokens 8 --mode learned --output runs/generation_new.json
```

The [application](WORKSPACE/bezier_nested_llm_7b_v6/applications/generate.py) checks source, base checkpoint, selected core, and protocol hashes before loading. Modes `learned`, `stay`, and `dense` use the same selected weights. Inference is serial with no KV cache; each next token recomputes context. Choose a fresh output path. Further training/policy selection requires a fresh final holdout; the completed corpus is now marked inspected.

## Practical limits

There is one persistent GPU weight store. Spatial metadata, route plans, temporary selected-weight gathers, activations, gradients, and optimizer state consume additional memory. The implementation uses PyTorch operations and dense causal attention after channel selection; it is not a custom sparse-attention CUDA kernel. No KV cache is used. Faster inference or lower total VRAM cannot be inferred merely from fewer selected hidden nodes or shared core weights; both require measurement against an explicit baseline.
