# Research notes

Snapshot: **November 25, 2026**. These notes summarize exploratory work and preserve its uncertainty. The [project index](data/project_index.json) and [project notes](projects/) contain the fuller record.

## One geometric idea, several different roles

The original question was whether an input could follow a learned Bézier curve through a spatial arrangement of processing modules. The model carried a hidden feature vector; its three-dimensional position acted as an address for selecting work. The address was not a replacement for the feature representation.

Later projects used related geometry for different purposes: feeding activations back into routing, sharing processing fields, reconstructing quantized weights, selecting nodes in pretrained MLPs, revisiting transformer regions, or controlling a nested shared core. Those mechanisms need separate explanations and controls. Success in one does not validate another.

The [original architecture](evidence/docs/architecture.md), [routing study](evidence/Bezier-Routing/README.md), and [pretrained node-routing report](evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md) document that progression.

## Findings worth carrying forward

### Routing is causally active

The digit experiments learned useful computation, and interventions showed that routes mattered to a trained model. Later feedback experiments changed subsequent neuron or group selections after perturbing an activation. Pretrained 7B v6 connected spatial coordinates to actual MLP-node selection; moving relevant coordinates changed selected sets and logits.

These checks establish a functional mechanism. They do not, by themselves, establish that learned geometry is the best mechanism for the task. At the largest trained EMNIST setting in the original scaling study, Bézier routing used 8.900 MiB at 83.59% mean accuracy and 1.234 ms batch-one latency. The compact residual control used 2.830 MiB at 84.04% and 0.457 ms. Those values are two-seed means under that study's checkpoint and timing protocol. [Scaling report](evidence/runs/scaling/report.md)

### Sharing and quantization can give practical savings

The shared-field experiments reduced storage on some handwriting and known-association tasks, while larger association sets and stronger corruptions exposed capacity and robustness costs. Constant or straight controls often performed similarly or better. The useful result is the measured tradeoff from sharing under particular assumptions. [Shared-field results](evidence/Bezier-Explore/RESULTS.md)

Granite's six-bit deployment passed the project's limited language and QA screens at 22.245 GiB. The later offline mixed-precision study produced a validation-selected artifact at 16.464 GiB that passed a fresh final screen. This is the smallest screened artifact selected by that study's validation process, not the smallest artifact to pass its final test: smaller fixed-four-bit and random-mixed controls also passed that test after failing validation. The precision policy is chosen offline; it is not a reasoning controller active during generation. [Six-bit results](evidence/Bezier-Granite/RESULTS.md) · [Mixed-precision results](evidence/Bezier-Granite/rl_compression/RESULTS.md)

### Input-preserving nested-core v4 is a positive exploratory result

Earlier nested-core revisions established shared storage and execution behavior without satisfying their combined quality criteria. V4 added a fixed input-preserving skip and compared matched clean and shared models.

On 512 held-out answer-prefix cases from 32 program files in the seed-7 experiment, the same shared checkpoint improved from 52.73% accuracy with redirects disabled to 57.62% with redirects enabled. The recorded NLL comparisons against zero turns, one turn, and a separately trained clean control had positive paired-program intervals. Both clean and shared models benefited from the changed representation. The clean model reached 57.03% accuracy, so the 4.89-point improvement is against disabled redirects in the same checkpoint, not against the clean model.

This remains a single-seed adaptive follow-up using reused holdouts. Longer-program core reasoning remained near chance, and controlled inference was slower than clean inference. Replication should preserve those boundaries. [V4 report](evidence/bezier_nested_core_v4/runs/input_preserving/report.md) · [Matched comparison](evidence/bezier_nested_core_v4/runs/input_preserving/comparison.md) · [Saved verification](evidence/bezier_nested_core_v4/runs/input_preserving/verification.json)

### The pretrained 7B line has not established useful input-specific routing

The eight pretrained revisions operate on a real 7B Mistral model with shared core storage. They explore distinct questions about adaptation, distillation, memory residency, node-level execution, route selection, and recurrence.

V7's optimized executor substantially improved earlier forced-route timings, but its learned stage policy did not beat fixed and shuffled controls. Its arbitrary recurrent pilot lost answer quality. The later reinforcement-learning continuation reached 512 total updates while retaining a constant path and constant answer at 50% on balanced test and out-of-distribution panels.

V8 follows the stage-ordered branch, not that later RL continuation. Its learned controller used diverse action pairs and had a small test-point advantage, but the uncertainty intervals and shuffled comparison did not establish an input-specific benefit. The frozen validation rule therefore retained fixed `(2,2)` as the released default. [V7 stage report](evidence/bezier_nested_llm_7b_v7/REPORT.md) · [Recurrent pilot](evidence/bezier_nested_llm_7b_v7/runs/recurrent_pilot/REPORT.md) · [RL continuation](evidence/bezier_nested_llm_7b_v7/runs/reinforcement_continued/measured_summary.json) · [V8 report](evidence/bezier_nested_llm_7b_v8/REPORT.md)

### Extra internal computation did not deliver dependable coding gains

Think and Revisit tested additional passes or region visits with pretrained coding models. Their comparisons did not establish a consistent Bézier advantage over ordinary recurrence. LongHorizon's compact pre-RL variant solved one additional task in a four-task pilot, but the difference depended on completing output under a token cap; its controller RL regressed measured outcomes. [Think summary](evidence/Bezier-Think/results/summary.json) · [Revisit results](evidence/Bezier-Revisit/RESULTS.md) · [LongHorizon results](evidence/Bezier-LongHorizon/RESULTS.md)

The separate from-scratch MemorySphere line demonstrated bounded internal scratch memory and large-model training mechanics. The 7.41B ten-hour run completed 22,042 updates and approximately 4.77 million supervised tokens, but its final task panels were only partially scored: 0/60 successful tasks at eight visits and 0/40 with adaptive stopping, each from a requested 96-task panel. The later check-in identified invalid tool actions throughout those recorded panels. Completed training and improved token loss did not establish a working coding agent. [Completion record](evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/COMPLETE.json) · [Later check-in](evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/checkin_2026-11-24.json)

## Keep the accounting separate

Parameter storage, checkpoint size, peak GPU tensors, active parameter applications, and elapsed request time answer different questions. Offloading can lower GPU allocation while making inference slower. Sparse execution retains inactive weights unless a separate storage mechanism changes that. A small logical artifact does not automatically imply a small dense runtime allocation.

The execution studies include useful narrow exceptions, such as the small CPU-actor/GPU-learner PPO workload. They also contain many negative split-placement and dispatch results. Any new speed claim should include routing, dispatch, transfers, synchronization, precision, batch shape, and output quality under a matched comparison. [Engine results](evidence/Bezier-Engine/README.md) · [Reversed layer placement](evidence/Bezier-Layers/runs/reversed_large/report.md) · [Streamed pretrained core](evidence/bezier_nested_llm_7b_v3/EXPERIMENT_REPORT.md)

## Next experiments

1. **Replicate the positive v4 steering result.** Use new seeds and a newly sequestered program holdout. Retain input-preserving clean/shared arms and zero-turn/one-turn controls; measure the usefulness of later decisions.
2. **Demand an input-specific routing gain.** Compare learned 7B policies against validation-selected fixed and shuffled policies on the actual deployed execution path. Keep stage routing, arbitrary recurrence, and RL continuation as separate experiments.
3. **Repair the coding interface before expanding training.** Measure valid actions, valid edits, code correctness, and completed workflows separately. Resolve cached versus full-context discrepancies and finish fixed evaluation panels.
4. **Optimize against one declared resource target.** Compare complete artifacts and full request cost at matched quality. Preserve ordinary quantizers, compact residuals, CPU-only execution, and dense/fixed routes as controls.

## Reading the evidence snapshot

The consolidation reviewed project documentation and selected saved artifacts. It did not independently rerun the historical experiments. Saved checks establish what those checks measured at their recorded revision; they are not a live health check or a guarantee that every result reproduces on another machine.

The 26 folders share ancestry and sometimes evaluation material. Their test, run, and checkpoint totals should not be added as though they were independent evidence. Different answer formats, attempt limits, sample counts, and seeds also prevent a single accuracy leaderboard. In particular, confidence intervals over sampled programs do not substitute for replication across training seeds.

Older source documents are retained for provenance and can contain superseded status text. The current project notes incorporate the later 7.41B completion record, Granite mixed-precision study, actual deployed v6 evaluation, v7 RL continuation, and v8 fixed-policy release decision.

The public package is a research record with selected supporting artifacts. It is not a complete checkpoint or dataset release, and it does not promise a one-command reproduction of all 26 projects.
