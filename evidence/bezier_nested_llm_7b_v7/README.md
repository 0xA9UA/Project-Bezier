# Recurrent spatial computation

<!-- CONTINUATION_LATEST_SUMMARY:START -->
The latest run completed **256 more reinforcement updates (512 total)** from the parent’s learned weights and layout. Held-out learned accuracy is **50.00%** (64/128) on test and **50.00%** (32/64) on longer OOD programs. The deployed policy remains constant: `continue → continue → stop`, always predicting B. This is chance-level accuracy on the balanced binary task, with no evidence of useful input-dependent routing.

66,118 node coordinates changed since the parent; 130,376 are displaced cumulatively from the original atlas. All 140 tests and 57 independent audit checks passed. The central core, parameter ownership and GPU storage sharing remain fixed.

[Continued run: measured results](runs/reinforcement_continued/REPORT.md) · [Clean inference benchmark](runs/reinforcement_continued/runtime.json) · [Independent continuation audit](runs/reinforcement_continued/independent_audit.json)

The first continuation used fresh optimizer state because the original pilot had not saved it. The parent and continued evaluations use different holdouts; cross-run accuracy changes are unpaired.

## Original pilot and implementation
<!-- CONTINUATION_LATEST_SUMMARY:END -->

The current training variant uses **reinforcement learning for routing and stopping**, with node positions contracting only after a correct answer and positive reward. [RL implementation and commands](REINFORCEMENT.md) · [RL pilot results](runs/reinforcement_pilot/REPORT.md)

The 256-update RL pilot moved **115,295 outer node positions** and learned to stop after two turns. All 113 tests and 40 independent audit checks passed. Its held-out policy nevertheless collapsed to a constant route and B answer: **50% accuracy**, identical to fixed/shuffled routing. Useful conditional parameter selection remains unproven.

The new implementation replaces the outer layer schedule and two checkpoints with recurrent node selection, shared-core decisions at every turn, explicit revisits and learned stopping. Start with [the recurrent implementation and commands](RECURRENT.md) and [the new pilot report](runs/recurrent_pilot/REPORT.md). The earlier v7 experiment below is retained for reproducibility.

The earlier supervised 64-update pilot passed all 75 correctness tests and GPU sharing checks, but scored **0% recurrent accuracy versus 61.72% dense accuracy** on 128 fresh test examples. Every learned run reached the eight-turn cap. Its report includes fixed/shuffled comparisons, actual revisits, VRAM and clean latency measurements. The reinforcement-learning experiment above supersedes that routing-training method.

## Earlier stage-ordered experiment

This experiment adapts the 7.25-billion-parameter spatial Mistral model. The outer model and the controller read one GPU weight store. The central core contains 1,087,201,280 shared parameters at radius 0.25. The controller reads the current activation and candidate Bézier geometry to choose which outer nodes execute.

**The optimized engine ran faster, and the trained controller fell short of fixed routing.** Actual learned routing scored 59.24% on fresh test programs and 64.32% on longer programs. The fixed pair selected on validation scored 60.42% and 64.58%. All outside-core weights passed bitwise preservation checks. [Pilot report](WORKSPACE/bezier_nested_llm_7b_v7/REPORT.md)

The comparison of interest is the difference in full-vocabulary answer accuracy between the learned policy and a fixed route chosen on validation with the same weights. Paired comparisons resample complete programs so the four answer variants are kept together. Separate comparisons with shuffled controller decisions measure whether applying each decision to its own input matters.

The previous pilot's strongest fixed route turned in direction 2 at both checkpoints. On its 128 validation examples, that route got 86 answers correct. Allowing a perfect choice of second action while keeping the first action fixed at 2 could have increased the result to 91. The v7 pilot uses fresh held-out programs to assess how often a trained core can identify opportunities of this kind.

During the new pilot, all 48 core-bearing tensors received gradients across 256 reasoning and 64 replay updates. Validation selected a constrained fit to two shared output rows on the original warm start. The two gradient-trained candidate checkpoints scored lower. That distinction matters when using the selected core weights. [Training and selection](WORKSPACE/bezier_nested_llm_7b_v7/runs/pilot/training_summary.json)

With identical forced routing decisions for four 140-token prompts, the engine improved from 141 to 95 ms on one common path and from 1,452 to 155 ms with four different paths. The same-weight dense pass took 69 ms. All measured policies retained a single GPU weight store. [Matched runtime comparison](WORKSPACE/bezier_nested_llm_7b_v7/runs/benchmark/comparison.json)

The selected model's end-to-end learned forward took 148 ms on four 140-token prompts, including its decisions. A separate generation check answered the arithmetic example below incorrectly. [Selected runtime](WORKSPACE/bezier_nested_llm_7b_v7/runs/selected_runtime.json) · [Generation result](WORKSPACE/bezier_nested_llm_7b_v7/runs/generation.json)

## Run inference

The generation command uses the selected checkpoint and its two validation-selected thresholds. It applies the training chat template, executes a single outer continuation after each decision, and records the action pair for every generated token. Choose a fresh output filename.

```powershell
Set-Location 'WORKSPACE/bezier_nested_llm_7b_v7'
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' applications/generate.py --run runs/pilot --mode learned --prompt 'Evaluate (12 + 5) * 2. A: 34 B: 29 Answer:' --max-new-tokens 8 --output runs/generation_new.json
```

The modes `learned`, `best-fixed`, and `dense` use the same resident weights. The default model call and `conditional_forward` run the new controller. Generation recomputes the current context at each step. [Generation application](WORKSPACE/bezier_nested_llm_7b_v7/applications/generate.py)
