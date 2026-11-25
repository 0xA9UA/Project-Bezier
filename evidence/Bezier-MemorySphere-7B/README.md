<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Memory Sphere 7B

The 7B model replaces the previously proposed 14B target. This is a full-stack
custom sphere model, with **7,346,916,807 independent parameters**, initialized
randomly and trained locally without a pretrained teacher, encoder, decoder,
transformer stack, or imported language-model weights.

This is an early training experiment, not a pretrained general coding assistant.
Its parameter count is architectural capacity, not evidence of coding capability.
The completed results are in `results/report.html` and
`results/release_summary.json`. The 600-update pilot passed **0/60 task stages**
at each of 2, 4, 8, 16, and 32 visits and with adaptive stopping. It frequently
emitted `finish` before a useful edit. Better held-out token prediction did not
translate into completed programs.

Training supervised 14,612 sampled answer tokens and took 8.6 minutes including
validation and checkpoint saves. Observed peak tensor allocation was 20.02 GiB
for training and 13.72 GiB for the batch-one inference benchmark. The same
32-token reference continuation took 0.277 seconds at two visits and 3.044
seconds at 32 visits. Inference allocation stayed level through 128 visits on
that fixed workload. These numbers exclude driver/display overhead and do not
establish efficient longer-context processing.

The final audit verifies the exact parameter count, checkpoint hashes, sampled
weight changes in all 64 regions, and zero scoring discrepancies on independent
regrading. All 32 representative full-size first-action checks matched serial
generation. Save/reload preserves optimizer statistics and the next update
exactly in the dedicated BF16 resume test.

## What was scaled

The model has 64 independently learned spatial regions. Each region contains a
10,496 by 10,496 matrix. A 4,096-dimensional recurrent state, 512-dimensional keys,
clustered context attention, and four cyclic scratch KV slots support the shared
field. Four regions are selected per token per visit. Input formation uses the
same field once before recurrent processing, and output uses a learned surface
address plus the model's own tied token embedding. The existing own 2,048-token
byte-level BPE was learned only from the training partition.

The 3D coordinates address regions. They are not replacements for all the
high-dimensional activation values, and the model still contains shared neural
projections and nonlinearities. It does not have a conventional deep stack of
independent transformer blocks. Cubic Bezier routing replans every three visits.
Straight and direct routes remain available for architecture checks, but the
current trained checkpoint uses Bezier routing only.

The original small model evaluated every region before mixing the selected
outputs. This version dispatches only the selected region computations. A
reduced-size FP32 comparison confirms equivalent outputs to the original dense
implementation. Geometry, route probabilities, and attention scores accumulate
in FP32; matrix weights and most activations use BF16. A fixed output multiplier
of 256/4096 keeps the tied readout's initialization scale reasonable at this width.
This multiplier was fixed before training; it is not a fitted test-set setting.

The regional matrices contain 7,050,625,024 parameters; shared operations and
input/output contain 296,291,783. Four selected matrices contain 440,664,064
parameters. Including every shared parameter gives a conservative upper bound
of 736,955,847 parameters accessible in one token visit, rather than 7.35B active
at every visit. All 7.35B weights remain resident in VRAM: sparse dispatch reduces
matrix work, not the resident weight storage. Longer real-text contexts still
use more memory. Clustering does not physically compress the text KV cache.

## Training scope

The first local pilot has 600 optimizer updates, batch size one, and up to 32
sampled answer-token targets per update. Internal caps cycle through 2, 4, 8, 16,
and 32 visits. The training corpus contains 12,192 examples (2,565,970 tokens in
the entire stored corpus), drawn from generated program-chain tasks and 96 MBPP
training functions. Sampling 600 examples does not consume the entire corpus.
The exact presented and supervised token counts are recorded in the checkpoint.

The language objective averages available depth exits. Shorter exits therefore
receive more direct training supervision. The halting target is trained only on
32-step unrolls using a token-loss/time-cost surrogate, never hidden evaluation
answers. Surface arrival is an output transition, not proof of correctness.

Factorized Adafactor updates each parameter after its accumulated gradient is
complete, then immediately releases that gradient. It stores row/column second
moments and no full first moments or FP32 master weights. The update was tested
against an ordinary Adafactor step with repeated shared-parameter use. BF16
rounding and the narrow corpus remain limitations. There is no gradient
accumulation or activation-checkpoint recomputation combined with these hooks.

The training and validation partitions are preserved from the earlier experiment.
The 60 held-out stages are reserved for evaluation: 24 short-chain tasks, 12
long-chain tasks, 12 stages in four projects, and 12 MBPP transfer tasks. The
terminal-style harness supports `write_file`, `ls`, `cat solution.py`,
`pytest -q`, and `finish`, with six actions per stage. It runs a restricted Python
checker, not an unrestricted shell or a real repository benchmark. The checker
is byte-for-byte identical to the previously audited one; all 60 reference
programs pass its 468 assertions. The public MBPP tasks are not newly authored.

This is one model seed and one training recipe. There are no freshly trained
7B straight-route or ordinary-recurrence controls, so this run cannot establish
a specific advantage for Bezier geometry.

## Local use

From this directory in PowerShell:

```powershell
$spherePython = '..\Bezier-Think\.venv\Scripts\python.exe'
& $spherePython -m sphere_scale.infer --model models/scratch_pilot_u000600 --steps 8 --prompt 'def add(a, b):'
```

The runtime path reuses installed Python packages only. Inference loads no Qwen
model. Supply `--task task.json` for a JSON object with `spec`, `initial`, and
`public` fields to use the trained tool-action prompt. Generated text is printed,
not executed. `--steps adaptive` uses a maximum of 32 visits. This early model
should not be assumed to follow arbitrary prompts reliably.

To reproduce the implementation checks and score a completed checkpoint:

```powershell
& $spherePython -m sphere_scale.verify
& $spherePython -m sphere_scale.verify_generation
& $spherePython -m sphere_scale.evaluate --model models/scratch_pilot_u000600
```

To continue training from the saved optimizer/RNG state, choose a new run name:

```powershell
& $spherePython -m sphere_scale.train --resume models/scratch_pilot_u000600 --run scratch_continued --updates 3000
```

`--updates` is the desired cumulative update number. Changing the total updates
changes the remaining learning-rate schedule. A checkpoint includes model
shards, exact tensor names, SHA-256 hashes, the own tokenizer, and small trusted
local optimizer/RNG state. Completed checkpoints are never overwritten. The
BF16 weights occupy 14,693,833,614 bytes (13.6847 GiB), plus small file headers.
The optimizer checkpoint is separate. Sharded saving and loading avoid keeping
a full second model in system RAM.

The memory-saving update pattern follows PyTorch's documented
[optimizer step in backward](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)
mechanism. Actual local measurements, rather than that reference's example
numbers, are recorded in this experiment's results.
