<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Memory Sphere: coding and agent training

This replaces the earlier 600-update feasibility pilot with a substantial,
time-bounded local run. The user authorized 8–12 hours; the configured job lasts
up to 10 hours, including a final evaluation reserve. It trains the full custom
sphere model from fresh random weights, using no pretrained model or teacher.
The earlier experiments and checkpoints remain unchanged.

**Training status is live, not a claim that training or evaluation has finished.**
Open [the status page](runs/coding_agent_overnight_v1/status.html), which refreshes
every 30 seconds, or inspect [status.json](runs/coding_agent_overnight_v1/status.json).
The worker writes `COMPLETE.json` only after the configured run finishes. A
failure writes `FAILURE.json` and names the last completed checkpoint.

## What is different

The previous pilot supervised only 14,612 sampled answer tokens. This run uses
every next-token target in each sampled Python window and every answer token
in its sampled agent examples. Agent examples are sampled by answer length,
with additional human-task sampling, so very short `finish` examples do not
receive the same sampling share as long code edits. Tool-name tokens receive
extra loss weight. Inference has no rule that forces the correct action.

There are **7,405,637,063 independent parameters**, 64 spatial regions, four
selected regions per visit, a 4,096-dimensional state, clustered KV addressing,
and four overwritten internal scratch slots. The core is unchanged. A new own
16,384-token byte-level BPE, trained on this training split only, replaces the
pilot's 2K vocabulary. The extra embeddings account for the small increase
from 7.35B to 7.41B. There is no Qwen input/output stack.

Factored Adafactor releases each completed gradient during backpropagation.
Stochastic BF16 rounding preserves small updates in expectation without retaining
a full FP32 master copy. Tests cover causal masking, full target coverage,
equivalence to ordinary Adafactor when stochastic rounding is disabled, rounding
bias, exact optimizer-state reload, and exact next-update reproduction with the
saved random state. Full-size preflight backward passes reached about 18.53 GiB
of tensor allocation, including 512-token code windows and a 32-visit case.
This excludes driver/display memory. Training memory can vary by workload.

## Data and separation

- **32,000 Python source files**, 42,602,262 stored training tokens, from a pinned
  subset of [CodeParrot's cleaned training corpus](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train).
- **6,575 coding tasks** and **48,118 supervised action examples**, containing
  1,018,374 answer-target tokens and 13,300,122 total prompt/answer tokens.
- **296 MBPP training problems** with executable human solutions, plus procedural
  Python projects across 14 function families and both one/two-file layouts.
  [MBPP source](https://huggingface.co/datasets/google-research-datasets/mbpp).
- An own 16K tokenizer trained only on training files and training demonstrations.

CodeParrot filtering retains MIT, Apache, BSD, ISC, Unlicense and CC0-labeled
files, parseable Python, source provenance, and per-file hashes. It caps files
per repository, removes exact AST duplicates and known exact held-out MBPP
functions, and separates code data by repository-name hash. Repository forks
and semantic near-duplicates can remain; this is not a blanket decontamination
claim. Downloaded code is used as text, not executed.

Agent demonstrations include file inspection, actual failing initial programs,
edits, test runs, successful observations, and completion only after success.
The reference solutions are checked; no language-model teacher generates them.
Tests remain separate: 427 validation tasks and 717 candidate held-out tasks.
The main protocol fixes a 12-task validation panel and a 96-task final panel
before training. Earlier MBPP test ids are excluded from the new test panel.
Procedural tasks are split by reference-program hash. Repeated demonstrations
for the same task stay in the same split.

The environment is intentionally bounded: model files live in memory, and a
restricted Python checker runs in a separate memory/time-limited process.
Supported terminal actions are `ls`, `cat PATH`, and `pytest -q`; tools also
support `write_file` and `finish`. It includes real Python execution and local
project imports, but does not represent unrestricted shell access, arbitrary
dependencies, or large production repositories. All 29 family/layout oracle
episode checks complete successfully through this environment.

## Overnight schedule

The first two hours mix 70% source-code records with 30% agent records. The middle
stage uses 25% code / 75% agent records, and the last stage 10% code / 90% agent
records. These are record-selection probabilities, not token percentages.
Code windows contain up to 512 supervised next-token targets. Agent answers
are supervised in full. Depth starts at 4/8 visits, then includes 2/4/8/16/32;
long target sequences use at most eight visits to keep memory bounded.

Held-out token losses and live agent episodes are evaluated approximately every
30 minutes. Code correctness, completed test-verified workflows, and premature
finishes are separate metrics. Checkpoints are saved hourly, on validation
improvement, and shortly after startup. The final half hour is reserved for a
held-out evaluation at eight visits and adaptive stopping, using validation-only
checkpoint selection. If the evaluation time budget expires, the report gives
the actual number evaluated and marks the result incomplete.

The run can create roughly 150–300 GB of checkpoints; over 2 TB was free at
preparation time. It uses the local GPU and keeps the PC awake while the worker
is alive; it permits the display to sleep. It exits automatically at the end
and releases that wake request. No paid service, upload, or cloud job is used.

Ten hours is a meaningful first local training run, not enough evidence to call
this a fully pretrained general coding model. Results must establish what it
actually learned. Training completion and competitive coding ability are
different milestones.

## Run, inspect, stop, resume

The active worker is launched hidden by `Start-Training.ps1`. Do not launch a
second worker while one is running; the process also holds an exclusive run lock.

```powershell
# Start a new run, only if one is not already active.
& .\Start-Training.ps1

# Inspect progress.
Get-Content .\runs\coding_agent_overnight_v1\status.json
Get-Content .\runs\coding_agent_overnight_v1\worker.stdout.log -Tail 5

# Request a graceful stop after a completed checkpoint.
New-Item .\runs\coding_agent_overnight_v1\STOP_AFTER_CHECKPOINT -ItemType File
```

To resume a stopped or failed run, remove the stop flag if present and pass the
completed checkpoint path from `latest_checkpoint.json` to
`Start-Training.ps1 -ResumeCheckpoint PATH`. The saved optimizer, data-sampler
state, CPU/CUDA random states and supervised-token counts are restored. A failed
backward pass is never saved as a good checkpoint because it may contain partial
updates. Changing the architecture or tokenizer when resuming is rejected.

To print a model action without executing it:

```powershell
& '..\Bezier-Think\.venv\Scripts\python.exe' -m sphere_train.infer --task task.json --steps 8
```

`task.json` contains `spec`, `initial_files`, and `public`; the same prompt
protocol is used in training and evaluation. Omit `--model` to use the latest
completed checkpoint. This code path reuses installed Python packages from
the earlier environment, never an earlier pretrained model.

Data hashes, source revisions, exact evaluation ids, configuration, source-code
hashes and learning curves are stored alongside the run. The preparation checks
are in `results/`; `training_smoke` and `resume_smoke` are discarded validation
runs, not the main model or its quality results.
