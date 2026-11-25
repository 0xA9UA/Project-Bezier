# Bézier Think

A local experiment comparing Bézier routing, straight routing and ordinary
shared recurrence in Qwen2.5-Coder-1.5B-Instruct at **2, 4, 8, 16 and 32 internal
steps**. Five counts times three methods gives fifteen settings per fitting
seed. Two seeds, two trained zero-step controls and two unmodified model
controls give **34 conditions**, each evaluated on **24 coding stages**.

The larger control is Qwen2.5-Coder-14B-Instruct. Both bases use BF16, so model
size is not confounded with different quantization formats. Original Granite
experiments and model files are unchanged.

## What an internal step does

The prompt is encoded once. The field transforms the last hidden state into a
continuous embedding. That embedding passes through the complete shared causal
language model and adds a KV-cache position. The process repeats before the
model generates a JSON tool action. Internal steps are additional model passes,
not terminal commands, textual chain-of-thought tokens or refinement of a fixed
integration interval. The KV cache is released after each action request.

The latent state has 256 coordinates divided into sixteen groups. A shared
128-dimensional processor updates four geometrically selected groups in the
Bézier and straight variants. The Bézier controller predicts interior control
points and a spherical endpoint; the straight controller uses a chord. The
ordinary variant updates all groups without a spatial router. Differences in
parameter count and active group computation are reported explicitly.

At each step the router samples the midpoint of its newly predicted cubic or
straight segment, selects nearby groups, and carries that position into the
next step. Thus this specific implementation tests adaptive midpoint routing;
it does not integrate or search every point along a complete curve. At the
cubic midpoint the two interior control points have equal coefficients.

This is an adapter experiment using a small pretrained LM. It **does not replace
the LM's transformer layers with a smaller shared core** and does not establish
that Bézier routing compresses model weights. Its comparison tests whether
additional internal computation helps this small coding agent.

## Fixed evaluation

See [docs/protocol.md](docs/protocol.md) and the immutable
[runs/manifest.json](runs/manifest.json) for settings, hashes and task IDs.

- 96 MBPP training tasks, 12 validation tasks, 12 held-out MBPP test tasks.
- Four authored projects, each with three cumulative stages; code carries forward.
- Terminal-style tools for reading fixture files, editing `solution.py`, and
  running public tests; no GUI or arbitrary operating-system commands.
- Six actions and 1,536 output tokens per stage; 384 tokens per action.
- Hidden tests score the final code only and never enter prompts or feedback.
- Training uses 160 updates per fit, two seeds, rank-eight Q/V attention LoRA,
  and validation-only selection at updates 80 and 160.
- Quality evaluation uses batches of four. Separate single-request measurements
  use three repeated, shuffled timing probes at every count.

MBPP references are filtered for the declared restricted Python executor and
length limits before fitting or scoring. Exclusions are recorded. Public MBPP
may have appeared in model pretraining; the authored projects are an additional
check. This pilot does not establish SWE-bench or general coding-agent parity.

## Running

The local `.venv` reuses read-only Torch/Transformers packages from the existing
project runtime via `shared_runtime.pth`; Accelerate and psutil are installed
only in this new environment. Exact runtime versions are recorded in manifests.

```powershell
# Public, pinned downloads with SHA-256 verification. Requires network access.
.\scripts\download.ps1

# These preparation commands precede the immutable manifest and final fitting.
.\.venv\Scripts\python.exe -m bezier_think.tasks
.\.venv\Scripts\python.exe -m bezier_think.pilot
.\.venv\Scripts\python.exe -m bezier_think.freeze

# Sequential GPU work; completed fits/conditions are skipped on restart.
.\scripts\run.ps1 *> experiment.log

# Audits and report after completion.
.\.venv\Scripts\python.exe -m bezier_think.verify
.\.venv\Scripts\python.exe -m bezier_think.report
```

Do not regenerate the dataset or manifest within an existing run. Keep a new
run directory for a changed scientific protocol. Interrupted individual fits
restart from their seed; completed fits are retained. Hidden results are never
used to select checkpoints, budgets or code candidates.

## Artifacts

`assets_manifest.json` records publisher revisions and actual file hashes.
`runs/adapters` contains compact adapters and fitting logs. `runs/evaluation`
contains editable fixture snapshots, complete tool traces and final scores.
`runs/timing` contains individual latency probes. `results/report.html`,
`results/comparison.png` and `results/summary.json` are generated from completed
conditions and explicitly mark incomplete runs.

The restricted Python worker uses AST and builtin restrictions, a separate
process, a three-second timeout and a Windows Job Object memory limit. It is
intended only for these pure-function fixtures, not arbitrary Python packages.

## Sources

- [Qwen2.5-Coder report](https://arxiv.org/abs/2409.12186)
- [Continuous latent reasoning](https://arxiv.org/abs/2412.06769)
- [Scaling test-time compute with recurrent depth](https://arxiv.org/abs/2502.05171)

This is a new local adaptation, not a reproduction of these papers.
