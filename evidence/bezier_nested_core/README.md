<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Nested-sphere Bézier experiment

The revised experiment requested on November 24, 2026 is in
[bezier_nested_core_v2](../bezier_nested_core_v2/README.md). It constrains the
entire central region to 1,312 shared parameters and adds direct reasoning and
live control training. This folder preserves the original slice-based prototype
and its results.

This experiment tests whether a tiny direct-path view of an outer model's own
core weights can predict useful mid-route changes. The checker consumes the
live activation, position, tangent, bend, remaining tail area, progress, and
local candidate regions. It chooses the largest predicted gain above a margin;
otherwise the existing route continues. At inference, only the selected expert
continuation executes.

The core adds **zero learned parameters** in the shared arms. It still uses
compute, intermediate activations, and CUDA scheduling resources. Better
predictions, reduced latency, and useful simultaneous execution are separate
experimental questions; parameter sharing alone establishes none of them.

## Run on this workstation

Open PowerShell in `WORKSPACE\bezier_nested_core`.

```powershell
# Bounded six-arm integration run (24 updates per arm, one seed).
.\scripts\run_experiment.ps1

# Resume that run after interruption.
.\scripts\run_experiment.ps1 -Resume

# Tier 1 training matrix: six arms, five seeds, 3,000 updates each.
.\scripts\run_experiment.ps1 -Full -Tier 1

# Tier 2: promote only after reviewing Tier 1 evidence.
.\scripts\run_experiment.ps1 -Full -Tier 2
```

The direct Python CLI defaults to a deterministic 512-prefix sample per split to make
iteration practical. The PowerShell `-Full` command uses the complete held-out
corpus. When invoking the Python CLI directly, explicitly use
`--max-eval-prefixes 0` with `scripts/run_suite.py`. Capped evaluation is a pilot
and must not be represented as the full confirmatory experiment. Full enumeration
and the full training matrix can take substantially longer than a smoke run.

The isolated `.venv` has independent copies of the verified workstation
dependencies: Python 3.14.3, PyTorch 2.10.0+cu128, and the complete dependency
closure recorded in `requirements-lock.txt`. `environment_files.json` records
every copied file's SHA-256 hash. This avoids modifying any earlier experiment.
To recreate this offline environment after moving the existing `.venv` aside,
run `..\.venv\Scripts\python.exe scripts/bootstrap_env.py` from this folder.
That bootstrap script expects the working parent environment.

This is a deliberate difference from the pasted outline's Python 3.12 and
PyTorch 2.14+ version prescription. The hardware and needed operations are
validated directly. The corpus is an explicitly versioned Python 3.14.3
snapshot; it does not claim hash parity with a Python 3.12 corpus. For a fresh
installation elsewhere, use the [official PyTorch installation instructions](https://docs.pytorch.org/get-started/)
and rerun all environment and invariant checks. CUDA event ordering and tensor
lifetimes follow [PyTorch's CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda).

## Model and causal schedule

The unit ball contains one core expert at the origin and seven shell experts
in Tier 1 (fifteen in Tier 2). A causal GRU encodes 64 preceding bytes. Its
activation predicts a cubic Bézier curve with both endpoints on the unit sphere
and both interior controls inside it. Each visit executes exactly the nearest
two experts, weighted by distance, and updates the activation. The original
exit remains fixed. The 256-way readout predicts the next byte.

Visits occupy the interior of the route at normalized positions `j/(V+1)`.
After the final expert visit, the cursor follows the remaining tail to its
original surface exit without another expert computation. Thus the final
eligible redirect can affect a visit while every branch still pays exactly
`2*V` expert calls.

The checker is a direct `64 -> 16 -> 1` feedforward computation made from
views of the origin expert's first-layer and second-layer tensors. It scores
two nearby alternative regions independently. These same tensor elements
participate in ordinary outer expert execution; they are not reserved checker
capacity. The private-core control copies those views into separate parameters.

A redirect splits the current cubic tail and moves its second control toward
the proposed target. It preserves the join position, the first control/tangent,
and the surface exit; convexity keeps the entire tail inside the unit ball.
The tested bend scale is a fixed hyperparameter (`0.5` or `1.0`), not an
additional learned steering head. A region is a computational address, not a
physical location on the GPU. The core can be selected by nearest-neighbor
routing even at positions slightly outside radius 0.25; radius entry and actual
core-expert execution are reported separately.

Immediate serial checking scores after visit `j` and steers visit `j+1`.
Pipelined checking scores the immutable activation after odd visit `j` while
the outer model executes `j+1`. It waits for the score only before `j+2`, then
retargets the current tail. `pipeline_reference` has the same delayed causal
schedule on one stream and is the correctness comparator for concurrency.
Training uses that reference schedule with autograd; dual-stream execution is
an inference operation. Stream scheduling permits overlap; a CUDA kernel trace
is required to establish observed simultaneous work.

`CapturedInference` captures the complete fixed-shape, fixed-depth dual-stream
forward as a CUDA Graph. Device tensor masks select the tail, so capture does
not need a host-side accept/reject branch or speculative expert execution.
Training matrix multiplies use bf16 on supported CUDA devices; current held-out
evaluation and timing use fp32, explicitly avoiding an unlabelled precision
change within comparisons. Geometry stays fp32 in both cases.

## What makes the check meaningful

For each selected training prefix, a frozen snapshot reaches an eligible
post-region state. The trainer runs Stay and each candidate Turn from exactly
the same activation and curve using the same remaining expert budget and
snapshot policy. For pipelined execution, both branches share the unchanged
lookahead visit before the forced decision is applied. The regression target
is `NLL(Stay) - NLL(Turn)` on the correct next byte. Positive gain means Turn
was better. Labels and snapshot activations are detached.

The live model jointly trains next-byte cross entropy and checker gain MSE.
Snapshots refresh every 100 optimizer steps. Twenty percent exploration applies
only during training. Training depths cycle evenly through 3, 8, 16, and 32.
No counterfactual expert rollouts run during inference.

## Controls and evidence

| Arm | Purpose | Added checker weights |
| --- | --- | ---: |
| clean-outer | Clean next-byte-only baseline | 0 |
| unshared-core | Dedicated checker, same delayed schedule as concurrent | 1,057 |
| nested-core-serial | Shared checker, immediate decisions | 0 |
| nested-core-concurrent | Shared checker, pipelined CUDA inference | 0 |
| ablation-geom-masked | Retrain with geometric inputs zeroed | 0 |
| ablation-entry-state | Retrain using the entry activation in the checker | 0 |

`configs/tier1.json` uses outer width 64 and expert hidden size 128.
`configs/tier2.json` uses width 256 and hidden size 512. The checker view remains
64 by 16 in both. Audit output reports actual counts and fractions.

The corpus deduplicates source bytes by SHA-256, excludes tests and third-party
packages, and rejects files without at least one valid 64-byte prefix plus a
target. Splits are whole-file 80/10/10. Training samples uniformly over valid
prefixes. Evaluation uses fixed prefixes paired across all controls; statistics
resample files and compute the ratio of summed NLL to summed prefix counts.
Validation selects margins independently for zero/one/full-turn caps. A second
full-turn result locks the one-turn margin to isolate repeated steering.

The primary quality criterion is positive lower 95% paired file-bootstrap
bounds for concurrent full-turn routing against zero turns, one turn, and the
separately trained clean baseline at 16 visits. Inspect all five seeds, the
locked-margin comparison, and the retrained ablations. A smoke run cannot
establish that criterion. Shared-weight activation/gradient norms and
outer-only cross-masking test whether the reused core matters to the main
model. Cross-masking degradation is evidence to measure, not assume.

Checkpoints save model, optimizer, frozen teacher, RNG states, completed step,
configuration, source hashes, and corpus identity. `resume.pt` is atomic;
`final.pt` means requested updates finished, not that the scientific hypothesis
succeeded. Resume refuses changed source/config/corpus. A completed run can
be extended by increasing `--steps` with `--resume`.

Use `--bend-scale 1.0 --output runs/tier1-full-bend` for the separate full-bend
variant. Keep it in a distinct output directory because this changes the
experiment. Latency and memory audits report batch-one and batch-sixteen results
with warmup and isolated CUDA allocations. Test results, traces, observed
overlap, and any unrun portions are recorded in the generated run report.
