# Project Bézier

A working, trained PyTorch proof of concept of **input-dependent Bézier paths through the interior of a sphere**. An input enters at a learned surface point, carries a hidden feature vector through spatially selected processing modules, and is decoded at another learned surface point. The original model has 64 hidden features; the trained scaling study also tests widths 128 and 256.

All source, the Python environment, package caches, dataset splits, trained weights, measurements, and visualizations are under `WORKSPACE`. The local GPU was detected as an **NVIDIA GeForce RTX 5090, 32 GB**, and was used for training and inference measurements. No cloud training, credentials, or paid services are required.

## Scaling study: efficiency, storage and reliability

**The tested implementation shows a conditional sparse-computation speed benefit, but no overall advantage in knowledge per byte or reliability.** The main experiment trained and evaluated 92 runs across two seeds, up to 2.33 million parameters, 100,000 real training images and 16,384 independent synthetic facts. A separate untrained execution check reaches 68.1 million parameters. Every planned run completed; all 92 saved checkpoints reproduced their recorded nominal test predictions, and all 24 tests passed.

- [Complete scaling report](runs/scaling/report.html), with tables, figures and measurement boundaries.
- [Exact aggregate results and paired comparisons](runs/scaling/summary.json).
- [Prospective methodology](docs/scaling_plan.md) and [frozen configuration/environment](runs/scaling/manifest.json).
- [Reproduction commands and checkpoint usage](docs/scaling_usage.md).
- [Independent artifact verification](runs/scaling/verification.json).
- [Supplementary execution report, untrained models](runs/scaling_execution/report.html).
- [Mathematical benefits and proposed shared-field variants](docs/mathematical_benefits.md), with primary research sources and reproducible algebra checks.

At the large trained image setting, each model has hidden width 256 and expert bottleneck 128. The routed models and full residual store 32 experts; the compact residual stores eight. These values are means across seeds 7 and 19, using validation-loss-selected checkpoints and the same 18,800 official test images:

| Model | Checkpoint MiB | Test accuracy | Batch 1 ms | Batch 256 ms |
|---|---:|---:|---:|---:|
| Bézier | 8.900 | 83.59% | 1.234 | 0.998 |
| Straight routes | 8.899 | 84.13% | 0.933 | 0.974 |
| Ordinary expert routing | 8.897 | 83.76% | 0.627 | 0.641 |
| Full residual | 8.865 | 84.51% | 1.821 | 1.778 |
| Compact residual | 2.830 | 84.04% | 0.457 | 0.470 |

Bézier is about 1.48× faster than the full residual at batch one here, with lower measured accuracy. The compact residual is about 2.70× faster than Bézier, stores about 3.15× less, and has slightly higher mean accuracy. The small residual also meets the predefined 80% validation target with 0.335 MiB, versus 0.346 MiB for small Bézier. No Bézier configuration reaches the 85% validation threshold under the specified checkpoint-selection rule. These are comparisons within the tested grid, not globally optimized architecture limits.

At fixed medium capacity, increasing image-training examples from 10,000 to 40,000 to 100,000 raises Bézier test accuracy from 76.05% to 81.20% to 83.81%. The residual comparison reaches 76.96%, 81.58% and 84.33%. More data helps both without showing a distinctive Bézier scaling gain on this task.

For 16,384 known synthetic associations, the medium models give:

| Model | Checkpoint MiB | Fresh-query recall | Stronger-noise recall |
|---|---:|---:|---:|
| Bézier | 1.130 | 99.98% | 69.43% |
| Full residual | 1.111 | 100.00% | 79.77% |
| Compact residual | 0.605 | 100.00% | 61.06% |

The compact model achieves the nominal recall target in less storage, but its stronger-noise recall is worse at this size. At the large setting, compact residual reaches 96.52% stronger-noise recall using 2.206 MiB, versus Bézier's 90.78% using 8.276 MiB. Image noise, translations, confidence calibration and seed variation likewise do not show a consistent Bézier reliability advantage. These are artificial known-key recall tests, not natural-language knowledge or recall of previously unseen facts.

FP16 weight-storage round trips change test accuracy by at most 0.055 percentage points across the 92 runs and roughly halve tensor storage for every architecture. Computation remains float32 in this check. A saved packed lookup table stores the 16,384 independent key/value pairs in 147,456 bytes before runtime/index overhead and retrieves nominal queries exactly; this is an explicit memory reference, not a generalizing neural model.

The separate 68-million-parameter **untrained** benchmark gives Bézier/full-residual latency of 0.969/3.012 ms at batch one, and 5.767/3.578 ms at batch 128. Bézier stores 259.76 MiB of float32 tensors, versus 259.63 MiB for the full residual; batch-128 accounted inference allocation is 775.78 versus 261.77 MiB. The per-input matrix-gather dispatcher introduces temporary allocations. This supports investigating a better dispatcher; these random-weight timings do not establish matched-quality efficiency or knowledge compression.

All reported timings include routing and dispatch in eager float32 on the local RTX 5090. They exclude transfer, preprocessing, startup and loading. Desktop scheduling caused visible variability, and every raw repeat is retained. Two training seeds, one image domain, fixed training budgets and models up to 2.33 million trained parameters cannot settle billion-parameter language-model scaling. Knowledge remains in the stored expert weights; sparse routing skips some arithmetic while retaining those weights.

```powershell
# Reproduce all 92 trained comparisons in a new folder using the cached data.
.\scripts\run_scaling.ps1 -Output runs\scaling_repeat -Device cuda

# Run the supplementary execution benchmark separately, with no concurrent GPU job.
.\.venv\Scripts\python.exe -m bezier_sphere.scaling_execution --output runs\execution_repeat --device cuda
.\.venv\Scripts\python.exe -m bezier_sphere.execution_report runs\execution_repeat
```

## Open the original proof of concept

Open [the offline interactive report](runs/digits/report.html) in a browser. Drag the sphere to rotate it; use the sample and step sliders to inspect actual saved predictions, routes, selected experts, and mixture weights. No server or internet connection is required. The viewer replays saved inference traces; it is not a live inference engine.

- [Mathematical specification and sources](docs/architecture.md)
- [Core model implementation](bezier_sphere/model.py)
- [Results, all seeds and diagnostics](runs/digits/results.json)
- [Experiment configuration and environment](runs/digits/manifest.json)
- [Separate inference benchmark metadata](runs/digits/benchmark_manifest.json)
- [Trained model used by the viewer](runs/digits/bezier_seed7.pt)

![Learned routes](runs/digits/routes.png)

## What was built

The router learns surface entry/exit points and two strictly interior control points. A cubic Bézier curve stays inside the unit ball between its endpoints. Four steps sample the curve; at each step only the two nearest of eight learned expert regions transform the hidden state. Geometry therefore changes which computations execute.

The model retains the information in a feature vector, not in three coordinates. A trainable encoder and decoder connect the spatial computation to a real classification task. The `bezier_transfer` variant additionally learns scalar cubic Bézier activations in the expert bottlenecks. See the specification for the equations, discrete integration rule, top-k gradient limitations, and differences from a full per-edge Bézier network.

There are five trained variants: Bézier routing, straight routing, ordinary non-spatial mixture of experts, a conventional residual network, and Bézier routing with Bézier transfer activations. The sparse expert dispatcher was checked against an all-expert reference for outputs and gradients.

## Observed result on this machine

Task: scikit-learn's bundled 8×8 handwritten digits, **1,078 training / 359 validation / 360 test examples**. Three training seeds (7, 19, 42), 80 epochs each. The saved checkpoint for each run is selected by validation loss. The same fixed stratified split is used for every model. These are ordinary digit classifications, not language-model benchmarks.

| Model | Test accuracy, mean ± sample SD | Parameters | Batch 1 latency | Batch 256 latency |
|---|---:|---:|---:|---:|
| Bézier routing | 96.76% ± 0.42 pp | 40,974 | 0.992 ms | 1.109 ms |
| Straight routing | 97.13% ± 0.85 pp | 40,776 | 0.946 ms | 1.047 ms |
| Ordinary mixture of experts | 96.94% ± 0.56 pp | 38,994 | 0.633 ms | 0.681 ms |
| Residual network | 96.67% ± 0.48 pp | 38,474 | 0.471 ms | 0.534 ms |
| Bézier routing + transfer | 96.85% ± 0.32 pp | 41,998 | 1.365 ms | 1.652 ms |

Latency is per batch, not per item, measured in eager float32 on the RTX 5090. The table uses the median across seeds of each seed's median over seven repeats of 100 forwards after warmup. CPU wall timing encloses CUDA synchronization, routing and dispatch. Inputs already reside on the device; preprocessing, transfer, startup and file loading are excluded. This is a Windows desktop GPU, so timings vary with other activity and clocks. Raw repeat timings are preserved.

**The model learns, but this experiment does not show an efficiency or accuracy advantage for Bézier geometry.** The residual baseline is about 2.1× faster at batch one with similar accuracy; the straight-routing baseline has slightly higher mean accuracy. Small differences on 360 fixed test examples are not evidence of general superiority. Error bars measure training-seed variation, not independent-data uncertainty.

The routes do affect computation: shuffling the learned routes among inputs reduces the Bézier model's average accuracy from 96.76% to 88.52%. Bypassing the interior transformations reduces it to 64.91%. Straightening the trained curves gives 96.30%. These are interventions on already-trained models, not newly trained comparators. Sparse execution and the diagnostic all-expert reference produced identical test logits for all three Bézier checkpoints on this device.

The default routed and residual models each execute eight expert transformations per example. Sparse routing reduces expert work relative to evaluating all eight experts at all four positions (32 transformations), but it does not reduce that count below the residual baseline. Routing, normalization, gathers, parameter storage, and memory movement still cost time. The report contains CUDA allocator peaks and partial linear multiply-accumulate counts; neither is a measurement of physical memory traffic or energy.

## Use the trained model now

Run these commands in PowerShell from `WORKSPACE`. The environment is already installed; activation is optional.

```powershell
# Predict a held-out example and print its actual route and selected experts.
.\.venv\Scripts\python.exe -m bezier_sphere.predict --checkpoint runs\digits\bezier_seed7.pt --sample 0 --device cuda

# CPU inference uses the same trained weights.
.\.venv\Scripts\python.exe -m bezier_sphere.predict --checkpoint runs\digits\bezier_seed7.pt --sample 4 --device cpu

# Run the geometry, sparse-dispatch, gradient, checkpoint-state and CUDA checks.
.\.venv\Scripts\python.exe -m pytest -q

# Reload every checkpoint and verify recorded accuracy, provenance and report links.
.\.venv\Scripts\python.exe -m bezier_sphere.verify runs\digits
```

For your own digit, supply a JSON array with 8 rows of 8 grayscale pixels (or a flat 64-value array). Use **0 for the dark background and 16 for the brightest stroke**. The command validates shape, range, and finite values, then divides by 16 exactly as in training:

```powershell
.\.venv\Scripts\python.exe -m bezier_sphere.predict --checkpoint runs\digits\bezier_seed7.pt --input-json examples\digit_zero.json --output runs\custom_prediction.json
```

The provided example is a digit from the held-out test set, included to demonstrate the input format. It is not an additional independent evaluation. This model expects normalized 8×8 digits; arbitrary photographs require a suitable encoder, dataset, and retraining.

## Train or change the architecture

```powershell
# Repeat all five architectures and three seeds into a new directory.
.\scripts\run_experiment.ps1 -Output runs\repeat -Device cuda

# Train only the geometric model with a larger hidden state.
.\.venv\Scripts\python.exe -m bezier_sphere.train --models bezier --seeds 7 --width 128 --bottleneck 64 --steps 6 --top-k 2 --output runs\wider --device cuda

# Short CPU end-to-end check.
.\.venv\Scripts\python.exe -m bezier_sphere.train --models bezier --seeds 7 --epochs 3 --device cpu --output runs\my_cpu_check --benchmark-iterations 10

# Benchmark already-trained checkpoints without a live optimizer.
.\.venv\Scripts\python.exe -m bezier_sphere.benchmark_run runs\digits --device cuda

# Regenerate plots/report without training.
.\.venv\Scripts\python.exe -m bezier_sphere.report runs\digits
```

Training refuses a nonempty output directory, preserving earlier runs. Benchmarking intentionally replaces timing measurements for the chosen run after every checkpoint has completed; the original training-process measurements remain in `training_results.json`. The README table records the delivered run and does not update automatically when you benchmark again; the HTML report and JSON do.

Use `--top-k 2` or larger when learning geometry. A hard top-1 gate normalized to one gives no classification gradient through the routing weights. Changing `--steps` changes the discrete model; it is not a free solver-accuracy knob. There is no adaptive early exit in this version.

## Recreate the local environment

Tested with Python 3.14.3, PyTorch 2.10.0+cu128 and the exact dependencies in `requirements-lock.txt`. The installed NVIDIA driver was 595.97. The CUDA runtime is included in the PyTorch wheel; this project does not install or modify a system CUDA toolkit or driver.

```powershell
.\scripts\setup.ps1
.\scripts\doctor.ps1

# For a machine without CUDA:
.\scripts\setup.ps1 -CpuOnly
```

Setup uses the existing lock file as constraints and stores its virtual environment, pip cache, temporary files and runtime caches inside the project. CUDA 12.8 is chosen for the local Blackwell GPU. On another Python version, compatible dependency pins may be needed. Training and prediction also support `--device auto`, which selects CUDA when available and otherwise CPU. An explicit `--device cuda` fails clearly if CUDA is unavailable.

## Project map

```text
bezier_sphere/                 Geometry, model, training, inference, benchmarking, viewer
tests/                        24 checks for geometry, dispatch, GPU, data and evaluation
scripts/                      PowerShell setup, environment check and experiment launcher
docs/architecture.md          Equations, assumptions, methodology and primary sources
docs/scaling_plan.md          Prospective scaling and fact-retention protocol
docs/scaling_usage.md         Reproduction, resuming and new checkpoint formats
docs/referenced_conversation.txt  Retrieved mathematical discussion
examples/                     Custom-inference input example
data/                         Saved dataset and exact train/validation/test indices
runs/digits/                  15 checkpoints, histories, metrics, figures, interactive report
runs/digits/source/           Source snapshot matching the original training manifest hashes
runs/digits/benchmark_source/ Source used for the separate inference measurement
runs/smoke/                   Initial 3-epoch GPU integration check
runs/cpu_smoke/               Custom-dimension CPU training/report integration check
runs/scaling/                 92 trained checkpoints, full evaluation, plots and report
runs/scaling_execution/       15 untrained execution configurations up to 68M parameters
data/scaling/                 EMNIST, saved splits, synthetic fact tables and packed lookups
.venv/                        Project-local Python and CUDA PyTorch packages
.cache/                       Project-local package and runtime caches
requirements-lock.txt         Exact installed dependency versions
```

Seeds, source hashes, hardware, dependency versions, data hash, split indices, learning curves and checkpoint-selection epochs are recorded. GPU gather-gradient accumulation can be nondeterministic, so exact bitwise retraining is not promised. Future work should test additional tasks and parameter/compute budgets, compare optimized implementations at matched accuracy, and investigate whether geometry helps at larger scales before claiming inference savings.
