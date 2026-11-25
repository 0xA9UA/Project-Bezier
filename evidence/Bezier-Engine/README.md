<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Bezier-Engine

A custom PyTorch execution engine testing both requested CPU/GPU splits on the existing Bézier sphere network. It includes inference, supervised learning, and a small PPO reinforcement-learning experiment. It uses the parent project's local Python environment and RTX 5090; no downloads or paid services are required.

## Four hidden layers in the control-point generator

The pipeline experiment now defaults to **four hidden layers per control-point network**, configurable with `--hidden-layers`. Each hidden layer has separate learned parameters and a SiLU activation. The output remains four 3D control points, followed by the original sphere/interior mappings and Bézier equation.

| Profile | Network dimensions | Control-point parameters |
|---|---|---:|
| Small | 64 → 32 → 32 → 32 → 32 → 12 | 5,644 |
| Large | 512 → 512 → 512 → 512 → 512 → 12 | 1,056,780 |

The dependent tree still evaluates four levels and reuses the same network at each node. Tree depth and control-point hidden depth are separate settings. In the shared sphere model API, set `ModelConfig(router_layers=4)`; its default stays 1 so existing trained checkpoints retain their exact architecture and predictions. The mixed-device engine now executes every router hidden layer correctly.

With the unrelated GPU workload finished, a fresh one-layer/four-layer comparison completed **216 configurations**, with three randomized rounds and three full executions per timed observation. Every configuration passed ordered output checks. Seventy related regression tests passed, including old-checkpoint compatibility, gradients through the new layers, geometry, CPU/GPU execution, and repeated pinned-buffer use. These deeper networks use random weights for execution measurements and have not been trained for predictive accuracy.

For the larger four-layer producer and 8,192 independent curves, full GPU took **1.472 ms**, the full-batch hybrid **1.877 ms**, and the best concurrent hybrid **4.030 ms**. For its dependent tree, full GPU took **2.385 ms**, the full-batch hybrid **3.066 ms**, and the best concurrent hybrid **15.141 ms**. These are observed medians on this desktop, not hardware limits; no hybrid beat the fastest CPU/GPU baseline in the tested grid.

- [Four-layer report](runs/curve_pipeline_deep4_idle/report.html)
- [Fresh one-layer comparison](runs/curve_pipeline_shallow_idle/report.html)
- [Architecture and comparison protocol](docs/curve_pipeline_depth.md)

```powershell
# Four hidden layers is now the experiment default. Choose another count explicitly.
..\.venv\Scripts\python.exe -m bezier_engine.curve_pipeline --hidden-layers 4 --iterations 3 --chunks 64 1024 8192 --output runs/curve_pipeline_deep4_repeat
..\.venv\Scripts\python.exe -m bezier_engine.curve_pipeline_report runs/curve_pipeline_deep4_repeat
```

## GPU variables / CPU curve pipeline

A separate execution experiment now tests GPU-generated control points with CPU Bézier evaluation, using concurrent pinned-buffer slots and 1/2/4/8 CPU workers. It tests independent curves and a synthetic four-level binary tree separately. In the tree, CPU positions return to the GPU to generate child controls; this extends the original architecture and is not a trained-model result.

All 180 measured configurations across two three-round pilots passed output checks (maximum recorded absolute difference about 0.0000128). Ten CPU-only tests independently check the equation, recursive tree ordering, transfer accounting, and effective scheduling capacity. The initial 12-configuration CUDA smoke also covered partial batches and slot reuse.

The pilots ran while unrelated GPU work was active. No hybrid's median beat its workload's fastest CPU/GPU baseline, but this does not establish an uncontended winner or a CPU-thread saturation point. Larger batches helped more than extra workers: for 8,192 independent curves with the larger producer, the follow-up measured 1.479 ms for full GPU, 1.805 ms for GPU variables plus one CPU evaluation batch, and 4.399 ms for the best concurrent hybrid. These are eager PyTorch execution measurements, including transfers, not fused-kernel or CUDA-graph limits.

- [Initial concurrent-pipeline report](runs/curve_pipeline_pilot/report.html)
- [Larger-batch follow-up report](runs/curve_pipeline_large_batch_pilot/report.html)
- [Pre-results protocol](docs/curve_pipeline_protocol.md) and [follow-up rationale](docs/curve_pipeline_followup.md)
- [Pipeline implementation](bezier_engine/curve_pipeline.py) and [CPU reference tests](tests/test_curve_pipeline.py)

For a clean rerun after other compute work is idle, use a new directory:

```powershell
..\.venv\Scripts\python.exe -m bezier_engine.curve_pipeline --chunks 32 256 1024 8192 --load-context operator_confirmed_idle --output runs/curve_pipeline_clean
..\.venv\Scripts\python.exe -m bezier_engine.curve_pipeline_report runs/curve_pipeline_clean
```

## Findings

Both splits produce correct outputs and gradients. Neither beats the faster CPU-only/GPU-only baseline across nine tested inference settings. Moving every nonlinearity to CPU adds dependent transfers; GPU routing with CPU neurons can keep the large weight bank in system RAM, with a speed tradeoff.

The clearest positive result is a **CPU actor / GPU learner** split in the small RL task: **2.96 seconds** average training time, versus **3.43 seconds** for GPU policy execution and learning, about **14% less time**. All 15 RL policies reach 100% success on the finite evaluation. This is a small synchronous experiment, not evidence for large simulators or language-model reinforcement learning.

| Schedule | Digit training, seconds | PPO training, seconds |
|---|---:|---:|
| CPU only | 2.65 | 3.49 |
| GPU only | 2.24 | 3.43 |
| GPU matrices / CPU activations | 2.82 | 4.33 |
| GPU route / CPU neurons | 3.41 | 4.72 |
| CPU actor / GPU learner | — | 2.96 |

Means over three seeds. Digit training uses 40 epochs; PPO uses 20,480 transitions per run. Digit test accuracy is 96.02–96.11%. Detailed uncertainty, timing boundaries, raw repeats and limits are in the report.

- [Complete report with charts](runs/main/report.html)
- [Plain-text report](runs/main/report.md)
- [Protocol and primary research sources](docs/protocol.md)
- [Independent verification of all 27 trained checkpoints](runs/main/verification.json)
- [Core execution engine](bezier_engine/engine.py)

## What each schedule executes

| Mode | CPU | GPU |
|---|---|---|
| `cpu` | Entire model | None |
| `gpu` | Input/output handling | Entire model |
| `cpu_activation` | SiLU in encoder/router/experts, or the optional learned cubic expert response | All matrix products, curve construction, selection, gating and LayerNorm |
| `gpu_routing` | Encoder, selected experts, nonlinear responses, normalization and output head | Router, curve positions, expert selection and mixture weights |

The reference architecture chooses a route at entry. `gpu_routing` computes and transfers all step selections together. This experiment does not add adaptive rerouting or new model equations. Geometry remains a three-dimensional address; activations carry full hidden vectors.

The inference runtime also supports reusable pinned transfers and grouped CPU expert dispatch. Pinned buffers wait on CUDA events before CPU reads and before reuse. The runtime is single-request/single-stream and is not thread-safe. The two devices do not execute dependent stages concurrently. Training automatically uses autograd-aware tensor copies instead of scratch-buffer reuse. Construct an engine with the desired mode; do not move the whole mixed-device engine with `.to()` or `.cuda()`.

## Try inference

From PowerShell in `WORKSPACE\Bezier-Engine`:

```powershell
..\.venv\Scripts\python.exe run.py demo --mode cpu_activation
..\.venv\Scripts\python.exe run.py demo --mode gpu_routing
```

The demos load the copied original trained checkpoint, predict eight held-out digits, and print actual transfer counts and payloads. The CPU-activation split performs seven CPU-to-GPU and seven GPU-to-CPU copies for each batch in this four-step model. GPU routing uses one CPU-to-GPU and two GPU-to-CPU copies.

Python API, with this folder on the Python path:

```python
import torch
from bezier_engine.engine import load_engine

engine = load_engine("checkpoints/digits64.pt", mode="gpu_routing",
                     pinned=True, cpu_dispatch="grouped")
# x: float32 CPU tensor of shape [batch, 64], pixels normalized by 16.
logits = engine.predict(x)  # independent CPU output tensor

# The same engine supports differentiable training.
engine.train()
optimizer = torch.optim.AdamW(engine.parameters(), lr=0.002, foreach=False)
optimizer.zero_grad(set_to_none=True)
logits = engine(x)
loss = torch.nn.functional.cross_entropy(logits, targets.to(logits.device))
loss.backward()
optimizer.step()
```

## Reproduce in a new output folder

```powershell
# Check the runtime, cross-device gradients, pinned buffers and RL boundaries.
..\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider

# The reference checkpoints and fixed digit data already exist locally.
..\.venv\Scripts\python.exe run.py prepare

# Optional: redo the CPU pilot; main timing uses choices from runs/pilot/pilot.json.
..\.venv\Scripts\python.exe run.py pilot --output runs/pilot

..\.venv\Scripts\python.exe run.py benchmark --output runs/repeat
..\.venv\Scripts\python.exe run.py train --output runs/repeat --seeds 7 19 42 --epochs 40
..\.venv\Scripts\python.exe run.py rl --output runs/repeat --seeds 7 19 42 --updates 40
..\.venv\Scripts\python.exe -m bezier_engine.memory --output runs/repeat
..\.venv\Scripts\python.exe run.py verify --output runs/repeat
..\.venv\Scripts\python.exe run.py report --output runs/repeat
```

Run performance stages sequentially with other compute jobs idle. The reporting template describes the delivered 40-epoch/40-update study; custom budgets are recorded in raw settings and require corresponding report-text changes. Completed training jobs can resume only with matching settings. Source snapshots and hashes are frozen for each main stage; changed sources require a new output folder.

## Artifacts and limitations

`checkpoints/` holds the two copied reference models; `data/` holds the original fixed digit split. `runs/main/` contains 54 inference configurations (including pinned variants), five raw timing rounds per configuration, transfer microbenchmarks, 12 classifier checkpoints, 15 RL checkpoints, policies' exact greedy trajectories, histories, manifests and independent reload verification. `runs/pilot/` retains CPU timing choices; `runs/rl_pilot/` retains the separate seed-123 learning pilot. The 1024-wide stress workload is untrained and only tests execution.

This is an eager FP32 engineering prototype on one Intel Core Ultra 7 265KF / RTX 5090 system. It is not fused CUDA kernels, an inference compiler, distributed RL, or a proof that CPU/GPU splitting cannot help other workloads. Test data is reused from earlier project studies. Peak allocator memory is distinct from physical memory traffic, total device memory and power. The CPU actor / GPU learner is synchronous PPO with weight synchronization after each update, not IMPALA or an asynchronous pipeline.
