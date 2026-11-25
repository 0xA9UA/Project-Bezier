<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Reversed split: GPU selects layers, CPU executes the sphere

The new `gpu_layers` mode reverses the earlier experiment. The GPU runs the learned
layer controller. The CPU runs encoding, normalization, summary projection, cubic
Bézier routing, within-layer expert selection, expert matrix products, SiLU and
decoding. After every step the CPU sends a new summary to the GPU and receives a
layer ID. The full state and expert weights remain in system RAM.

The completed evidence is in [the report](runs/reversed_large/report.html), with
[raw timings](runs/reversed_large/timing.json), [training results](runs/reversed_large/trained/result.json),
[placement and output checks](runs/reversed_large/execution_checks.json), and
[independent verification](runs/reversed_large/verification.json).

## Model sizes and tasks

The learned model has 50,915,372 FP32 parameters: width 512, bottleneck 256, 12
z layers, 16 experts per layer, two selected experts per step and four steps.
It is about 566 times the size of the earlier 89,968-parameter digit model. It was
trained on a fixed 23,500-example subset of 47-class EMNIST Balanced, selected by
validation loss and evaluated on 9,400 official-test examples. Its test accuracy
is 77.95%. All tested execution variants preserve its predictions and routes.

A second model has roughly 270 million parameters, width 1024, bottleneck 512,
32 layers and eight experts per layer. This model is untrained: it tests execution
and memory only. Neither model is a pretrained language model. No original Granite
artifact or other experiment has been modified.

Layers in the learned model have assigned groups of four consecutive class IDs
(three in the final layer). An auxiliary classification loss teaches the GPU
controller those assignments. Labels are never supplied during inference. The
[catalogue](runs/reversed_large/catalog.json) maps IDs to actual characters.

## Try the trained model

From PowerShell in `WORKSPACE\Bezier-Layers`:

```powershell
..\.venv\Scripts\python.exe run_reversed.py demo --mode gpu_layers --threads 8
..\.venv\Scripts\python.exe run_reversed.py demo --mode cpu --threads 8
..\.venv\Scripts\python.exe run_reversed.py demo --mode gpu --threads 8
```

The demo prints eight real held-out predictions (including mistakes), actual layer
sequences and logical transfer counts. The reversed mode sends eight FP32 values
per input per decision and receives one int64 ID: 160 bytes per input across four
decisions. There are four copies in each direction. No full activation vector or
expert weights cross devices during these inference steps.

```python
import torch
from bezier_layers.model import load

torch.set_num_threads(8)
engine = load("runs/reversed_large/trained/model.pt", "gpu_layers",
              cpu_dispatch="grouped", pinned=False)
# x: float32 CPU tensor [batch, 784], upright EMNIST images divided by 255.
logits, trace = engine.predict(x, trace=True)
print(trace["layers"])   # GPU choices after successive CPU updates
print(trace["experts"])  # CPU-selected experts within those layers
```

`cpu_dispatch="grouped"` reuses each selected expert matrix across its assigned
inputs, avoiding gathered matrix copies per input. `"gather"` reproduces the earlier
execution style. Both are tested for outputs and gradients. CPU-only execution
receives the same dispatch options. This is synchronous single-request code;
pinned buffers synchronize before reads and reuse and do not pipeline decisions.

Setting `decision_interval=2` or `4` asks the GPU less often. These are approximations
with measured accuracy in the report. Interval 4 fixes the layer for the entire
four-step prediction. All device modes use the same learned weights and geometry.
Do not move a mixed-device engine wholesale with `.to()` or `.cuda()`.

## Reproduce

Use a new output directory to retain the existing evidence. The existing parent
environment and local EMNIST archive supply everything; no downloads are required.

```powershell
..\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
..\.venv\Scripts\python.exe run_reversed.py train --output runs/reversed_repeat
..\.venv\Scripts\python.exe run_reversed.py diagnostics --output runs/reversed_repeat
..\.venv\Scripts\python.exe run_reversed.py benchmark --output runs/reversed_repeat

..\.venv\Scripts\python.exe run_reversed.py memory --output runs/reversed_repeat --workload trained_51m --mode cpu
..\.venv\Scripts\python.exe run_reversed.py memory --output runs/reversed_repeat --workload trained_51m --mode gpu
..\.venv\Scripts\python.exe run_reversed.py memory --output runs/reversed_repeat --workload trained_51m --mode gpu_layers
..\.venv\Scripts\python.exe run_reversed.py memory --output runs/reversed_repeat --workload stress_270m --mode cpu
..\.venv\Scripts\python.exe run_reversed.py memory --output runs/reversed_repeat --workload stress_270m --mode gpu
..\.venv\Scripts\python.exe run_reversed.py memory --output runs/reversed_repeat --workload stress_270m --mode gpu_layers

..\.venv\Scripts\python.exe run_reversed.py verify --output runs/reversed_repeat
..\.venv\Scripts\python.exe run_reversed.py report --output runs/reversed_repeat
```

Memory commands deliberately use separate processes. GPU tensor allocation excludes
CUDA context/driver overhead. Host working set includes runtime allocations; exact
weight placement is reported separately. The CPU-only memory measurement does not
initialize CUDA before sampling memory. GPU benchmarks should run without other Python GPU jobs; the
script checks at every round. Desktop and unreadable-process interference remain
possible. Timing compares one and eight CPU threads at batches 1, 16 and 128.

The [frozen protocol](docs/reversed_protocol.md) records the training budget and
comparison boundaries. Source snapshots, data selections, hashes, trained weights,
all predictions, routes, feedback interventions and raw timing repeats are saved.

This split relocates expert weights from VRAM to system RAM; it does not reduce
their total bytes. CPU-only execution is the control for whether GPU layer selection
adds useful performance. The report separates the memory result from speed and task
quality. No reinforcement-learning, language-model or training-speed benefit has
been established by this experiment.
