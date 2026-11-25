<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Bezier-Layers

The follow-up [reversed experiment](REVERSED.md) adds GPU-selected layers with CPU
Bézier/expert computation, a trained 50.9-million-parameter EMNIST model and an
untrained roughly 270-million-parameter stress model. Its results are saved
separately in `runs/reversed_large`; the original study below remains available.

A working implementation of the proposed **CPU-selected z layer / GPU inference** engine. The CPU holds a layer catalogue and a small learned controller. The GPU retains the full activation, expert weights, nonlinear functions and matrix products. After each processing step, the CPU can select a new layer from an eight-float GPU summary.

## Measured result

Layer switching works. Across three trained seeds, the adaptive model switched layers during **6.20%** of held-out digit predictions. Family-selection accuracy improved from **91.48% at entry to 95.83% at the final hop**. Overall digit accuracy was **95.46%**, versus **95.56%** for separately trained entry-only selection. The paired uncertainty interval includes zero, so better final accuracy has not been established.

For example, a held-out digit 4 produced layer IDs **3 → 2 → 2 → 2**: the CPU first chose the family 6/7, then changed to 4/5 after a GPU update, and the model predicted 4.

The CPU-controlled implementation beat GPU hierarchical selection in one of nine median timing settings, but did not beat the fastest tested single-device baseline in any. Reducing the fine search space is a real algorithmic saving available to either CPU or GPU layer selection: the large routing stress test scored **128 local experts instead of 8,192 global experts** per hop. Dependent CPU/GPU communication remains a cost. The timing repeats show desktop variability; small differences should not be overinterpreted.

- [Full report, actual paths and timing tables](runs/main/report.html)
- [Plain-text findings](runs/main/report.md)
- [Layer catalogue](runs/main/catalog.json)
- [Independent verification](runs/main/verification.json)
- [Protocol and primary references](docs/protocol.md)

## Execution

GPU hidden state → learned 8-float summary → CPU layer choice → GPU cubic route and two selected experts → updated GPU hidden state → repeat.

Each z plane contains its own bank of expert matrices. The five trained planes are assigned digit families {0,1}, {2,3}, {4,5}, {6,7}, and {8,9}. A supervised auxiliary loss teaches the family controller; labels are not used during inference. These categories are intentionally assigned, not evidence that a pretrained model spontaneously separated its knowledge into semantic layers.

The layer ID selects an expert-bank offset and z coordinate. GPU expert geometry is cached, and each input-dependent Bézier proposal is evaluated on GPU. Two experts within the chosen layer transform the state. Repeating the CPU decision allows jumps in either z direction. The final drawn exit reaches the sphere surface but adds no extra learned processing operation.

| Mode | Layer selection | Fine routing, activations and matrices |
|---|---|---|
| `cpu_layers` | CPU, from a small GPU summary | GPU, within selected layer |
| `gpu_layers` | GPU, from a small CPU summary | CPU, within selected layer |
| `gpu` | GPU | GPU, within selected layer |
| `gpu_flat` | GPU | GPU, full expert scan followed by the same layer mask |
| `cpu` | CPU | CPU |

`gpu_flat` is an exactly equivalent, deliberately flat-search control; the essential CPU-placement comparison is against `gpu`. The layer catalogue stays in host RAM, but expert weights remain GPU resident in all GPU modes. This is not weight streaming, paging, or model compression. The GPU caches coordinates needed for its local geometry.

## Run the engine

From PowerShell in `WORKSPACE\Bezier-Layers`:

```powershell
..\.venv\Scripts\python.exe run.py demo --mode cpu_layers
..\.venv\Scripts\python.exe run.py demo --mode gpu
```

This predicts eight original test examples and prints labels, predictions, actual layer sequences, catalogue entries and transfer payloads. Predictions include errors. For a single digit input, four CPU decisions require five transfers in each direction including the initial input and final output; the total logical payload is 456 bytes. Each CPU decision receives 32 bytes and returns an eight-byte integer ID. Logical tensor-copy counts are not PCIe transaction counts.

Python API, with this folder on the Python path:

```python
from bezier_layers.model import load

engine = load("runs/main/trained/adaptive_s7/model.pt", "cpu_layers",
              decision_interval=1, pinned=True)
# x is a float32 CPU tensor [batch,64] with digit pixels divided by 16.
logits, trace = engine.predict(x, trace=True)
print(trace["layers"])       # one chosen z layer at each step
print(trace["experts"])      # actual expert IDs within those layers
print(trace["controls"])     # actual interior cubic Bézier controls
```

Set `decision_interval=2` to reconsider the layer every other hop. At interval 4 the layer stays fixed for the four-hop run. Those are approximations using the same trained weights; the report includes their accuracy changes. The separately trained entry-only models isolate a different question and remain available.

The runtime is single-request/single-stream. Pinned buffers synchronize before CPU reads and before reuse; they do not overlap dependent steps. Training uses differentiable copies and bypasses the geometry cache. `engine.train()` and `engine.load_model_state(state)` invalidate it. After manual parameter edits, call `engine.clear_inference_cache()`. Do not move a mixed-device engine wholesale with `.cuda()` or `.to()`; construct the desired execution mode instead.

## Reproduce

The parent `.venv` supplies all dependencies. No downloads or external accounts are needed. Use a new output directory for new runs, especially after source changes.

```powershell
..\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
..\.venv\Scripts\python.exe run.py train --output runs/repeat --seeds 7 19 42 --epochs 80
..\.venv\Scripts\python.exe run.py diagnostics --output runs/repeat

# Run timing after other Python GPU jobs finish. The script checks each round.
..\.venv\Scripts\python.exe run.py benchmark --output runs/repeat --uncached
..\.venv\Scripts\python.exe run.py benchmark --output runs/repeat

..\.venv\Scripts\python.exe run.py verify --output runs/repeat
..\.venv\Scripts\python.exe run.py report --output runs/repeat
```

The report generator describes the delivered three-seed, eighty-epoch study. For custom training budgets, update its prose and aggregation expectations accordingly. Training records are resumable only when settings and source snapshots match. Timing refuses to proceed when another Python GPU process is reported; unnamed or non-Python GPU applications may still affect measurements.

## Saved evidence

The main study retains six trained checkpoints; original predictions, expert choices and paths for all 360 test images; confusion matrices; interventions at three strengths; two cadence approximations; six illustrative switching examples; 63 uncached and 63 cached inference cases with five repeats each; process snapshots; and source hashes. The cached pass was an exploratory implementation refinement after the initial timings, without changing model weights or training data. Both passes remain available.

The original study passed **14 unit checks**; the reversed-engine extension expands this to **23**. All **six original checkpoints** reproduce predictions, layer IDs and expert IDs across the original **four execution modes**, and the follow-up verifies the new reversed mode. Geometry, gradient equivalence, runtime device placement, feedback interventions, cached/uncached equality and buffer lifetime checks are included.

Limits: one desktop CPU/GPU, one reused digit split, one assigned category scheme, hard coarse selection trained using family labels, eager FP32 execution and synthetic untrained scaling cases. There are no fused CUDA kernels, request pipelines, model paging, general language-model evaluation, or reinforcement-learning tests in this experiment. The design demonstrates dynamic content-associated layer selection; it has not demonstrated a dependable CPU-placement speed advantage or improved overall accuracy.
