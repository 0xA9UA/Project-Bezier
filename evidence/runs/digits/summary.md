<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../../PUBLICATION_SCOPE.md) · [Reader guide](../../../READ_THIS_FIRST.md).

# Bézier sphere experiment

Fixed stratified 60/20/20 split, identical training budget, validation-loss checkpoint selection; seed variation is not independent-data uncertainty. All timing includes routing; inputs reside on device. Eager float32; no compilation or mixed precision.

| Model | Test accuracy (mean ± sample SD) | Parameters | Expert calls/sample | Batch 1 ms |
|---|---:|---:|---:|---:|
| Bézier routing | 96.76% ± 0.42 pp | 40,974 | 8 | 0.992 |
| Straight routing | 97.13% ± 0.85 pp | 40,776 | 8 | 0.946 |
| Ordinary experts | 96.94% ± 0.56 pp | 38,994 | 8 | 0.633 |
| Residual network | 96.67% ± 0.48 pp | 38,474 | 8 | 0.471 |
| Bézier routing + transfer | 96.85% ± 0.32 pp | 41,998 | 8 | 1.365 |

This is a small 8×8 digit-classification proof of concept, not a language model or evidence of a general inference speedup. Models share feature width, expert size and training budget; total parameters differ by router and transfer functions. The default residual baseline stores eight blocks and runs all eight; routed models perform four steps with two experts per step. Top-k routing computes all center distances but only selected expert transformations. Timings include routing and dispatch with inputs already on the device; they exclude startup, preprocessing and host transfers. Error bars describe initialization/training variation on one fixed split, not uncertainty across independent datasets. Reversing a drawn path does not invert the learned computation.

## Run details

```json
{
  "python": "3.14.3",
  "platform": "Windows-10-10.0.19045-SP0",
  "torch": "2.10.0+cu128",
  "cuda_runtime": "12.8",
  "device": "cuda",
  "precision": "float32; TF32 disabled; eager execution",
  "packages": {
    "numpy": "2.5.5",
    "scikit-learn": "1.9.1",
    "matplotlib": "3.11.2"
  },
  "gpu": "NVIDIA GeForce RTX 5090",
  "gpu_memory_bytes": 34190458880,
  "compute_capability": [
    12,
    0
  ]
}
```

See results.json for per-seed latency, accuracy, memory allocation and route interventions.