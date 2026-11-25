<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../../PUBLICATION_SCOPE.md) · [Reader guide](../../../READ_THIS_FIRST.md).

# Bézier scaling experiment

92 preregistered runs, seeds 7 and 19. EMNIST has 100,000 training / 12,800 validation / 18,800 official test images; the medium-size data sweep uses nested 10,000 and 40,000 training subsets. Fact tables contain 1,024, 4,096 or 16,384 independent associations. Twenty image epochs or 2,000 fact updates per run, identical optimizer settings and batch size. Checkpoints are selected by validation loss. Test labels do not select training settings or checkpoints. Reported table values are means over two training seeds; plot error bars show one sample standard deviation across those seeds.

GPU timings use eager float32 with TF32 disabled, warmup and synchronized wall-clock intervals; routing and dispatch are included. Batch timings exclude input transfer, preprocessing, model loading and startup. Models were evaluated sequentially without live optimizers. Batch 256 latency is per batch. CUDA allocator increments and accounted inference memory are in raw JSON; they are not physical memory traffic. Checkpoint sizes include all experts and inference metadata, excluding training data, optimizer and shared software runtime. Desktop scheduling and clocks caused visible timing variation; all raw repeat timings are retained.

## Efficiency at larger model sizes

| Model | Scale | Checkpoint MiB | Test accuracy | Expert calls | Batch 1 ms | Batch 256 ms | Training sec |
|---|---|---|---|---|---|---|---|
| Bézier | small | 0.346 | 83.59% | 8 | 0.954 | 1.318 | 45.9 |
| Straight | small | 0.345 | 83.12% | 8 | 1.234 | 1.557 | 43.8 |
| Ordinary MoE | small | 0.338 | 83.26% | 8 | 0.594 | 0.817 | 35.2 |
| Residual | small | 0.335 | 84.19% | 8 | 0.418 | 0.531 | 27.0 |
| Bézier | medium | 1.441 | 83.81% | 8 | 0.952 | 1.010 | 46.4 |
| Straight | medium | 1.441 | 83.74% | 8 | 0.906 | 1.190 | 44.4 |
| Ordinary MoE | medium | 1.431 | 83.43% | 8 | 0.783 | 0.793 | 35.3 |
| Residual | medium | 1.423 | 84.33% | 16 | 0.880 | 0.864 | 47.5 |
| Compact residual | medium | 0.917 | 84.24% | 8 | 0.419 | 0.464 | 26.9 |
| Bézier + transfer | medium | 1.457 | 83.52% | 8 | 1.416 | 1.623 | 59.8 |
| Bézier | large | 8.900 | 83.59% | 8 | 1.234 | 0.998 | 48.5 |
| Straight | large | 8.899 | 84.13% | 8 | 0.933 | 0.974 | 46.3 |
| Ordinary MoE | large | 8.897 | 83.76% | 8 | 0.627 | 0.641 | 36.6 |
| Residual | large | 8.865 | 84.51% | 32 | 1.821 | 1.778 | 95.0 |
| Compact residual | large | 2.830 | 84.04% | 8 | 0.457 | 0.470 | 27.2 |

## Storage at matched validation quality

| Validation target | Model | Seed | Selected scale | Minimum tested MiB | Test accuracy |
|---|---|---|---|---|---|
| 80.00% | Bézier | 7 | small | 0.346 | 83.62% |
| 80.00% | Bézier | 19 | small | 0.346 | 83.55% |
| 80.00% | Straight | 7 | small | 0.345 | 82.76% |
| 80.00% | Straight | 19 | small | 0.345 | 83.48% |
| 80.00% | Ordinary MoE | 7 | small | 0.338 | 83.02% |
| 80.00% | Ordinary MoE | 19 | small | 0.338 | 83.49% |
| 80.00% | Residual | 7 | small | 0.335 | 84.24% |
| 80.00% | Residual | 19 | small | 0.335 | 84.14% |
| 80.00% | Compact residual | 7 | medium | 0.917 | 84.35% |
| 80.00% | Compact residual | 19 | medium | 0.917 | 84.13% |
| 80.00% | Bézier + transfer | 7 | medium | 1.457 | 83.78% |
| 80.00% | Bézier + transfer | 19 | medium | 1.457 | 83.27% |
| 85.00% | Bézier | 7 | not reached | not reached | not reached |
| 85.00% | Bézier | 19 | not reached | not reached | not reached |
| 85.00% | Straight | 7 | not reached | not reached | not reached |
| 85.00% | Straight | 19 | not reached | not reached | not reached |
| 85.00% | Ordinary MoE | 7 | not reached | not reached | not reached |
| 85.00% | Ordinary MoE | 19 | not reached | not reached | not reached |
| 85.00% | Residual | 7 | medium | 1.423 | 84.72% |
| 85.00% | Residual | 19 | not reached | not reached | not reached |
| 85.00% | Compact residual | 7 | not reached | not reached | not reached |
| 85.00% | Compact residual | 19 | not reached | not reached | not reached |
| 85.00% | Bézier + transfer | 7 | not reached | not reached | not reached |
| 85.00% | Bézier + transfer | 19 | not reached | not reached | not reached |

## Retention of 16,384 independent facts

| Model | Scale | MiB | Recall | Noisy recall | FP16 MiB | FP16 recall |
|---|---|---|---|---|---|---|
| Bézier | small | 0.190 | 35.55% | 19.40% | 0.098 | 35.54% |
| Straight | small | 0.190 | 34.69% | 18.97% | 0.097 | 34.68% |
| Ordinary MoE | small | 0.182 | 35.14% | 19.11% | 0.093 | 35.16% |
| Residual | small | 0.180 | 30.78% | 17.95% | 0.091 | 30.77% |
| Bézier | medium | 1.130 | 99.98% | 69.43% | 0.567 | 99.98% |
| Straight | medium | 1.129 | 99.98% | 68.97% | 0.567 | 99.98% |
| Ordinary MoE | medium | 1.119 | 99.98% | 77.34% | 0.562 | 99.98% |
| Residual | medium | 1.111 | 100.00% | 79.77% | 0.557 | 100.00% |
| Compact residual | medium | 0.605 | 100.00% | 61.06% | 0.304 | 100.00% |
| Bézier + transfer | medium | 1.145 | 99.95% | 59.89% | 0.575 | 99.95% |
| Bézier | large | 8.276 | 100.00% | 90.78% | 4.140 | 100.00% |
| Straight | large | 8.275 | 100.00% | 89.20% | 4.140 | 100.00% |
| Ordinary MoE | large | 8.273 | 99.99% | 94.12% | 4.139 | 99.99% |
| Residual | large | 8.241 | 100.00% | 99.22% | 4.122 | 100.00% |
| Compact residual | large | 2.206 | 100.00% | 96.52% | 1.105 | 100.00% |

## Reliability on the independent image test set

| Model | Scale | Clean | Noise | Shift right | Shift down | NLL | ECE | High-confidence errors |
|---|---|---|---|---|---|---|---|---|
| Bézier | small | 83.59% | 54.89% | 69.95% | 55.91% | 0.527 | 3.18% | 2.19% |
| Straight | small | 83.12% | 56.01% | 69.11% | 56.38% | 0.532 | 2.48% | 2.00% |
| Ordinary MoE | small | 83.26% | 52.24% | 69.19% | 55.93% | 0.528 | 3.18% | 2.15% |
| Residual | small | 84.19% | 51.48% | 69.99% | 57.53% | 0.505 | 2.83% | 2.15% |
| Bézier | medium | 83.81% | 62.94% | 70.79% | 59.77% | 0.503 | 2.53% | 2.19% |
| Straight | medium | 83.74% | 63.76% | 70.45% | 59.37% | 0.499 | 2.54% | 2.02% |
| Ordinary MoE | medium | 83.43% | 65.51% | 71.54% | 59.90% | 0.516 | 3.18% | 2.26% |
| Residual | medium | 84.33% | 64.24% | 73.18% | 60.35% | 0.489 | 2.76% | 2.16% |
| Compact residual | medium | 84.24% | 63.18% | 72.74% | 62.15% | 0.490 | 2.66% | 2.07% |
| Bézier + transfer | medium | 83.52% | 57.99% | 70.65% | 58.69% | 0.523 | 3.09% | 2.27% |
| Bézier | large | 83.59% | 64.25% | 72.98% | 61.28% | 0.491 | 3.20% | 2.05% |
| Straight | large | 84.13% | 62.72% | 73.33% | 63.07% | 0.489 | 3.41% | 2.35% |
| Ordinary MoE | large | 83.76% | 63.77% | 72.23% | 62.22% | 0.509 | 3.56% | 2.50% |
| Residual | large | 84.51% | 64.56% | 73.66% | 63.69% | 0.467 | 3.16% | 2.21% |
| Compact residual | large | 84.04% | 65.81% | 73.70% | 63.99% | 0.488 | 3.17% | 2.36% |

A concrete packed lookup reference for 16,384 arbitrary associations occupies 147,456 bytes (0.140625 MiB): eight bytes per explicit binary key plus one byte per value, before an in-memory index or shared runtime. It retrieves clean known keys exactly. This is a useful storage reference for arbitrary fact tables; it is not a learned generalization model. Structured knowledge may be compressible in ways that independent random values are not. The actual packed tables and dataset fingerprints are saved in the project.

## Fact capacity within the training budget

Selection uses validation recall. A largest qualified table is the largest tested size meeting the target, not a proven maximum capacity.

| Validation recall target | Model | Scale | Seed | Largest qualified table | Checkpoint MiB | Test recall |
|---|---|---|---|---|---|---|
| 95.00% | Bézier | small | 7 | 4096 | 0.190 | 99.98% |
| 99.00% | Bézier | small | 7 | 4096 | 0.190 | 99.98% |
| 95.00% | Straight | small | 7 | 4096 | 0.190 | 100.00% |
| 99.00% | Straight | small | 7 | 4096 | 0.190 | 100.00% |
| 95.00% | Ordinary MoE | small | 7 | 4096 | 0.182 | 99.98% |
| 99.00% | Ordinary MoE | small | 7 | 4096 | 0.182 | 99.98% |
| 95.00% | Residual | small | 7 | 4096 | 0.180 | 100.00% |
| 99.00% | Residual | small | 7 | 4096 | 0.180 | 100.00% |
| 95.00% | Straight | small | 19 | 4096 | 0.190 | 100.00% |
| 99.00% | Straight | small | 19 | 4096 | 0.190 | 100.00% |
| 95.00% | Ordinary MoE | small | 19 | 4096 | 0.182 | 100.00% |
| 99.00% | Ordinary MoE | small | 19 | 4096 | 0.182 | 100.00% |
| 95.00% | Residual | small | 19 | 4096 | 0.180 | 100.00% |
| 99.00% | Residual | small | 19 | 4096 | 0.180 | 100.00% |
| 95.00% | Bézier | small | 19 | 4096 | 0.190 | 100.00% |
| 99.00% | Bézier | small | 19 | 4096 | 0.190 | 100.00% |
| 95.00% | Bézier | medium | 7 | 16384 | 1.130 | 99.98% |
| 99.00% | Bézier | medium | 7 | 16384 | 1.130 | 99.98% |
| 95.00% | Straight | medium | 7 | 16384 | 1.129 | 99.99% |
| 99.00% | Straight | medium | 7 | 16384 | 1.129 | 99.99% |
| 95.00% | Ordinary MoE | medium | 7 | 16384 | 1.119 | 99.98% |
| 99.00% | Ordinary MoE | medium | 7 | 16384 | 1.119 | 99.98% |
| 95.00% | Residual | medium | 7 | 16384 | 1.111 | 100.00% |
| 99.00% | Residual | medium | 7 | 16384 | 1.111 | 100.00% |
| 95.00% | Straight | medium | 19 | 16384 | 1.129 | 99.98% |
| 99.00% | Straight | medium | 19 | 16384 | 1.129 | 99.98% |
| 95.00% | Ordinary MoE | medium | 19 | 16384 | 1.119 | 99.98% |
| 99.00% | Ordinary MoE | medium | 19 | 16384 | 1.119 | 99.98% |
| 95.00% | Residual | medium | 19 | 16384 | 1.111 | 100.00% |
| 99.00% | Residual | medium | 19 | 16384 | 1.111 | 100.00% |
| 95.00% | Bézier | medium | 19 | 16384 | 1.130 | 99.98% |
| 99.00% | Bézier | medium | 19 | 16384 | 1.130 | 99.98% |
| 95.00% | Bézier | large | 7 | 16384 | 8.276 | 100.00% |
| 99.00% | Bézier | large | 7 | 16384 | 8.276 | 100.00% |
| 95.00% | Straight | large | 7 | 16384 | 8.275 | 99.99% |
| 99.00% | Straight | large | 7 | 16384 | 8.275 | 99.99% |
| 95.00% | Ordinary MoE | large | 7 | 16384 | 8.273 | 99.99% |
| 99.00% | Ordinary MoE | large | 7 | 16384 | 8.273 | 99.99% |
| 95.00% | Residual | large | 7 | 16384 | 8.241 | 100.00% |
| 99.00% | Residual | large | 7 | 16384 | 8.241 | 100.00% |
| 95.00% | Straight | large | 19 | 16384 | 8.275 | 100.00% |
| 99.00% | Straight | large | 19 | 16384 | 8.275 | 100.00% |
| 95.00% | Ordinary MoE | large | 19 | 16384 | 8.273 | 99.99% |
| 99.00% | Ordinary MoE | large | 19 | 16384 | 8.273 | 99.99% |
| 95.00% | Residual | large | 19 | 16384 | 8.241 | 100.00% |
| 99.00% | Residual | large | 19 | 16384 | 8.241 | 100.00% |
| 95.00% | Bézier | large | 19 | 16384 | 8.276 | 99.99% |
| 99.00% | Bézier | large | 19 | 16384 | 8.276 | 99.99% |
| 95.00% | Compact residual | medium | 7 | 16384 | 0.605 | 100.00% |
| 99.00% | Compact residual | medium | 7 | 16384 | 0.605 | 100.00% |
| 95.00% | Compact residual | medium | 19 | 16384 | 0.605 | 100.00% |
| 99.00% | Compact residual | medium | 19 | 16384 | 0.605 | 100.00% |
| 95.00% | Compact residual | large | 7 | 16384 | 2.206 | 100.00% |
| 99.00% | Compact residual | large | 7 | 16384 | 2.206 | 100.00% |
| 95.00% | Compact residual | large | 19 | 16384 | 2.206 | 100.00% |
| 99.00% | Compact residual | large | 19 | 16384 | 2.206 | 100.00% |
| 95.00% | Bézier + transfer | medium | 7 | 16384 | 1.145 | 99.94% |
| 99.00% | Bézier + transfer | medium | 7 | 16384 | 1.145 | 99.94% |
| 95.00% | Bézier + transfer | medium | 19 | 16384 | 1.145 | 99.96% |
| 99.00% | Bézier + transfer | medium | 19 | 16384 | 1.145 | 99.96% |

## Interpretation limits

This study covers one image domain and a synthetic memory task, with models up to 2.33 million parameters. It does not measure language-model knowledge, reasoning, or billion-parameter scaling. Two seeds cannot establish a universal reliability advantage. Fact recall concerns known keys with new query noise, not facts never presented in training. Equal epoch/update budgets are not equal training FLOPs, and the final learning curves can remain unfinished. The residual comparator with all blocks has more active compute at larger sizes; the compact residual uses eight blocks. Both are included. Inactive experts still occupy storage. Matched-quality storage means the smallest tested configuration satisfying a validation threshold, not a globally optimized minimum. FP16 results quantize stored weights then compute in float32; they do not claim FP16 execution speed. Confidence is uncalibrated. Gaussian noise and shifts are limited stress tests, not a guarantee of general reliability.

See summary.json for paired bootstrap intervals, per-seed threshold selections, and all aggregated measurements.

![Scaling comparison](scaling_overview.png)

![Fact recall](fact_recall.png)