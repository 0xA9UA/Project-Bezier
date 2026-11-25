# Supplementary execution scaling

Fifteen configurations, random weights, seed 2026. Widths 256, 512 and 1024; 32, 64 and 64 stored experts, respectively, except the compact residual with eight. All routed models perform eight expert transformations per input; the full residual executes every stored block. Reported timings are medians of five synchronized wall-clock repeats of 50 forwards after warmup, including routing and dispatch, excluding host transfer, preprocessing, loading and startup. Each configuration ran sequentially after the main experiment released the GPU.

**Untrained weights: no quality or knowledge claim.**

| Model | Width | Stored experts | Parameters | FP32 state MiB | Active expert calls | Batch 1 ms | Batch 128 ms | Accounted inference MiB, B128 |
|---|---|---|---|---|---|---|---|---|
| Bézier | 256 | 32 | 2,331,707 | 8.895 | 8 | 0.967 | 1.024 | 42.23 |
| Straight | 256 | 32 | 2,331,509 | 8.894 | 8 | 1.431 | 0.991 | 42.23 |
| Ordinary MoE | 256 | 32 | 2,331,215 | 8.893 | 8 | 0.977 | 0.712 | 42.17 |
| Residual | 256 | 32 | 2,322,991 | 8.862 | 32 | 1.706 | 1.727 | 9.68 |
| Compact residual | 256 | 8 | 740,911 | 2.826 | 8 | 0.719 | 0.475 | 3.65 |
| Bézier | 512 | 64 | 17,270,427 | 65.881 | 8 | 0.972 | 1.607 | 196.15 |
| Straight | 512 | 64 | 17,270,229 | 65.881 | 8 | 0.906 | 1.592 | 196.15 |
| Ordinary MoE | 512 | 64 | 17,286,255 | 65.942 | 8 | 0.608 | 1.677 | 196.11 |
| Residual | 512 | 64 | 17,253,423 | 65.817 | 64 | 3.218 | 3.416 | 67.07 |
| Compact residual | 512 | 8 | 2,530,351 | 9.653 | 8 | 0.452 | 0.474 | 10.91 |
| Bézier | 1024 | 64 | 68,094,619 | 259.760 | 8 | 0.969 | 5.767 | 775.78 |
| Straight | 1024 | 64 | 68,094,421 | 259.760 | 8 | 1.475 | 5.775 | 775.78 |
| Ordinary MoE | 1024 | 64 | 68,126,831 | 259.883 | 8 | 0.979 | 6.485 | 775.80 |
| Residual | 1024 | 64 | 68,061,231 | 259.633 | 64 | 3.012 | 3.578 | 261.77 |
| Compact residual | 1024 | 8 | 9,254,959 | 35.305 | 8 | 0.727 | 0.488 | 37.44 |

These models have not been trained. Speed at the same nominal capacity does not establish speed at the same quality. All experts remain stored even when inactive. State sizes count tensors in float32; they are not checkpoint-file sizes. Accounted inference memory adds state, input and the forward allocator increment; it is not physical memory traffic. The dispatcher gathers per-input expert matrices, so larger batches can incur substantial temporary memory. These measurements characterize this eager implementation on this machine, not an optimal kernel or a theoretical limit.

![Execution scaling](execution_scaling.png)

See `manifest.json` for environment, settings and source hashes; `results.json` contains every repeat and memory measurement.