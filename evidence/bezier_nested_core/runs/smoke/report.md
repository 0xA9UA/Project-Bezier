# Nested-sphere experiment run

Status: **complete**. Scientific status: **pilot_only**.

Tier 1; device `cuda`; 6 completed arm/seed runs.
Configured updates per run: 24. Evaluation prefix cap per split: 64.

A short or capped run verifies the pipeline. It does not establish the proposed accuracy benefit.

## Parameter sharing and 16-visit evaluation

| Seed | Arm | Parameters | Added checker parameters | Test NLL, full turns | Mean turns |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | clean-outer | 191,372 | 0 | 4.084265 | 0.000 |
| 7 | unshared-core | 192,429 | 1057 | 4.118100 | 0.844 |
| 7 | nested-core-serial | 191,372 | 0 | 4.105450 | 4.234 |
| 7 | nested-core-concurrent | 191,372 | 0 | 4.101342 | 2.844 |
| 7 | ablation-geom-masked | 191,372 | 0 | 4.098662 | 1.062 |
| 7 | ablation-entry-state | 191,372 | 0 | 4.118800 | 3.688 |

NLL is in nats per next byte; lower is better. Each arm tunes its decision margin on validation only.

## Paired comparisons for the primary concurrent arm

Positive gain means the full-turn concurrent model has lower NLL. Intervals resample files (10,000 draws by default).

| Seed | Comparison at 16 visits | Mean NLL gain | 95% interval |
| ---: | --- | ---: | --- |
| 7 | full_vs_zero | 0.012108 | [-0.010313, 0.034523] |
| 7 | full_vs_one | -0.001781 | [-0.019655, 0.018584] |
| 7 | full_locked_vs_one | -0.001781 | [-0.019655, 0.018584] |
| 7 | full_vs_clean | -0.017078 | [-0.046866, 0.015950] |

## Concurrent execution and isolated resource measurements

Eager kernel overlap observed: **True**. Captured graph overlap observed: **True**. Graph without checker overlap: **False**.

Profiler overlap establishes simultaneous kernel execution, not a speedup. Timings below use separate unprofiled forwards. The pipeline reference reuses the concurrent checkpoint and the identical delayed policy to isolate scheduling effects.

| Execution | Batch | Median ms | p95 ms | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| clean-outer | 1 | 6.693 | 8.255 | 17.251 |
| clean-outer | 16 | 7.681 | 8.098 | 22.998 |
| nested-core-serial | 1 | 20.737 | 30.695 | 18.251 |
| nested-core-serial | 16 | 21.032 | 32.172 | 22.998 |
| nested-core-concurrent | 1 | 21.172 | 22.002 | 26.376 |
| nested-core-concurrent | 16 | 21.374 | 21.849 | 32.123 |
| nested-core-pipeline-reference | 1 | 19.227 | 20.398 | 18.251 |
| nested-core-pipeline-reference | 16 | 13.531 | 20.871 | 22.998 |
| clean-outer-graph | 1 | 0.660 | 0.661 | 16.986 |
| clean-outer-graph | 16 | 0.732 | 0.733 | 19.045 |
| nested-core-serial-graph | 1 | 2.348 | 2.665 | 18.986 |
| nested-core-serial-graph | 16 | 2.537 | 2.618 | 19.045 |
| nested-core-concurrent-graph | 1 | 1.285 | 1.292 | 26.111 |
| nested-core-concurrent-graph | 16 | 1.397 | 1.413 | 28.170 |
| nested-core-pipeline-reference-graph | 1 | 1.509 | 1.521 | 18.986 |
| nested-core-pipeline-reference-graph | 16 | 1.639 | 1.653 | 19.045 |

Inference measurements use fp32, 50 warmup forwards, and 100 timed forwards per entry. Each arm has isolated allocator accounting. Graph rows include their graph-pool memory.


## Evidence files

- `suite.json`: configuration, status, summary measurements, success gates.
- `seed*/ARM/final.pt`: final requested training state, including snapshot and optimizer.
- `seed*/ARM/train.jsonl`: next-byte loss, checker loss, labels, gradient norms and hardware telemetry.
- `seed*/ARM/audits.json`: parameter identity, actual participation, and outer-only cross-masking.
- `seed*/ARM/evaluation/`: validation margins, paired per-file test losses, confidence intervals and traces.
- `seed*/ARM/trajectory.png`: measured visit positions inside the two spheres.

- `seed*/ARM/trajectory_multiturn.png`: examples selected for 16 visits and higher turn counts; not a representative frequency sample.

## Interpretation

The core reuses outer weight storage, so shared arms add no learned parameters. The private-core control uses the same delayed schedule and adds its own checker weights. CUDA streams, graph capture, actual kernel overlap, elapsed latency and peak memory are separate measurements.

The full study remains unrun unless this report explicitly records the complete configured seed/arm/depth matrix, all requested updates and uncapped held-out evaluation. A statistical pass also requires reviewing shared-unit participation, cross-masking, locked-margin repeated steering and retrained ablations.
