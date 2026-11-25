# Revision 2: complete shared-core reasoning pilot

Status: **complete**. Scientific status: **pilot_only**.

Tier 1; device `cuda`; 7 completed arm/seed runs.
Configured updates per run: 128. Evaluation prefix cap per split: 128.

The entire central region is a fixed 1,312-parameter core. All its tensors are reused for outer computation, initial-route control, redirect ranking and direct Boolean reasoning. There is no larger central expert.

A short or capped run verifies the pipeline. It does not establish the proposed accuracy benefit. The v1 Python next-byte task and v2 Boolean reasoning task are not directly comparable.

## Parameter sharing and 16-visit evaluation

| Seed | Arm | Parameters | Added checker parameters | Test NLL, full turns | Mean turns |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | clean-outer | 175,328 | 0 | 0.703922 | 0.000 |
| 7 | unshared-core | 176,640 | 1312 | 0.702837 | 3.125 |
| 7 | nested-core-serial | 175,328 | 0 | 0.703165 | 0.000 |
| 7 | nested-core-concurrent | 175,328 | 0 | 0.704778 | 0.000 |
| 7 | ablation-geom-masked | 175,328 | 0 | 0.703269 | 7.000 |
| 7 | ablation-entry-state | 175,328 | 0 | 0.703171 | 7.000 |
| 7 | ablation-control-detached | 175,328 | 0 | 0.703982 | 0.000 |

NLL is in nats per answer byte; lower is better. Each arm tunes its decision margin on validation only.

## Reasoning and control competence

Direct core reasoning uses only the compact core and parameter-free syntax inputs. It cannot use the outer encoder or shell. Each complete program has balanced answers, so a constant answer is 50% accurate.

| Arm | Outer answer accuracy, V16 | Core-only accuracy | Core binary NLL | Longer-program core accuracy | Local control regret (nats) |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean-outer | 50.00% | 50.00% | 0.693387 | 46.88% | 0.014072 |
| unshared-core | 50.00% | 47.66% | 0.693198 | 43.75% | 0.077714 |
| nested-core-serial | 50.00% | 50.78% | 0.693229 | 44.53% | 0.014608 |
| nested-core-concurrent | 50.00% | 50.78% | 0.693326 | 46.88% | 0.000497 |
| ablation-geom-masked | 50.00% | 51.56% | 0.693356 | 45.31% | 0.005018 |
| ablation-entry-state | 50.00% | 50.78% | 0.693334 | 46.09% | 0.008164 |
| ablation-control-detached | 50.00% | 48.44% | 0.693514 | 45.31% | 0.009906 |

Local control regret compares the selected continuation against the best actually evaluated continuation, including Stay, at the first eligible checkpoint. It uses matched remaining budgets and is measured offline. Longer-program results use validation-selected margins without retuning.

## Paired comparisons for the primary concurrent arm

Positive gain means the full-turn concurrent model has lower NLL. Intervals resample files (10,000 draws by default).

| Seed | Comparison at 16 visits | Mean NLL gain | 95% interval |
| ---: | --- | ---: | --- |
| 7 | full_vs_zero | 0.000000 | [0.000000, 0.000000] |
| 7 | full_vs_one | 0.000000 | [0.000000, 0.000000] |
| 7 | full_locked_vs_one | 0.000000 | [0.000000, 0.000000] |
| 7 | full_vs_clean | -0.000855 | [-0.000983, -0.000739] |

Whole-core budget and sharing audit: **True**. Every complete core tensor receives gradients in each role: **True**. Control gradients reach the outer encoder: **True**.

Combined reasoning/control/routing criterion passed on this evaluation sample: **False**.

The primary concurrent arm accepted no redirects in the 16-visit held-out evaluation. Its full-, one- and zero-turn policies therefore produced the same predictions on this sample. This pilot has not demonstrated useful repeated steering.

Outer routing blends the two nearest region centers. The core has a fixed central parameter address, but can be selected from outside the inner radius; that radius is not a hard execution gate. The separate `routing_coverage.json` records actual core and shell use. This pilot concentrates computation on a small subset of regions and does not demonstrate broad routing diversity.

## Shared-core objective interaction

A separate CPU audit compares complete-core gradients on 32 held-out examples from 2 programs without updating weights. Negative cosine means two objectives locally push some shared weights in opposing directions. This single diagnostic batch does not establish the cause of any accuracy result.

| Objective pair | Gradient cosine |
| --- | ---: |
| outer_answer / direct_core_reasoning | 0.018573 |
| outer_answer / live_control | -0.030264 |
| direct_core_reasoning / live_control | -0.095453 |


## Concurrent execution and isolated resource measurements

Eager kernel overlap observed: **True**. Captured graph overlap observed: **True**. Graph without checker overlap: **False**.

Profiler overlap establishes simultaneous kernel execution, not a speedup. Timings below use separate unprofiled forwards. The pipeline reference reuses the concurrent checkpoint and the identical delayed policy to isolate scheduling effects.

| Execution | Batch | Median ms | p95 ms | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| clean-outer | 1 | 11.640 | 11.995 | 17.228 |
| clean-outer | 16 | 8.287 | 12.247 | 22.975 |
| nested-core-serial | 1 | 33.933 | 34.302 | 18.228 |
| nested-core-serial | 16 | 23.867 | 35.055 | 22.975 |
| nested-core-concurrent | 1 | 17.568 | 18.014 | 26.353 |
| nested-core-concurrent | 16 | 25.700 | 26.180 | 32.100 |
| nested-core-pipeline-reference | 1 | 15.888 | 16.366 | 18.228 |
| nested-core-pipeline-reference | 16 | 16.242 | 24.288 | 22.975 |
| clean-outer-graph | 1 | 0.965 | 0.967 | 16.963 |
| clean-outer-graph | 16 | 1.056 | 1.058 | 19.022 |
| nested-core-serial-graph | 1 | 2.697 | 2.705 | 18.963 |
| nested-core-serial-graph | 16 | 2.976 | 2.979 | 19.022 |
| nested-core-concurrent-graph | 1 | 1.511 | 1.513 | 26.088 |
| nested-core-concurrent-graph | 16 | 1.679 | 1.683 | 28.147 |
| nested-core-pipeline-reference-graph | 1 | 1.842 | 1.845 | 18.963 |
| nested-core-pipeline-reference-graph | 16 | 2.018 | 2.023 | 19.022 |

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

The entire core contains 1,312 learned parameters, including every weight and bias, with no additional central expert. Shared arms add zero checker parameters; the private control adds another 1,312. CUDA streams, graph capture, actual kernel overlap, elapsed latency and peak memory are separate measurements.

The full study remains unrun unless this report explicitly records the complete configured seed/arm/depth matrix, all requested updates and uncapped held-out evaluation. A statistical pass also requires reviewing shared-unit participation, cross-masking, locked-margin repeated steering and retrained ablations.
