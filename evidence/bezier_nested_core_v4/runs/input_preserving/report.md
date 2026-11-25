# Revision 4: input-preserving follow-up with a 5% shared core

Status: **complete**. Scientific status: **pilot_only**.

Tier 2; device `cuda`; 2 completed arm/seed runs.
Configured updates per run: 2256. Evaluation prefix cap per split: none (complete split).

The entire central region contains 235,210 learned parameters, approximately 5% of the shared model. All its tensors are reused for outer computation, initial-route control, redirect ranking and direct Boolean reasoning. There is no larger central expert.

All requested optimizer updates are retained in final.pt. Evaluation uses best.pt selected by validation loss during joint training. Test and longer-program labels never select checkpoints, schedules or margins.

This bounded scaling study does not establish the full multi-seed, seven-arm research protocol. The earlier 128-update pilot also used less training data and a smaller core fraction; comparison with that pilot does not isolate a single change.

## Parameter sharing and 16-visit evaluation

| Seed | Arm | Parameters | Added checker parameters | Selected update | Test NLL, full turns | Mean turns |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | clean-outer | 4,704,970 | 0 | 2256 | 0.657949 | 0.000 |
| 7 | nested-core-concurrent | 4,704,970 | 0 | 1756 | 0.646470 | 6.055 |

NLL is in nats per answer byte; lower is better. Each arm tunes its decision margin on validation only.

## Reasoning and control competence

Direct core reasoning uses only the compact core and parameter-free syntax inputs. It cannot use the outer encoder or shell. Each complete program has balanced answers, so a constant answer is 50% accurate.

| Arm | Outer answer accuracy, V16 | Core-only accuracy | Core binary NLL | Longer-program core accuracy | Local control regret (nats) |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean-outer | 57.03% | 62.50% | 0.639612 | 50.59% | 0.058330 |
| nested-core-concurrent | 57.62% | 59.57% | 0.645263 | 49.80% | 0.019460 |

Local control regret compares the selected continuation against the best actually evaluated continuation, including Stay, at the first eligible checkpoint. It uses matched remaining budgets and is measured offline. Longer-program results use validation-selected margins without retuning.

## Paired comparisons for the primary concurrent arm

Positive gain means the full-turn concurrent model has lower NLL. Intervals resample files (10,000 draws by default).

| Seed | Comparison at 16 visits | Mean NLL gain | 95% interval |
| ---: | --- | ---: | --- |
| 7 | full_vs_zero | 0.010572 | [0.008350, 0.012941] |
| 7 | full_vs_one | 0.004572 | [0.003005, 0.006242] |
| 7 | full_locked_vs_one | 0.004572 | [0.003005, 0.006242] |
| 7 | full_vs_clean | 0.011479 | [0.001261, 0.021992] |

Whole-core budget and sharing audit: **True**. Every complete core tensor receives gradients in each role: **True**. Control gradients reach the outer encoder: **True**.

Combined reasoning/control/routing criterion passed on this evaluation sample: **True**.


## Concurrent execution and isolated resource measurements

Eager kernel overlap observed: **True**. Captured graph overlap observed: **True**. Graph without checker overlap: **True**.

Profiler overlap establishes simultaneous kernel execution, not a speedup. Timings below use separate unprofiled forwards. The pipeline reference reuses the concurrent checkpoint and the identical delayed policy to isolate scheduling effects.

| Execution | Batch | Median ms | p95 ms | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| clean-outer | 1 | 9.436 | 14.334 | 36.597 |
| clean-outer | 16 | 14.686 | 15.084 | 76.793 |
| nested-core-concurrent | 1 | 18.983 | 27.887 | 45.722 |
| nested-core-concurrent | 16 | 18.985 | 28.547 | 85.988 |
| nested-core-pipeline-reference | 1 | 18.999 | 26.671 | 37.597 |
| nested-core-pipeline-reference | 16 | 17.846 | 27.625 | 76.863 |
| clean-outer-graph | 1 | 1.377 | 1.379 | 34.781 |
| clean-outer-graph | 16 | 2.144 | 2.314 | 36.839 |
| nested-core-concurrent-graph | 1 | 1.894 | 1.986 | 43.906 |
| nested-core-concurrent-graph | 16 | 2.780 | 3.082 | 45.964 |
| nested-core-pipeline-reference-graph | 1 | 2.297 | 2.465 | 36.781 |
| nested-core-pipeline-reference-graph | 16 | 3.182 | 3.189 | 36.839 |

Inference measurements use fp32, 50 warmup forwards, and 100 timed forwards per entry. Each arm has isolated allocator accounting. Graph rows include their graph-pool memory.

The no-checker graph also contains some overlapping kernels. Captured-graph overlap alone therefore cannot identify checker/outer overlap. The separate eager profile attributes overlapping kernels to the named checker and outer-update scopes.


## Evidence files

- `suite.json`: configuration, status, summary measurements, success gates.
- `seed*/ARM/final.pt`: final requested training state, including snapshot and optimizer.
- `seed*/ARM/train.jsonl`: next-byte loss, checker loss, labels, gradient norms and hardware telemetry.
- `seed*/ARM/audits.json`: parameter identity, actual participation, and outer-only cross-masking.
- `seed*/ARM/evaluation/`: validation margins, paired per-file test losses, confidence intervals and traces.
- `seed*/ARM/trajectory.png`: measured visit positions inside the two spheres.

- `seed*/ARM/trajectory_multiturn.png`: examples selected for 16 visits and higher turn counts; not a representative frequency sample.

## Interpretation

The entire core contains 235,210 learned parameters, including every weight and bias, with no additional central expert. Shared arms add zero checker parameters; the private control adds another 235,210. CUDA streams, graph capture, actual kernel overlap, elapsed latency and peak memory are separate measurements.

The full study remains unrun unless this report explicitly records the complete configured seed/arm/depth matrix, all requested updates and uncapped held-out evaluation. A statistical pass also requires reviewing shared-unit participation, cross-masking, locked-margin repeated steering and retrained ablations.
