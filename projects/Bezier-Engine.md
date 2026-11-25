# Bezier-Engine

`Bezier-Engine` · Routing and reasoning

Test CPU/GPU placements for the sphere model, including supervised learning, PPO, and GPU-produced curves evaluated on CPU.

## Saved status

Completed execution, training, RL, and deeper-producer pipeline studies.

## Work completed or attempted

- Implemented CPU-only, GPU-only, GPU-matrix/CPU-activation, GPU-route/CPU-neuron, and CPU-actor/GPU-learner schedules.
- Completed 54 inference configurations, 12 classifier and 15 PPO checkpoints; added pinned-buffer concurrent curve pipelines, tree workloads, and one-/four-layer control-point producers.
- Ran 180 early pipeline configurations and 216 fresh shallow/deep configurations; preserved contention notes and independent output checks.

## Findings

**Measured.** Both original splits reproduced outputs/gradients but neither beat the fastest single-device baseline across nine tested inference settings. Mean digit training was 2.24 s GPU-only versus 2.82 s CPU activations and 3.41 s CPU neurons.

[S050](../evidence/Bezier-Engine/README.md) · [S057](../evidence/Bezier-Engine/runs/main/report.md)

**Measured.** CPU actor/GPU learner PPO averaged 2.96 s versus 3.43 s GPU-only (about 14% less); all 15 finite-task policies reached 100% evaluation success. This is a small synchronous RL workload.

[S050](../evidence/Bezier-Engine/README.md) · [S057](../evidence/Bezier-Engine/runs/main/report.md)

**Measured.** For the idle four-layer producer with 8192 independent curves, full GPU took 1.472 ms, full-batch hybrid 1.877 ms, and best concurrent hybrid 4.030 ms. The dependent tree was 2.385, 3.066, and 15.141 ms respectively. No hybrid beat the fastest baseline in the tested grid.

[S050](../evidence/Bezier-Engine/README.md)

**Reported.** Fresh one-/four-layer comparisons passed ordered output checks across 216 configurations; 70 related regression checks were reported passed. Deep producer timings use random weights rather than learned predictive performance.

[S050](../evidence/Bezier-Engine/README.md)

## Limits and open questions

- One eager FP32 desktop implementation; transfer dependencies, scheduling, and untrained stress cases limit generalization.
- Original mixed-device inference is single-request/single-stream and not thread-safe; early pipeline pilots ran with competing GPU work.
- CPU actor/GPU learner is synchronous PPO, not a distributed or language-model RL result.

## Recommended next steps

- Recommendation: Focus further optimization on transfer amortization or independent workloads; benchmark any kernel/pipeline change under matched uncontended conditions.

## Sources

- [All execution studies](../evidence/Bezier-Engine/README.md)
- [Original split and RL report](../evidence/Bezier-Engine/runs/main/report.html)
- [27-checkpoint verification](../evidence/Bezier-Engine/runs/main/verification.json)
- [Four-layer idle study](../evidence/Bezier-Engine/runs/curve_pipeline_deep4_idle/report.html)
- [Shallow idle control](../evidence/Bezier-Engine/runs/curve_pipeline_shallow_idle/report.html)

## Related projects

- [bezier_sphere](bezier_sphere.md)
- [Bezier-Layers](Bezier-Layers.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
