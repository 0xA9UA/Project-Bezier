# Bezier-Explore

`Bezier-Explore` · Compression and shared fields

Determine whether shared smooth processing fields and analytic integration can reduce storage and execution cost.

## Saved status

Completed 90-checkpoint study; conditional storage and analytic-compute gains demonstrated.

## Work completed or attempted

- Completed a 72-run main study and 18-run second-width recall study, covering handwriting, known-key associations, and synthetic smooth conditional transforms.
- Implemented shared nonlinear fields, commuting linear analytic fields, straight/constant/direct gates, and residual controls.

## Findings

**Measured.** On handwriting, shared Bezier used 510.4 KiB versus 1475.9 KiB for the original expert model (2.89x smaller), at 83.90% versus 83.88% accuracy. Constant gates reached 83.91% with still less storage, favoring weight sharing over a curvature-specific explanation.

[S027](../evidence/Bezier-Explore/RESULTS.md)

**Measured.** For 4,096 known random associations, shared Bezier achieved 100% recall at 6.05x less storage; straight, constant, direct-gate, and residual controls also reached 100%. At 16,384 associations the narrow shared model fell to 30.29%; width 256 reached 95.81% versus 99.988% for the original width-128 model.

[S027](../evidence/Bezier-Explore/RESULTS.md)

**Measured.** On the synthetic interpolation gap, analytic Bezier had normalized MSE 0.000496 versus 0.014439 for the original and used 76.9 versus 1076.8 KiB. Analytic straight was better still at 0.000386. The analytic evaluator was 1.63-3.16x faster than the smallest tested RK4 setting meeting relative output error <=1e-5.

[S027](../evidence/Bezier-Explore/RESULTS.md)

**Measured.** Robustness did not generally improve: stronger-noise recall was 74.48% versus 90.20% for 4,096 associations; noisy handwriting accuracy was 57.30% versus 61.72%. All 90 models reportedly reproduced saved test predictions.

[S027](../evidence/Bezier-Explore/RESULTS.md)

## Limits and open questions

- Two seeds; width sweep only for the larger recall task; no large language-model evaluation.
- Analytic integration requires commuting linear operators and polynomial rates; it is not a general nonlinear solver.
- Capacity-run latency was affected by another workload; compact fields trade capacity for storage.

## Recommended next steps

- Recommendation: Prioritize structured shared-field tasks; run quality-matched capacity/training sweeps before claiming general knowledge compression.

## Sources

- [Consolidated findings](../evidence/Bezier-Explore/RESULTS.md)
- [72-run study](../evidence/Bezier-Explore/runs/main/report.html)
- [18-run capacity extension](../evidence/Bezier-Explore/runs/capacity/report.html)
- [Field mathematics](../evidence/Bezier-Explore/docs/implementation_math.md)

## Related projects

- [bezier_sphere](bezier_sphere.md)
- [Bezier-Compress](Bezier-Compress.md)
- [Bezier-Granite](Bezier-Granite.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
