# Bezier-Steer

`Bezier-Steer` · Routing and reasoning

Measure whether curve-guided internal computation produces more distinct correct Boolean programs for a complete request budget.

## Saved status

Completed generative steering study; small steering effect, no overall advantage.

## Work completed or attempted

- Trained five architectures across three seeds on disjoint truth-table functions; evaluated 2/4/8 internal steps, latent/route interventions, candidate sampling, verification, and structural deduplication.
- Built an exhaustive 16-input correctness verifier, saved generated programs and routes, and exported verified Python alternatives.

## Findings

**Measured.** At 64 attempts per task, Bezier solved 85.29% with 3.217 distinct correct programs/task, 5.81 ms complete request time, and 1.278 MiB checkpoint. Fixed routing solved 86.91%, yielded 3.473 alternatives, and took 4.91 ms at 1.243 MiB.

[S065](../evidence/Bezier-Steer/README.md) · [S068](../evidence/Bezier-Steer/runs/main/summary.json)

**Measured.** Only 7.63% of raw Bezier candidates were correct. The 85.29% task solve rate is success in up to 64 attempts, not single-attempt accuracy; the difference from fixed routing was -1.63 percentage points with exploratory 95% interval [-4.36,+1.04].

[S065](../evidence/Bezier-Steer/README.md)

**Measured.** With hidden state fixed and deterministic decoding, 64 curves solved 14.71% versus 10.55% for one route, but only 0.52% of tasks produced two or more distinct correct structures from curve changes alone. Most variety came from token sampling.

[S065](../evidence/Bezier-Steer/README.md)

**Measured.** Bezier 2/4/8 updates produced 85.03%/85.29%/85.09% solve rates while request time increased 4.92/5.81/7.11 ms. No efficiency, correctness, or storage advantage was demonstrated.

[S065](../evidence/Bezier-Steer/README.md)

## Limits and open questions

- Finite four-input Boolean program synthesis; grammar enforces syntax and exhaustive verification applies only to this domain.
- Three seeds, 512 held-out functions, and post-training curve-only interventions do not establish general coding or reasoning performance.
- Same candidate count is not identical compute cost; timings include failures, verification, and deduplication but exclude startup/loading.

## Recommended next steps

- Recommendation: Evaluate explicit curve-only training objectives and retain verifier-aware full request cost as the comparison unit.

## Sources

- [Findings and protocol](../evidence/Bezier-Steer/README.md)
- [Complete study](../evidence/Bezier-Steer/runs/main/report.html)
- [Verified alternatives](../evidence/Bezier-Steer/runs/main/examples.html)
- [Metrics and uncertainty](../evidence/Bezier-Steer/runs/main/summary.json)
- [Saved verification](../evidence/Bezier-Steer/runs/main/verification.json)

## Related projects

- [bezier_sphere](bezier_sphere.md)
- [Bezier-Think](Bezier-Think.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
