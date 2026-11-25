# Original sphere and scaling studies

`bezier_sphere` · Foundations

Test whether input-dependent cubic Bezier routes through a sphere can select useful expert computation, then measure accuracy, storage, speed and robustness against simpler controls.

## Saved status

Completed local proof of concept, trained scaling study, untrained execution sweep and algebra checks; no overall geometry advantage established.

## Work completed or attempted

- Implemented learned surface entry/exit, interior control points, sparse top-k expert dispatch, a carried feature vector and a trained decoder; added a scalar Bezier-transfer variant.
- Trained five digit-classification architectures across three seeds on a fixed 1,078/359/360 split; saved checkpoints, route interventions, CPU/GPU verification and an offline viewer.
- Completed 92 scaling runs using EMNIST, up to 100,000 training images, and synthetic association recall up to 16,384 facts; compared routed, full-residual and compact-residual models.
- Benchmarked 15 untrained execution configurations up to 68.1 million parameters; derived and numerically checked routing factorization, structured weight sharing and restricted commuting dynamics.
- Published architecture, reproduction and methodology documents plus shared activation and recurrent-computation visualizations.

## Findings

**Measured.** Original digits: Bezier routing achieved 96.76% mean accuracy across seeds 7, 19 and 42, versus 96.67% for residual and 97.13% for straight routing. Bezier batch-one latency was 0.992 ms versus 0.471 ms residual; no general accuracy or speed advantage was established.

[S003](../evidence/runs/digits/summary.md) · [S017](../evidence/runs/digits/results.json)

**Reported.** Routes affect computation: shuffling routes lowered mean digit accuracy from 96.76% to 88.52%, bypassing interior transformations to 64.91%, and straightening trained curves to 96.30%. These are interventions on trained models, not retrained controls.

[S001](../evidence/README.md) · [S017](../evidence/runs/digits/results.json)

**Measured.** At the largest trained EMNIST setting, Bezier used 8.900 MiB at 83.59% accuracy and 1.234 ms batch-one latency; compact residual used 2.830 MiB at 84.04% and 0.457 ms. Means cover two seeds and 18,800 test images.

[S005](../evidence/runs/scaling/report.md) · [S006](../evidence/runs/scaling/summary.json)

**Measured.** At 16,384 synthetic associations, medium Bezier reached 99.98% fresh-query recall in 1.130 MiB; medium compact residual reached 100% in 0.605 MiB. Stronger-noise recall was 69.43% versus 61.06%, showing a task-specific tradeoff rather than a universal storage/reliability win.

[S005](../evidence/runs/scaling/report.md) · [S006](../evidence/runs/scaling/summary.json)

**Measured.** The saved verification covers all 92 scaling checkpoints and passes source, data, partition, checkpoint and report-link checks. The original digit verification separately reloads all 15 checkpoints. These are recorded checks, not tests rerun for this index.

[S007](../evidence/runs/scaling/verification.json) · [S004](../evidence/runs/digits/verification.json)

**Measured.** At 68.1M random-weight parameters, Bezier versus full residual latency was 0.969 versus 3.012 ms at batch one, but 5.767 versus 3.578 ms at batch 128. Accounted batch-128 memory was 775.78 versus 261.77 MiB. Sparse computation did not remove stored weights or dispatcher allocations.

[S008](../evidence/runs/scaling_execution/report.md) · [S018](../evidence/runs/scaling_execution/results.json)

**Measured.** Algebra checks gave routing-score rank four and 2.66e-15 reconstruction error. A deliberately cubic 64-matrix bank compressed 16x with near-zero error, while independent random matrices incurred 0.968 relative error. The 16x result is bank-only and depends on constructed structure.

[S010](../evidence/runs/theory_checks/results.json) · [S009](../evidence/docs/mathematical_benefits.md)

## Limits and open questions

- Digit results use one fixed small split; scaling uses two seeds and one image domain. Synthetic known-key recall is not natural-language knowledge.
- Timing is eager float32 on the recorded RTX 5090 with device-resident inputs; transfer, preprocessing, startup and loading are excluded.
- The 68.1M execution sweep is untrained. Algebraic constructions do not establish real-world model compression or language-model quality.
- Additional sampling steps change the implemented model; reversing a drawn route does not invert computation.

## Recommended next steps

- Recommendation: retain matched-accuracy compact residual and ordinary routing controls when evaluating a new geometric architecture.
- Recommendation: improve dispatch before drawing broader throughput conclusions, then test additional tasks and seeds.

## Sources

- [Workspace overview and usage](../evidence/README.md)
- [Architecture and assumptions](../evidence/docs/architecture.md)
- [Digit experiment summary](../evidence/runs/digits/summary.md)
- [Saved digit verification](../evidence/runs/digits/verification.json)
- [92-run scaling report](../evidence/runs/scaling/report.md)
- [Scaling aggregates](../evidence/runs/scaling/summary.json)
- [Saved scaling verification](../evidence/runs/scaling/verification.json)
- [Untrained execution sweep](../evidence/runs/scaling_execution/report.md)
- [Theory and restricted constructions](../evidence/docs/mathematical_benefits.md)
- [Numerical algebra checks](../evidence/runs/theory_checks/results.json)

## Related projects

- [Bezier-Routing](Bezier-Routing.md)
- [Bezier-Explore](Bezier-Explore.md)
- [bezier_nested_core](bezier_nested_core.md)

Source reports and algebra checks dated November 23, 2026; indexed November 25, 2026.

[Back to the project directory](README.md)
