# Bezier-Routing

`Bezier-Routing` · Routing and reasoning

Test whether intermediate neural activations can redirect later Bezier routes, groups, and individual neurons.

## Saved status

Completed mechanism study; feedback works, general quality gain unestablished.

## Work completed or attempted

- Implemented individual-neuron, group, and hierarchical routing with matched entry-state controls, straight routes, and direct selection.
- Completed 24 training runs on the reused digits split; saved checkpoints, prediction traces, intervention paths, and geometry/reload verification.

## Findings

**Measured.** Perturbing two activation components after hop 1 changed 44.80% of later neuron-selection sets and 16.35% of later group sets in the feedback hierarchy; the frozen hierarchy changed neither. Feedback is causally active.

[S019](../evidence/Bezier-Routing/README.md) · [S021](../evidence/Bezier-Routing/runs/digits/summary.json)

**Measured.** Digit accuracy with versus without feedback was 95.37% versus 94.54% for individual neurons, 96.85% versus 95.93% for groups, and 94.35% versus 95.19% for the hierarchy. All paired-image 95% intervals include zero.

[S019](../evidence/Bezier-Routing/README.md) · [S026](../evidence/Bezier-Routing/runs/digits/report.md)

**Measured.** The trained feedback hierarchy fell from 94.35% to 89.54% when feedback was disabled and to 50.19% when neuron updates were bypassed. Reliance on a mechanism does not establish superiority to a separately trained control.

[S019](../evidence/Bezier-Routing/README.md)

**Reported.** All 24 saved checkpoints reproduced predictions and passed intervention/geometry checks; 13 unit checks were reported passed.

[S022](../evidence/Bezier-Routing/runs/digits/verification.json) · [S019](../evidence/Bezier-Routing/README.md)

## Limits and open questions

- Only three seeds and 360 reused held-out digit images; exploratory accuracy intervals exclude training-seed uncertainty.
- Group-only executes 96 neuron calls versus 24 for individual/hierarchical modes; no clean latency advantage is claimed.

## Recommended next steps

- Recommendation: Test fresh tasks with matched budgets and training-seed uncertainty before extending accuracy claims.

## Sources

- [Project overview](../evidence/Bezier-Routing/README.md)
- [Complete experiment report](../evidence/Bezier-Routing/runs/digits/report.html)
- [Aggregate metrics](../evidence/Bezier-Routing/runs/digits/summary.json)
- [Saved checkpoint verification](../evidence/Bezier-Routing/runs/digits/verification.json)
- [Design and prospective comparisons](../evidence/Bezier-Routing/docs/design.md)

## Related projects

- [bezier_sphere](bezier_sphere.md)
- [Bezier-Layers](Bezier-Layers.md)
- [Bezier-Granite](Bezier-Granite.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
