# Bezier-Revisit

`Bezier-Revisit` · Routing and reasoning

Isolate routing and leave-and-return visits in a shared recurrent coding model under matched stored parameters and matrix-computation budgets.

## Saved status

Completed matched recurrence study and audit; no clear Bezier coding benefit.

## Work completed or attempted

- Fitted Bezier, non-geometric routing, and ordinary recurrence on a frozen Qwen2.5-Coder-1.5B base; tested 4/8/16 passes and no-return interventions.
- Audited 864 stage trials and 3335 actions, six adapters, source/data hashes, arithmetic budgets, revisit traces, and standalone/batched generation behavior.

## Findings

**Measured.** At the prespecified eight-pass budget, Bezier passed 28/72 stages, ordinary recurrence 30/72, and non-geometric routing 28/72; all completed 0/8 projects. Each primary arm had 1,545,352,240 total and 1,637,936 trainable parameters.

[S077](../evidence/Bezier-Revisit/RESULTS.md) · [S078](../evidence/Bezier-Revisit/results/summary.json)

**Measured.** Allowing Bezier returns improved stage success by 1.39 percentage points versus a post-training blocked-return intervention, with paired interval [0,+4.69]. This is not a retrained ablation and does not isolate geometry from changed route distributions.

[S077](../evidence/Bezier-Revisit/RESULTS.md)

**Measured.** All 277 eight-pass Bezier actions had leave-and-return events; blocking eliminated those events. Frequent revisits are measurable but did not establish more useful reasoning.

[S077](../evidence/Bezier-Revisit/RESULTS.md)

**Measured.** One standalone demo failed although its original batch-of-four evaluation passed. Diagnostic replay reproduced the batch's first actions; serial generation repeated a different first action. The saved BF16 benchmark is batch-sensitive.

[S077](../evidence/Bezier-Revisit/RESULTS.md) · [S080](../evidence/Bezier-Revisit/results/batch_sensitivity.json)

## Limits and open questions

- Two seeds, 24 fresh MBPP functions and four reused projects; no general coding-agent or compression claim.
- Only 190 detailed update records remain although final metadata reports 192 updates; final two individual loss/gradient records were not saved.
- Dense field execution matches arithmetic but is not a sparse-dispatch speedup; timing probes were noisy.

## Recommended next steps

- Recommendation: Investigate batch-sensitive generation and preserve exact evaluation configuration; expand task families and independent seeds.

## Sources

- [Primary results and final-audit limitations](../evidence/Bezier-Revisit/RESULTS.md)
- [Machine-readable results](../evidence/Bezier-Revisit/results/summary.json)
- [Independent trial and adapter audit](../evidence/Bezier-Revisit/results/audit.json)
- [Batch diagnostic](../evidence/Bezier-Revisit/results/batch_sensitivity.json)
- [Frozen protocol](../evidence/Bezier-Revisit/docs/protocol.md)

## Related projects

- [Bezier-Think](Bezier-Think.md)
- [Bezier-Routing](Bezier-Routing.md)
- [Bezier-MemorySphere](Bezier-MemorySphere.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
