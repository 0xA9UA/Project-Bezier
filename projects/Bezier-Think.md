# Bezier-Think

`Bezier-Think` · Routing and reasoning

Test whether extra shared latent model passes help a small coding agent approach a larger pretrained model.

## Saved status

Completed 34-condition coding pilot; extra steps did not close the quality gap.

## Work completed or attempted

- Compared Bezier, straight, and ordinary recurrence at 2/4/8/16/32 internal steps across two seeds, plus trained zero-step and unmodified 1.5B/14B controls.
- Completed 34 conditions, each on 12 MBPP functions and four three-stage authored projects, with adapter fitting, tool traces, latency probes, and saved audits.

## Findings

**Measured.** Recurrent 1.5B variants passed 9-10 of 24 stages, compared with 9 for trained zero-step controls and 20 for the 14B model. Small models completed 0/4 projects; 14B completed 2/4.

[S072](../evidence/Bezier-Think/results/report.html) · [S073](../evidence/Bezier-Think/results/summary.json)

**Measured.** No consistent quality rise occurred from 2 to 32 steps, and every paired 95% interval for Bezier versus ordinary recurrence included zero. The unmodified small control passed only 1/24, so protocol training is a major separate factor.

[S072](../evidence/Bezier-Think/results/report.html) · [S073](../evidence/Bezier-Think/results/summary.json)

**Measured.** At 32 steps Bezier averaged about 1.35 seconds for a common 32-token probe versus 1.15 seconds for the 14B control; batch-one peak CUDA allocations were about 2.92 and 27.65 GiB. Straight and ordinary recurrence had almost the same small footprint.

[S072](../evidence/Bezier-Think/results/report.html)

**Implementation.** The added recurrence is an adapter around the full pretrained transformer; each internal step adds a complete shared causal-LM pass and KV-cache position. It does not replace or compress the base stack.

[S071](../evidence/Bezier-Think/README.md)

## Limits and open questions

- Small restricted-Python benchmark, two fitting seeds, 16 task families; public-task pretraining overlap unknown.
- Training teaches JSON tool protocol as well as code; executor restrictions can reject otherwise ordinary Python.
- Lower memory primarily reflects 1.5B versus 14B weights; added latent work still costs time and cache entries.

## Recommended next steps

- Recommendation: Improve protocol/code training and use matched compute/parameter controls before increasing latent depth.

## Sources

- [Architecture and frozen evaluation](../evidence/Bezier-Think/README.md)
- [Completed results](../evidence/Bezier-Think/results/report.html)
- [Condition scores and comparisons](../evidence/Bezier-Think/results/summary.json)
- [Protocol](../evidence/Bezier-Think/docs/protocol.md)

## Related projects

- [Bezier-Steer](Bezier-Steer.md)
- [Bezier-Revisit](Bezier-Revisit.md)
- [Bezier-MemorySphere](Bezier-MemorySphere.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
