# Bezier-MemorySphere

`Bezier-MemorySphere` · Memory systems

Train a full-stack spatial recurrent token model from scratch using clustered context keys, fixed scratch slots, Bezier routing, and learned surface output.

## Saved status

Completed from-scratch prototype; bounded memory verified, coding quality low.

## Work completed or attempted

- Implemented a 1,538,295-parameter custom model with 16 addressed regions, 256-dimensional state, 64-dimensional keys, and four overwritten scratch KV slots.
- Completed 44 conditions and 2640 stage-runs across route/depth/halting settings plus independent pretrained capability references; saved models, tokenizer, traces, memory probes, and regrading.

## Findings

**Measured.** Adaptive Bezier and straight routing each passed 6/120 stage-runs across two seeds; direct routing passed 7/120. The independent 14B reference passed 48/60.

[S091](../evidence/Bezier-MemorySphere/README.md) · [S093](../evidence/Bezier-MemorySphere/results/summary.json)

**Measured.** No sphere condition solved long-chain or MBPP transfer tasks or a complete project. Bounded internal scratch memory did not translate into a coding-quality advantage.

[S091](../evidence/Bezier-MemorySphere/README.md)

**Measured.** All three routing controls kept single-row tensor allocation flat through 128 internal steps. This result concerns fixed internal scratch state, not unbounded text-context or training memory.

[S091](../evidence/Bezier-MemorySphere/README.md)

**Reported.** Independent regrading of 1400 unique code/check pairs found zero score discrepancies, and wider-batch replay produced identical outputs.

[S091](../evidence/Bezier-MemorySphere/README.md)

## Limits and open questions

- Small from-scratch model and narrow restricted agent task domain, not a general coding assistant.
- All 16 regions were evaluated densely before mixing, so this study does not measure optimized sparse dispatch.
- Surface contact is an output address, not correctness; pretrained references are independent capability baselines, not matched architectures.

## Recommended next steps

- Recommendation: Use this model as an auditable mechanism baseline; improve training/data and maintain non-geometric controls during scaling.

## Sources

- [Model and outcomes](../evidence/Bezier-MemorySphere/README.md)
- [Condition and memory report](../evidence/Bezier-MemorySphere/results/report.html)
- [Full results](../evidence/Bezier-MemorySphere/results/summary.json)
- [Frozen run plan](../evidence/Bezier-MemorySphere/frozen_plan.json)
- [Mechanism and scope](../evidence/Bezier-MemorySphere/docs/protocol.md)

## Related projects

- [Bezier-Think](Bezier-Think.md)
- [Bezier-Revisit](Bezier-Revisit.md)
- [Bezier-MemorySphere-7B](Bezier-MemorySphere-7B.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
