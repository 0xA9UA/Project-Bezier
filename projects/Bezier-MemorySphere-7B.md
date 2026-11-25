# Bezier-MemorySphere-7B

`Bezier-MemorySphere-7B` · Memory systems

Scale the from-scratch sphere architecture to 7.35B independent parameters and validate training, sparse dispatch, checkpointing, and bounded-depth inference.

## Saved status

Completed 7.35B feasibility pilot; 0/60 at every evaluated depth.

## Work completed or attempted

- Built 64 spatial regions with large independent matrices, a 4096-state recurrent core, selected-region sparse execution, own 2048-token BPE, and factored Adafactor gradient-release updates.
- Completed a 600-update local pilot from random initialization, six task evaluation depth/stopping conditions, depth/memory timing, independent regrading, and save/resume checks.

## Findings

**Measured.** The model has 7,346,916,807 independent parameters; the pilot supervised 14,612 sampled answer tokens and took about 8.6 minutes including validation/saving. This is a feasibility run, not full pretraining.

[S097](../evidence/Bezier-MemorySphere-7B/README.md) · [S098](../evidence/Bezier-MemorySphere-7B/results/release_summary.json)

**Measured.** Every 2/4/8/16/32-visit and adaptive condition passed 0/60 task stages, often emitting finish before a useful edit. Better held-out token prediction did not yield completed programs.

[S097](../evidence/Bezier-MemorySphere-7B/README.md) · [S098](../evidence/Bezier-MemorySphere-7B/results/release_summary.json)

**Measured.** Observed peak training allocation was 20.02 GiB and batch-one inference 13.72 GiB. A 32-token continuation took 0.277 s at two visits and 3.044 s at 32; fixed-workload allocation stayed level through 128 visits.

[S097](../evidence/Bezier-MemorySphere-7B/README.md) · [S098](../evidence/Bezier-MemorySphere-7B/results/release_summary.json)

**Implementation.** All 7.35B weights remain resident; four selected matrices contain 440,664,064 parameters, with a conservative per-visit accessible total of 736,955,847 including shared parameters. Sparse dispatch saves matrix work, not resident weight storage.

[S097](../evidence/Bezier-MemorySphere-7B/README.md)

## Limits and open questions

- One seed/recipe with only 600 updates and no freshly trained 7B straight or ordinary controls.
- Tiny supervised-token budget relative to model size; clustering does not physically compress text KV cache.
- No full FP32 optimizer master copy, BF16 rounding limits, and narrow corpus; context and training memory still grow.

## Recommended next steps

- Recommendation: Use the later 7.41B full-target training run for the next evidence point; retain this pilot as feasibility evidence.

## Sources

- [Architecture, pilot, and limitations](../evidence/Bezier-MemorySphere-7B/README.md)
- [Released measurements](../evidence/Bezier-MemorySphere-7B/results/release_summary.json)
- [Evaluation report](../evidence/Bezier-MemorySphere-7B/results/report.html)
- [Release artifacts](../evidence/Bezier-MemorySphere-7B/release_manifest.json)

## Related projects

- [Bezier-MemorySphere](Bezier-MemorySphere.md)
- [Bezier-MemorySphere-7B-Training](Bezier-MemorySphere-7B-Training.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
