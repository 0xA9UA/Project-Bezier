<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../../../PUBLICATION_SCOPE.md) · [Reader guide](../../../../READ_THIS_FIRST.md).

# Input-preserving follow-up versus revision 3

This adaptive, exploratory follow-up was motivated by a training-only diagnostic of input-bit loss through the outer GRU. It adds a parameter-free syntax/input skip, with the same training recipe, corpus, parameter budget, and central shared core. It uses one seed and is not a confirmatory experiment.

Positive gains below favor v4. NLL intervals resample paired complete heldout programs; they do not quantify training-seed uncertainty. Checkpoints and margins were selected on validation. This script only compares saved results and performs no test tuning.

| Arm / policy | v3 NLL | v4 NLL | NLL gain [95% CI] | v3 accuracy | v4 accuracy | v3 / v4 mean turns |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| clean-outer / full | 0.694765 | 0.657949 | +0.036815 [+0.024529, +0.048552] | 50.00% | 57.03% | 0.000 / 0.000 |
| clean-outer / zero | 0.694765 | 0.657949 | +0.036815 [+0.024529, +0.048552] | 50.00% | 57.03% | 0.000 / 0.000 |
| nested-core-concurrent / full | 0.670497 | 0.646470 | +0.024027 [+0.016075, +0.032539] | 53.52% | 57.62% | 0.000 / 6.055 |
| nested-core-concurrent / zero | 0.670497 | 0.657042 | +0.013455 [+0.004157, +0.023021] | 53.52% | 52.73% | 0.000 / 0.000 |

| Direct core | v3 accuracy | v4 accuracy | Accuracy gain [95% CI], percentage points |
| --- | ---: | ---: | --- |
| clean-outer | 59.38% | 62.50% | +3.12 [-1.17, +7.42] |
| nested-core-concurrent | 58.79% | 59.57% | +0.78 [-1.95, +3.71] |

Validation-selected checkpoints and margins:

- clean-outer: v3 step 757, threshold 0.2; v4 step 2256, threshold 0.2.
- nested-core-concurrent: v3 step 2256, threshold 0.2; v4 step 1756, threshold 0.005.

Limits:

- One shared training seed; confidence intervals resample programs, not training seeds.
- Reused benchmark holdouts, not a new sequestered confirmation set.
- This adaptive follow-up is exploratory and cannot establish a confirmatory success claim.
- Both revisions select checkpoints and margins on validation; selected steps and margins can differ.
- Multiple descriptive intervals are reported without a multiplicity-adjusted study-level success gate.

Evidence:

- [v4 suite](WORKSPACE/bezier_nested_core_v4/runs/input_preserving/suite.json)
- [v4 report](WORKSPACE/bezier_nested_core_v4/runs/input_preserving/report.md)
- [v3 suite](WORKSPACE/bezier_nested_core_v3/runs/scaleup/tier2_seed7/suite.json)
- [v3 report](WORKSPACE/bezier_nested_core_v3/runs/scaleup/tier2_seed7/report.md)

The accompanying comparison.json contains accuracy intervals, direct-core NLL intervals, matched-program counts, exact hashes, checkpoint provenance, and configuration checks.
