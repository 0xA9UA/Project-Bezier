# Nested core v1 - shared checker prototype

`bezier_nested_core` · Nested cores

Test whether a tiny direct-path view of the outer model's own core weights can predict useful mid-route changes on Python next-byte prediction.

## Saved status

Completed integration pilot; no demonstrated quality benefit; retained as the original shared-slice prototype.

## Work completed or attempted

- Built a causal byte encoder, cubic sphere routes, nearest-two expert visits, fixed-exit tail redirection, and a 64-to-16-to-1 checker drawn from outer-expert tensor views.
- Implemented immediate serial, delayed single-stream reference, and concurrent CUDA-stream checking, including fixed-shape CUDA Graph inference.
- Completed a six-arm, seed-7 integration experiment with 24 updates per arm and a 64-prefix evaluation cap per split.
- Preserved reproducible corpus and source identities, resumable checkpoints, shared-weight audits, paired file-bootstrap evaluation, trajectory plots, and profiler/resource measurements.
- Preserved a separate untrained overlap probe and an earlier smoke run that failed its initial audit.

## Findings

**Measured.** Shared arms contain 191,372 learned parameters and add zero checker parameters; the private checker adds 1,057. Sharing is a storage result, not zero computational cost.

[S110](../evidence/bezier_nested_core/runs/smoke/report.md)

**Measured.** At 16 visits, concurrent full-turn test NLL was 4.101342 nats per byte versus clean-outer 4.084265. Full versus zero-turn gain was 0.012108 with 95% paired interval [-0.010313, 0.034523]; full versus clean gain was -0.017078 [-0.046866, 0.015950]. Useful accuracy improvement was not established.

[S110](../evidence/bezier_nested_core/runs/smoke/report.md) · [S111](../evidence/bezier_nested_core/runs/smoke/suite.json)

**Measured.** The concurrent model averaged 2.844 redirects, but full versus one-turn NLL gain was -0.001781 [-0.019655, 0.018584]. Redirect activity did not establish a repeated-steering benefit.

[S110](../evidence/bezier_nested_core/runs/smoke/report.md)

**Measured.** Profiler evidence observed eager and captured-graph kernel overlap; the no-checker graph did not overlap. Batch-one captured concurrent inference was 1.285 ms versus 1.509 ms for the same-policy pipeline reference, but clean graph inference was 0.660 ms. Eager concurrent inference was slower than the reference (21.172 versus 19.227 ms).

[S110](../evidence/bezier_nested_core/runs/smoke/report.md)

**Measured.** The suite records complete execution, pilot_only scientific status, a failed sample criterion, and an incomplete full protocol.

[S111](../evidence/bezier_nested_core/runs/smoke/suite.json)

## Limits and open questions

- One seed, 24 updates, and only 64 prefixes per split; the five-seed full protocol was not executed.
- Python next-byte scores are not directly comparable with the Boolean answer tasks introduced in v2.
- Training labels use detached snapshot activations; v2 later introduced live control gradients into the outer representation.
- A region is a computational address, and the inner-sphere radius is not a hard gate on core-expert execution.
- Saved benchmark results are hardware- and execution-mode-specific; this index did not rerun them.

## Recommended next steps

- Recommendation: use the later whole-core revisions for new work; retain v1 as the Python next-byte and concurrency reference.
- Recommendation: if revisiting its hypothesis, run the preregistered uncapped multi-seed matrix before making quality claims.

## Sources

- [Architecture, protocol, and reproduction](../evidence/bezier_nested_core/README.md)
- [Completed six-arm pilot report](../evidence/bezier_nested_core/runs/smoke/report.md)
- [Machine-readable completion and scientific gates](../evidence/bezier_nested_core/runs/smoke/suite.json)
- [Separate untrained overlap probe](../evidence/bezier_nested_core/runs/profile_probe/overlap_summary.json)

## Related projects

- [bezier_nested_core_v2](bezier_nested_core_v2.md)
- [bezier_nested_core_v3](bezier_nested_core_v3.md)
- [bezier_nested_core_v4](bezier_nested_core_v4.md)

Indexed saved artifacts on November 25, 2026. README identifies v2 as the revision requested November 24; no live training status was inferred.

[Back to the project directory](README.md)
