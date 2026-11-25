# Paired action-value fitting in shared core weights

`bezier_nested_llm_7b_v8` · Nested language models

Fit input-sensitive route utility into existing shared core slices using counterbalanced program pairs and out-of-fold second-stage targets.

## Saved status

Completed frozen pilot; released default remains fixed (2,2), learned conditional advantage unresolved.

## Work completed or attempted

- Fitted six 1,024-element row slices (6,144 scalars) of an existing central MLP down projection with no independent controller weight store.
- Collected 27,648 route outcomes from 768 fit programs, refreshed 6,912 after first writeback, and calibrated on 192 separate training programs.
- Evaluated the frozen learned candidate against fixed, shuffled, prior-v7 and dense controls on fresh verified programs; retained fixed routing as the released default.
- Validated cache/dispatch optimizations and selected-path output parity.

## Findings

**Measured.** The full 7B model and fixed shared-core budget are retained, but the fit changes only 6,144 scalars in one existing tensor. All 6,160,822,272 outside-core scalars stayed bitwise unchanged; GPU storage addresses remained stable.

[S244](../evidence/bezier_nested_llm_7b_v8/runs/pilot/weight_audit.json) · [S236](../evidence/bezier_nested_llm_7b_v8/REPORT.md)

**Measured.** Validation learned and fixed (2,2) each answered 164/256 correctly. Because the release rule required learned accuracy to exceed fixed and shuffled controls, the application defaults to fixed (2,2).

[S237](../evidence/bezier_nested_llm_7b_v8/runs/pilot/selection.json) · [S245](../evidence/bezier_nested_llm_7b_v8/runs/pilot/validation.json)

**Measured.** Learned test accuracy was 490/768 (63.80%) versus 485/768 (63.15%) fixed and 481/768 (62.63%) shuffled. Learned-minus-fixed was +0.65 percentage points [-0.65,+1.95]; learned-minus-shuffled +1.17 [-0.65,+2.99]. Neither met the registered criterion.

[S238](../evidence/bezier_nested_llm_7b_v8/runs/pilot/test_evaluation.json) · [S236](../evidence/bezier_nested_llm_7b_v8/REPORT.md)

**Measured.** On longer programs, learned scored 237/384 (61.72%), fixed 224/384 (58.33%), and shuffled 238/384 (61.98%). Learned-minus-fixed +3.39 points [-0.52,+7.29] still included zero; dense fitted-weight inference scored 61.98%.

[S239](../evidence/bezier_nested_llm_7b_v8/runs/pilot/ood_evaluation.json) · [S236](../evidence/bezier_nested_llm_7b_v8/REPORT.md)

**Measured.** The controller used all nine action pairs on test, rescuing 17 and spoiling 12 versus fixed; OOD rescues/spoils were 35/22. Diverse decisions and net rescues did not establish input-specific benefit with current uncertainty.

[S242](../evidence/bezier_nested_llm_7b_v8/runs/checks/paired_audit.json) · [S236](../evidence/bezier_nested_llm_7b_v8/REPORT.md)

**Measured.** End-to-end learned/fixed/dense median latency was 151.17/95.50/69.79 ms on four 140-token prompts; learned and dense peak allocation was 13.714/13.625 GiB. Replaying selected actions reproduced logits with zero unchosen continuations.

[S243](../evidence/bezier_nested_llm_7b_v8/runs/checks/selected_runtime.json)

**Measured.** A matched grouping optimization reduced full-tree median 398.81 to 390.59 ms and learned inference 148.61 to 146.64 ms, with bitwise outputs. A 20-entry cache already held all 14 plans in the tested workload. The report records 12 v8 plus 35 inherited tests.

[S246](../evidence/bezier_nested_llm_7b_v8/runs/checks/dispatch_benchmark.json) · [S247](../evidence/bezier_nested_llm_7b_v8/runs/cache_benchmark.json) · [S236](../evidence/bezier_nested_llm_7b_v8/REPORT.md)

## Limits and open questions

- Bounded arithmetic task, one fitted candidate, short warm full-context timing without KV cache.
- Paired intervals condition on one shuffle assignment; primitive operations and prefixes may overlap across distinct final functions.
- The same shared rows affect dense predictions and routing features, requiring refreshed outcomes and preservation checks.
- The off-template generation smoke answered incorrectly; unrestricted reasoning is not established.

## Recommended next steps

- Recommended: keep the fixed default and report the learned candidate as experimental.
- Recommended: target reliable first-checkpoint action values, where validation oracle headroom is largest, using a new holdout for further selection.

## Sources

- [Usage and released policy](../evidence/bezier_nested_llm_7b_v8/README.md)
- [Completed v8 report](../evidence/bezier_nested_llm_7b_v8/REPORT.md)
- [Frozen released policy](../evidence/bezier_nested_llm_7b_v8/runs/pilot/selection.json)
- [Test metrics](../evidence/bezier_nested_llm_7b_v8/runs/pilot/test_evaluation.json)
- [Longer-program metrics](../evidence/bezier_nested_llm_7b_v8/runs/pilot/ood_evaluation.json)

## Related projects

- [bezier_nested_llm_7b_v7](bezier_nested_llm_7b_v7.md)

Saved report updated November 25, 2026; v8 branches from stage-ordered v7 and should not be confused with v7's later RL continuation.

[Back to the project directory](README.md)
