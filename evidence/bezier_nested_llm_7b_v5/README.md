# V5: shared-core controller with a Stay reference

**Neither arm demonstrates useful learned steering on the fresh holdout.** Training, both evaluations, diagnostics, preservation audits, and runtime checks are complete. Absolute remains the preregistered primary; relative's selected fallback always stays. Exact shared storage and update scope passed independently of the accuracy result.

The goal remains better full-model predictions through a fixed central shared core, not high standalone core accuracy. One 7,248,023,552-parameter Mistral model stays on the GPU; the core uses 1,087,201,280 existing parameters (14.999968918%) through a separate execution context. It adds no learned parameter or independent core weight store.

Both arms warm-start the same v4 reasoning-update-768 checkpoint with fresh optimizers and seed 83. They share corrected tie-loss normalization, stronger control supervision, FP32 arithmetic for the existing two-row readout, outer prediction loss, preservation targets, and replay. The controlled contrast is:

| Arm | Controller readout | Core queries per example |
|---|---|---:|
| Absolute | Existing candidate A/B margin, computed in FP32 | 2 |
| Relative | Candidate margin minus the same-context Stay margin | 3 |

The planned example order and update budget match; controller compute does not. The relative guard stop also makes **actual completed training budgets unequal**. Both arms retain fixed geometry and one MLP substitution: normal MLP 28 versus MLP 27 or 29, after decision layer 26.

| Training outcome | Absolute | Relative |
|---|---:|---:|
| Completed reasoning / replay updates | 1,024 / 341 | 256 / 85, guard stop |
| Selected reasoning / replay updates | 768 / 256 | 128 / 42 |
| Selected reasoning / replay example exposures | 3,072 / 512 | 512 / 84 |
| Selected validation accuracy | 65.4297% | 64.2578% |
| Validation control gain over disabled / best fixed | +0.9766 points | 0; Always Stay |
| Main training peak allocated memory | 18.9554 GiB | 19.6797 GiB |

The relative arm stopped when disabled validation accuracy fell from 64.0625% to 62.8906%, exceeding the permitted one-percentage-point drop. Its earlier selected checkpoint is a no-proof fallback. The preregistered validation ranking selected **absolute as primary before final testing**, without changing the protocol. Unequal training exposure prevents interpreting the final arm difference as a matched-budget comparison.

| Full-vocabulary accuracy | Fresh test | Longer-program OOD |
|---|---:|---:|
| Initial warm-start, disabled | 65.4297% | 60.2539% |
| Absolute selected weights, disabled | 65.0391% | 60.1563% |
| Absolute learned control | 64.7461% | 59.3750% |
| Matched shuffled scores | 64.3555% | 60.1563% |
| Relative fallback = disabled = shuffled | 65.1367% | 61.0352% |

Primary learned-minus-disabled accuracy is **−0.293 percentage points [95% CI −2.148, +1.563]** on test and **−0.781 [−2.441, +0.881]** on OOD. All key control intervals cross zero. Weight adaptation slightly improves NLL point estimates while accuracy stays near the warm-start level; enabling routing then worsens NLL and accuracy point estimates. These are separate comparisons. Test diagnostics show 60 rescues and 63 spoils, with an A-to-B answer shift and no reliable semantic-reasoning gain.

Relative chooses Stay for every test, OOD, and replay example, giving exactly zero conditional gain. Its higher OOD point estimate is a weight/checkpoint outcome, not successful steering. Relative-minus-absolute learned accuracy intervals also include zero. No arm is promoted after testing, and both arms retain 100% on the old replay examples.

New final-function signatures exclude earlier v2/v3/v4 final and intermediate functions; primitive operations and intermediate prefixes may still be shared. The corpus contains 128 validation, 256 test, and 256 longer-program OOD groups, with four counterbalanced variants each. Both selected policies belong to the same frozen final cohort. Validation gains remain selection observations, not held-out evidence.

Both checkpoint audits verified **all 6,160,822,272 outside-core elements bitwise unchanged**, with updates in all 48 core-bearing tensors. Creating another core execution view used **zero extra CUDA weight bytes**. Final peak allocated inference memory was **13.6959 GiB absolute / 13.7823 GiB relative**. Controlled forward medians were **90.017 / 99.841 ms**, versus disabled **68.971 / 68.368 ms**, on four 140-token validation prompts. Serial/separate-stream outputs matched bitwise; no kernel overlap was observed. The full weights still occupy 13.50050 GiB; sharing avoids a hypothetical 2.02507-GiB duplicate core copy, not the full-model allocation.

**68 CPU tests and all eight execution tasks completed.** The generation smoke produced `B: B` for `(12 + 5) × 2`, whose correct choice was `A: 34`: runtime worked, but the answer was wrong. This does not establish general prompt reasoning.

See the [full report and reproduction commands](WORKSPACE/bezier_nested_llm_7b_v5/EXPERIMENT_REPORT.md), [frozen results](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary/results_summary.json), [primary figure](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary/summary.png), and [corrected arm comparison](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary_final/cohort_comparison.png). The corrected chart explicitly labels unequal training budgets; its [render provenance](WORKSPACE/bezier_nested_llm_7b_v5/runs/summary_final/render_provenance.json) preserves input, source, and image hashes without changing metrics or frozen scripts. The early stop does not conclusively disprove reference subtraction as a broader technique. Further selection requires fresh holdouts.
