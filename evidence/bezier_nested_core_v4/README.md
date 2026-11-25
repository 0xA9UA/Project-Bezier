# Revision 4: preserve input information while retaining the 5% shared core

This is a separate, exploratory follow-up to the completed revision-3 scaling study. It preserves its architecture dimensions, training recipe, corpus, and learned parameter counts. The one intervention is a parameter-free syntax/input skip into the outer encoder activation.

The larger shared model still contains **4,704,970 learned parameters**, of which the central core contains **235,210 (4.99918%)**. All core weights and biases remain shared between reasoning, control, and outer visits; its address is fixed at the origin. The core sphere's radius remains 0.25. Region selection still uses the two nearest centers and does not impose a hard radial execution gate.

## Why this follow-up

A training-only diagnostic on 48 programs and all sixteen input assignments found that revision 3's outer predictor barely responded to changed input bits, while the direct core responded strongly. The recurrent encoder attenuated input perturbations severely; downstream traversal further suppressed their effect on prediction. This identifies a representation bottleneck, not proof that the encoder was the only limitation.

Revision 4 adds the following fixed skip to the recurrent encoder output:

```text
initial_activation = recurrent_encoding
                   + syntax_skip_scale * (syntax_features @ syntax_projection @ input_fold)
```

`syntax_skip_scale` is fixed at 1.0, not learned. The features contain input bits, operator identities, variable references, and the queried variable. They do not evaluate any Boolean operation, include intermediate answers, or read the training target. All projections already exist as deterministic buffers. Setting the scale to zero reproduces the previous model exactly.

The skip is active in both the clean outer and shared-controller arms. It changes the input representation; any improvement should not be attributed solely to extra parameters or extra training.

## Matched training and evaluation

Both seed-7 arms use exactly the revision-3 recipe: 500 core-only updates at batch 256, 256 outer warmup updates at batch 64, and 1,500 joint updates at batch 64. Direct-core reasoning, normalized FP32 counterfactual supervision, learning-rate schedules, and validation selection remain unchanged.

The frozen corpus is reused directly from `WORKSPACE\bezier_nested_core_v3\corpus\manifest.json`: 1,024 training structures covering 256 distinct logical functions and 16,384 training examples. Validation, test and longer-program evaluation each contain 512 examples. No training truth table overlaps any heldout truth table. These are reused benchmark holdouts.

The evaluated `best.pt` is selected using joint-phase validation NLL; `final.pt` retains all 2,256 optimizer updates. Routing margins are selected on validation only. Test/OOD labels do not select this intervention's strength or checkpoints. The experiment remains a single-seed, adaptive follow-up, not an independent confirmation of the hypothesis.

## Run

```powershell
& "WORKSPACE\bezier_nested_core_v4\scripts\run_experiment.ps1"
# Resume an interrupted identical run:
& "WORKSPACE\bezier_nested_core_v4\scripts\run_experiment.ps1" -Resume
```

The entry point uses the existing local RTX 5090 environment and runs the clean/shared pair. Results are saved in `runs/input_preserving/report.md`; `comparison.md` contrasts the matched revision-3 and revision-4 checkpoints after both finish. The earlier studies and checkpoints remain unchanged.

## Completed results

Both arms completed all 2,256 updates. All 72 tests passed, and the saved source, data, parameter-sharing, and validation-selection artifacts passed verification. The selected shared checkpoint is update 1,756; the full endpoint is also retained.

On 512 heldout examples, the same shared checkpoint scored **52.73% with redirects disabled** and **57.62% with redirects enabled**, averaging **6.05 redirects per 16 visits**. Answer NLL fell from 0.65704 to 0.64647 nats. The paired-program 95% interval for that NLL improvement was [0.00835, 0.01294] nats. Repeated steering also improved NLL over the one-redirect limit and the separately trained clean control in this seed.

Direct core accuracy was **59.57%**, but only **49.80% on longer programs**. The result supports a useful steering effect in this one-seed experiment; it does not establish robust longer-program reasoning or a replicated architectural benefit.

See [results](runs/input_preserving/report.md), [matched revision comparison](runs/input_preserving/comparison.md), and [artifact verification](runs/input_preserving/verification.json).

## Frozen-model diagnostic follow-up

The [action-value audit](CONTROL_VALUE_AUDIT.md) measured unused one-action headroom. The subsequent [diagnostic predictor experiment](DIAGNOSTIC_VALUE_PREDICTOR.md) trained three independent 16,513-parameter heads on the frozen v4 checkpoint and evaluated fresh, historically excluded Boolean programs. Neither the fresh-test nor longer-program gate established gains over both fixed and shuffled controls; all primary adjusted intervals include zero. The original v4 weights remain bitwise unchanged, and 140 tests pass. See the [consolidated diagnostic report](runs/diagnostic_value_predictor_20261125/report.html) for measurements, uncertainty and reproducibility artifacts.
