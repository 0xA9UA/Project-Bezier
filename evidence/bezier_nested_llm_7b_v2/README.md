# Joint reasoning and control with a shared 15% core

This is the second full-size adaptation experiment. It starts from the **adapted, non-steering Mistral-7B-Instruct-v0.3 checkpoint** in the preceding experiment. The original project and its checkpoints remain unchanged. Completed results and checkpoint selection are reported in [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).

The recovery pilot completed 768 updates and selected `runs/distill_recovery/best_u000512` using validation. On the fresh direct-answer test, outer accuracy was 53.71%, core accuracy 50.98%, and steered outer accuracy 53.32%. The old outer task remained at 100%. On the separate scaffolded free-running panel (16 programs, 64 examples), outer final-answer accuracy improved from 51.56% to 95.31%; core final answers were 53.13%, with no complete trace correct. Learned steering improved probability loss versus Stay but did not beat Always Turn. **Useful independent core reasoning and activation-dependent control are not established.**

All 93 automated tests passed. All three final GPU evaluations completed, including temporary original-core restoration and exact output restoration afterward. The [recovery plot](runs/distill_recovery/distillation_progress.png) shows training and final direct-answer measurements. Full provenance is in `runs/final_evaluation_suite.json` and `runs/final_selection.json`.

The objective is to make one small, centrally located subset of the outer model's weights useful for three roles: independent reasoning, predicting whether a nearby continuation will help, and ordinary outer-model inference. These are separate empirical requirements; an improvement in one does not establish the others.

## Fixed architecture

| Quantity | Value |
|---|---:|
| Unique model parameters | 7,248,023,552 |
| Unique core parameters | 1,087,201,280 |
| Core share | 14.9999689184% |
| Added learned parameters | 0 |
| Core position | Logical center `(0, 0, 0)`, radius `0.25` |
| Core execution | Four passes through the same four shared blocks |
| Executed core blocks per prediction | 16 |

The core owns shared input/output embeddings, final normalization, the first four attention modules and block norms, the first three complete MLPs, and the first 9,967 neurons of the fourth MLP. The outer path uses those same tensors. There is no copied controller network or separate trainable routing head. Geometry and score readout use fixed rules and existing vocabulary weights. Parameter counting includes the interfaces, rather than counting only the core's transformer blocks.

The sphere is a fixed logical placement of parameter regions, not a physical GPU memory arrangement. Routing currently substitutes **one MLP operation** while preserving the surrounding attention and layer order. Candidate regions include adjacent shell MLPs and a complete core MLP. This is a bounded implementation of the hypothesis, not unrestricted traversal through arbitrary transformer blocks. A later regular visit to a substituted region is still allowed.

Reusing weights keeps the stored parameter budget fixed. It increases executed computation, activations, and training time. The controller uses the live outer activation and Bézier geometry; it is not an independent full conversation model running without that activation.

## Training changes

1. **Recurrent reasoning:** reuse the four-block core four times, passing the raw residual between passes. Normalize only for the readout. This creates depth without new weights.
2. **Verified intermediate supervision:** train on bounded numeric programs with independently checked intermediate states and final answers. Every group includes two inputs with different answers and both A/B option orders. Always choosing one option or one numeric value scores 50%.
3. **Selective teacher guidance:** cache the initial outer model's logits and raw residuals after layers 4, 12, 20, and 32 on training examples only. Gold interpreter targets always supervise the student; late teacher hints and teacher logits on the new task are used only where the teacher's unrestricted top token is correct. The first-pass hint softly preserves the early representation.
4. **Joint optimization with retention:** combine outer prediction, core prediction, core distillation, intermediate hints, and old-task preservation in a single backward pass. Shared complete core transformer tensors receive a larger learning rate. All three partially shared fourth-MLP matrices use the conservative outer rate to avoid accelerating their shell slices.
5. **Matched control comparisons:** after an initial reasoning phase, evaluate both continuations from the exact same activation with the same continuation budget on selected training examples. Train the shared core to predict the change in correct-token log probability. Only the chosen continuation runs during deployed-policy evaluation.
6. **Leakage prevention:** route learning uses final-only prompts. Teacher-forced traces cannot supply future gold states to the controller. Free-running trace evaluation feeds back predicted states rather than true states.

The initial schedule is 1,024 updates of four examples, with 75% new reasoning and 25% old-task replay. Reasoning moves from short programs to mixed depths; three of every four reasoning views supervise intermediate traces. The control loss ramps in after update 256, contingent on genuine positive training-set headroom. Core/outer maximum learning rates are `4e-6`/`1e-6`. Full-weight BF16 training uses in-backward Adafactor and stochastic rounding. No adapters or extra learned parameters are introduced.

The original hypothesis motivates recurrence and intermediate distillation. Related methods provide precedents, not evidence that this particular architecture works: [recurrent-depth language models](https://arxiv.org/abs/2502.05171) and [TinyBERT intermediate distillation](https://arxiv.org/abs/1909.10351).

## Dataset and evaluation

The new corpus has 1,024 training programs, 64 validation programs, 128 test programs, and 128 longer-program out-of-distribution programs, each with four counterbalanced examples. Training includes 1–4 operations; the out-of-distribution split uses 5–6. The DSL and an independent AST interpreter agree on all 100 input pairs for each program. Semantic functions are separated across splits, including held-out final functions versus training intermediate functions. Earlier patch examples are retained as a separate regression task.

Validation checks every 128 updates preserve old outer accuracy within two percentage points and old outer negative log likelihood within 0.10 nats. A violation stops training and retains the last eligible checkpoint. Checkpoint selection combines direct core raw accuracy on reasoning/replay and a fixed validation panel of free-running trace final answers; direct negative log likelihood breaks ties. This composite is a selection rule, not one benchmark's accuracy.

After selection, validation alone selects the decision region, candidate, and turn threshold. Fresh test and out-of-distribution results report:

- Unrestricted next-token accuracy and answer validity, alongside forced A/B accuracy and full-vocabulary loss.
- Core alone and outer with steering disabled; steered outer versus that **same checkpoint** disabled.
- Fixed Stay, fixed Turn, matched oracle, and shuffled-score diagnostics to distinguish conditional control from a useful constant choice.
- Program-level paired confidence intervals, treating variants as dependent.
- Scaffolded free-running state and final-answer accuracy. Fixed format tokens are supplied; predicted digit values are fed back. This is not unrestricted text generation.
- Core transformer zero ablation, preserving shared interfaces and the outer-only part of the fourth MLP. A separate restoration diagnostic replaces the learned core transformer weights with their pre-experiment values to distinguish structural dependence from contribution of newly learned weights.

The frozen protocol in `runs/protocol.json` permits extending to 2,048 updates only on validation progress and retained outer performance, before test performance is opened. Test results never select the checkpoint or controller threshold.

## Validation-motivated recovery

The first run stopped at update 384: old outer validation accuracy fell from 100% to 93.75%, with full-vocabulary loss rising from 0.0021 to 0.3052 nats. The rejected weights were not saved as a selected checkpoint. Core direct reasoning remained near chance. Its eligible checkpoint at update 128 was retained.

A frozen-weight validation screen then tested three equal-budget core memberships, each at one, two, and four passes, plus two input/middle/output execution schedules. It preserved all outer parameters and logits exactly. Simply selecting later weights did not recover direct reasoning, so membership remains unchanged.

Before any new test performance was inspected, `runs/recovery_protocol.json` froze a second pilot starting from the eligible update-128 checkpoint with a fresh optimizer. It caps training at 768 additional updates, reduces both learning rates fourfold, doubles core gold supervision, increases replay preservation KL fourfold, and reduces/ramp-delays the control loss. During training, the turn threshold is 1.0, above the controller's 0.1-nat bound. Thus the route estimator learns from real matched alternatives while the differentiable outer prediction follows the normal path. A deployment threshold is still selected on validation afterward. All shared weights remain jointly trainable.

This is an adaptive recovery experiment informed by validation failure. The bundled changes do not isolate the cause of any improvement. Final results must distinguish preserved outer behavior, actual core reasoning, and useful conditional steering.

## Reproduce locally

Use `WORKSPACE\Bezier-Think\.venv\Scripts\python.exe` from this directory. The tokenizer, model, and teacher cache are local. Checkpoint loads verify recorded file hashes; resume also checks package, configuration, wrapper, and cache identities.

```powershell
python -m pytest tests -q
python scripts/run_distillation.py --output runs/distill_main
python scripts/run_distillation.py --config configs/recovery_v2.json --initialize-from runs/distill_main/best_u000128 --output runs/distill_recovery
python scripts/status_v2.py
python scripts/evaluate_distillation.py --model <selected-checkpoint> --output runs/eval_selected --oracle-diagnostics --compare-core-passes
python scripts/evaluate_distillation.py --model ..\bezier_nested_llm_7b\runs\pilot_baseline\checkpoint_u000512 --initial --output runs/eval_initial --compare-core-passes
python scripts/evaluate_core_restore.py --model <selected-checkpoint> --output runs/eval_core_restore
python scripts/plot_v2.py --run-dir runs/distill_recovery --eval Initial=runs/eval_initial --eval Selected=runs/eval_selected
```

Run GPU commands sequentially. An existing training output requires explicit resume; an existing completed evaluation output is not silently overwritten. The 7B preflight used approximately 23.9 GiB on an RTX 5090. This is a single-seed, narrow-task experiment; it cannot establish broad language-model ranking or general logical competence.
