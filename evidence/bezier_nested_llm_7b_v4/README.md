# One resident model with a shared controller core

The completed pilot implements the fixed 15% shared core and proves exact weight sharing, but **does not establish useful conditional improvement of the outer model**. The objective is to steer the full model; standalone core task accuracy is not used as the success criterion.

All **7,248,023,552** Mistral parameters remain on the GPU. The separate `CoreExecutionView` references the same storage for a central **1,087,201,280-parameter core (14.99997%)**, with no added learned parameters. Its five core blocks and shared interfaces remain available to ordinary outer inference. Geometry and membership are fixed.

## Completed results

| Full-vocabulary accuracy | Fresh test | Longer-program OOD |
|---|---:|---:|
| Initial model, control disabled | 65.92% | 58.11% |
| Selected model, control disabled | 65.82% | 58.01% |
| Selected model, learned control | 66.70% | 57.71% |
| Validation-best fixed action: Stay | 65.82% | 58.01% |
| Label/slot-matched shuffled scores | 67.09% | 57.81% |
| Always use MLP 29, descriptive final arm | 67.58% | 57.13% |

Test control gain is **+0.879 percentage points**, with paired program-bootstrap 95% CI **[-0.879, +2.637]**. OOD gain is **-0.293 points [-1.270, +0.586]**. The controller uses MLP 29 on 47.75% of test examples and 14.26% of OOD examples; it never chooses MLP 27. Active routing does not demonstrate useful control: shuffled scores outperform the learned policy, the intervals cross zero, and answer-swap diagnostics show a strong A-to-B effect without improved all-variants correctness. Fixed MLP 29 was not validation-selected and does not retrospectively replace the fixed control.

The final resident check measured **zero extra CUDA bytes** when creating another core view, **13.6964 GiB peak allocated** and **14.2070 GiB reserved**. Median forward time was **90.377 ms controlled versus 70.724 ms disabled** (batch 4, 141 tokens, three repetitions). Serial and separate-stream outputs/actions match bitwise; no simultaneous GPU kernel overlap was observed. Sharing avoids a hypothetical **2.02507 GiB duplicate core weight copy**. It does not reduce the resident full model's 13.50050 GiB weight requirement; controller activations still cost memory.

The checkpoint audit verified **all 6,160,822,272 outside-core elements bitwise unchanged** and updates in all 48 core-bearing tensors. All 49 main CPU tests, the real-corpus diagnostic compatibility check, and five final suite tasks passed. The generation smoke ran but answered `B: 29` to `(12 + 5) * 2` instead of `A: 34`; it is an execution check, not a passed arithmetic test.

## Training and interpretation

The controller reads the full detached outer activation sequence with two shared query/geometry tokens. A TRAIN-only probe selected decision layer 26: normal MLP 28 versus MLP 27 or 29. Training compares equal-budget continuations from the same activation, rewarding rescues and penalizing spoils. Ordinary outer loss, frozen-initial logit preservation, and replay train the same exact core slices. Initial teacher logits are cached on CPU; no second teacher model resides on GPU.

The run completed 1,024 reasoning and 341 replay updates, exposing 4,096 reasoning and 682 replay examples. The selected checkpoint is earlier: **768 reasoning and 256 replay updates**, with **3,072 reasoning and 512 replay exposures**. Do not attribute completed-run coverage to this checkpoint.

The original validation selector chose update 1,024 with an **Always Stay** policy. A **non-preplanned, validation-only amendment**, made before fresh test/OOD inference, selected the sole saved active policy that beat disabled and validation-best fixed control on validation: update 768, threshold -0.005. Both original and amended selections are preserved. Final results do not confirm that validation gain. Update 1,024 was not evaluated on the final holdout; the initial comparison is the v3 starting checkpoint.

The new corpus contains 128 validation, 256 test, and 256 OOD programs, each with four counterbalanced variants and independent interpreter checks across all 100 inputs. Prior functions and short functions are excluded. These final holdouts are now inspected; further model/data selection needs fresh holdouts.

Inference uses one selected MLP substitution, original attention order, no CPU weight offloading, no quantization, and no KV cache. Learned placement and repeated trajectory correction are not implemented. Weighted tie-loss normalization and a candidate-minus-Stay shared-head readout are future hypotheses, not changes in this revision.

## Artifacts and reproduction

The [full report](WORKSPACE/bezier_nested_llm_7b_v4/EXPERIMENT_REPORT.md) contains exact counts, paired intervals, limitations, and reproduction commands. The [results summary](WORKSPACE/bezier_nested_llm_7b_v4/runs/summary/results_summary.json), [figure](WORKSPACE/bezier_nested_llm_7b_v4/runs/summary/summary.png), [frozen amended selection](WORKSPACE/bezier_nested_llm_7b_v4/runs/control_selection.json), [final suite](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/suite.json), and [secondary diagnostics](WORKSPACE/bezier_nested_llm_7b_v4/runs/final/control_diagnostics.json) retain provenance and the original selection. Package and evaluated policy remain frozen.

Use `WORKSPACE/Bezier-Think/.venv/Scripts/python.exe` from `WORKSPACE/bezier_nested_llm_7b_v4`. Training and repeated frozen evaluation require new output directories. Repeated evaluation must reuse the existing selection file and is not a new blind test.
