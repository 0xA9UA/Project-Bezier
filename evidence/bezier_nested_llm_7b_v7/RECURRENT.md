# Recurrent spatial computation

The first, supervised 64-update pilot passed its execution checks but failed answer accuracy: all recurrent policies scored 0% on fresh test and longer-program sets, and every learned run reached the maximum budget. See the [measured results](runs/recurrent_pilot/REPORT.md). The current training method is [reinforcement learning with reward-gated position changes](REINFORCEMENT.md).

The recurrent implementation starts from the selected v7 checkpoint. Its entry point is `spatial_core.RecurrentController`; the original stage-ordered implementation remains available to reproduce the earlier experiment. The new execution path does not call the old interval scheduler or its routing checkpoints.

Each computational turn selects 2,048 gated MLP neuron nodes globally across all 32 source layers. The cubic trajectory also chooses an attention region and one quarter of its Q/K/V channels. The selected gate/up rows and paired down columns operate together on the current residual activation. Causal attention keeps semantic token positions separate from spatial position. Shared normalization, bounded residual increments and a bounded activation RMS provide a common interface for arbitrary order and repeated use. These changes preserve tensor compatibility; they do not guarantee that pretrained representations retain their quality.

The immutable recurrent state carries the full activation, padding mask, position, direction, curvature, turn count, executed curves, actions and exact selected node identifiers. The shared core reads this state at every turn and scores continue, redirect, revisit and stop. Revisit reverses the preceding executed curve and selects exactly the same nodes. The updated activation is retained. Every cubic's control points lie inside the unit sphere, so the whole curve does too. There is no required destination on the sphere boundary.

The core remains centered at `(0,0,0)`, radius `0.25`, with 1,087,201,280 of the 7,248,023,552 parameters (14.99997%). It uses views of the outer model's existing tensors. There are no added learned weights, persistent gathered weight copies, CPU execution or weight offloading. All model weights are resident in BF16 on the GPU. Geometry and index caches contain metadata only.

Training compares three continuations from the same captured activation using identical turn and node budgets. Immediate stopping is a separate zero-extra-turn alternative, trained with task loss and an explicit computation price. Policy supervision, recurrent task cross-entropy and dense task cross-entropy enter one backward pass. The pilot updates the shared core slices and freezes outside-core weights. Gradients are truncated at the captured exploration prefix. Labels and program verification occur only in training and evaluation. Inference scores candidate geometry, executes one chosen continuation and stops on the learned action or the maximum budget.

The pilot uses at most eight turns and at least one turn. Fixed and shuffled routing retain each example's learned stopping budget and pay the same number of controller evaluations, even though their decisions are supplied. They isolate routing quality conditional on that budget. Dense inference is the same-weight native 32-layer pass, with its different computation cost reported explicitly. Parameter reuse in the pilot table measures outer selected-node use, including norms; repeated controller-core execution is accounted for separately. Pilot evaluation timings include execution-history diagnostics. A separate benchmark measures ordinary forward calls without those final diagnostics.

## Run the pilot

Use a fresh output directory. The pilot saves its source snapshot, protocol, fresh corpus, training log, selected core, correctness checks, storage audits and all paired evaluation records.

```powershell
Set-Location 'WORKSPACE'
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v7/scripts/recurrent_pilot.py --output bezier_nested_llm_7b_v7/runs/recurrent_pilot_new --steps 64 --validation-groups 16 --test-groups 32 --ood-groups 16
```

## Run the trained recurrent model

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v7/applications/recurrent_generate.py --run bezier_nested_llm_7b_v7/runs/recurrent_pilot --prompt 'Evaluate (12 + 5) * 2. A: 34 B: 29 Answer:' --max-new-tokens 8 --output bezier_nested_llm_7b_v7/runs/recurrent_generation.json
```

Generation recomputes the context for each output token; this prototype has no KV cache. Each forward call performs its own recurrent spatial computation with independent stopping per prompt. The application verifies the trained checkpoint checksum and loads no answer labels or verifier.

## Evidence

- [Pilot report](runs/recurrent_pilot/REPORT.md)
- [CPU correctness log](runs/recurrent_checks/cpu_tests.log)
- [GPU correctness](runs/recurrent_pilot/correctness.json)
- [Weight storage and preservation](runs/recurrent_pilot/weight_audit.json)
- [Core ownership](runs/recurrent_pilot/atlas_audit.json)
- [Frozen protocol](runs/recurrent_pilot/protocol.json)
- [Clean inference benchmark](runs/recurrent_pilot/runtime.json)
- [Independent artifact and trace audit](runs/recurrent_pilot/independent_audit.json)

To rerun correctness checks:

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' -m pytest bezier_nested_llm_7b_v7/tests -q
```
