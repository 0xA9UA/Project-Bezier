<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Learning when to redirect a 7B model

This pilot measures whether a smaller execution path through the same model can choose a better next-token prediction for each prompt. It adapts a 7,248,023,552-parameter Mistral decoder. The sphere of radius 0.25 at the origin contains 1,087,201,280 of its existing parameters, or 14.99997%. The outer model can use the weights in that sphere in its ordinary transformer computation.

**The learned controller scored 63.80% versus 63.15% for fixed routing on 768 fresh test prompts.** On 384 longer prompts it scored 61.72% versus 58.33% fixed and 61.98% with its actions shuffled. Both 95% intervals for the improvement over fixed include zero. Validation selected the fixed pair (2, 2) as the default. On four 140-token prompts, learned inference occupied 13.714 GiB and dense inference occupied 13.625 GiB. All policies used one GPU weight store. [Pilot report](REPORT.md) · [Frozen selection](runs/pilot/selection.json) · [Selected shared rows](runs/pilot/selected_rows.pt)

## How a redirect works

After decoder layers 10 and 21 (zero-based), the controller reads the current token activations and the position, tangent and bend of three candidate Bézier paths. It runs the candidates' geometry queries together through the shared core. Six slices of the first central MLP's down-projection weight serve a second purpose here. Three score the available actions at the first checkpoint, and three score the actions at the second. The engine continues on the incumbent path unless a candidate's estimated gain exceeds the calibrated margin. Each prompt executes one outer continuation at each checkpoint. [Controller implementation](model.py)

The node atlas assigns a fixed position to each parameter-owning node. At each decoder stage, the curve determines which 98% of hidden nodes execute. Input and output interfaces retain all required token rows. Path choices preserve causal attention and transformer layer order. On a training program the existing 20-entry coordinate-index cache held all 14 distinct plans needed to label the nine continuations. [Spatial engine](../bezier_nested_llm_7b_v7/spatial_core/engine.py) · [Cache benchmark](runs/cache_benchmark.json)

## What the task tests

The prompt gives a short arithmetic program, two input digits, and two possible results. The model must predict the correct answer letter with its full vocabulary available. Each program has two different input assignments with different results. Each input appears with both orders of the same two answer values. Scoring all four prompts keeps a constant answer-letter preference at 50%.

Each training example supplies outcomes for all nine combinations of first and second actions. The fitter uses the change in greedy correctness and correct-answer probability as its target. It learns both the absolute advantages and how those advantages change when the input or answer order changes. The first-stage fit uses second-stage decisions produced by a fit that excluded that program. At inference, each decision sees only its own prompt and current trajectory. [Action-value training](scripts/action_values.py)

Fresh evaluation uses 64 three- or four-operation validation programs, 192 new test programs of the same depths, and 96 longer programs with five or six operations. The full program is the unit of evaluation and resampling. All three sets exclude final truth functions present among 5,350 previously recorded function signatures. [Corpus verification](corpus_eval/verification.json)

## Use the selected policy

From the project root, run a prompt on the resident GPU model. The default mode uses the policy frozen on validation. Use `--mode learned` to inspect the trained controller's own decisions, including when a fixed route won validation. The output records the chosen action pair for each generated token. [Prediction application](applications/predict.py)

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v8/applications/predict.py --prompt 'Start with 7. Add 2, multiply by 3, then take the remainder modulo 10. A: 7  B: 9. Reply with A or B.' --max-new-tokens 1
```

## Reproduce the checks

Run the CPU tests from the project root in the existing PyTorch environment.

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' -m pytest bezier_nested_llm_7b_v8/tests -q
```

The GPU preflight exercises fitting, calibration, validation and evaluation on disjoint parts of the training corpus. The output directory must be new.

```powershell
& 'WORKSPACE/Bezier-Think/.venv/Scripts/python.exe' bezier_nested_llm_7b_v8/scripts/pilot.py --small --output bezier_nested_llm_7b_v8/runs/local_preflight
```
