<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Controller-first shared core and bounded GPU residency

The core's job is to improve the **full outer model's decisions**. It is not required to match that model as a standalone language model. This revision corrects the v2 training/selection objective and adds a runtime that can keep the full parameter set available without keeping all of it in VRAM.

The completed pilot improved fresh-program accuracy from **53.52% to 63.67%**. The selected policy made no turns, so this is an adaptation gain rather than demonstrated routing value. The final engine benchmark reduced peak CUDA tensor allocation from **13.60 to 3.44 GiB**, with identical tested outputs and about **15.2 times** the controlled-forward latency. See the [complete report](WORKSPACE/bezier_nested_llm_7b_v3/EXPERIMENT_REPORT.md), [checks](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/final_checks.json), and [results figure](WORKSPACE/bezier_nested_llm_7b_v3/runs/final/summary/summary.png).

## What changes

- The exact **1,087,201,280-parameter core** remains at the logical sphere center, inside a **7,248,023,552-parameter** model. No separate learned controller weights are added.
- Full vocabulary embeddings and output heads cease to count as core requirements. The controller reads two existing embedding rows and two existing output rows, freeing its budget for nearly **five shared transformer blocks**, instead of four.
- The initial fixed organization uses outer blocks 17–21, with four complete MLPs and 14,061 neurons of the fifth. The ordinary outer path still uses every full block and every vocabulary row.
- Training supervises the core's **predicted value of a routing action**, using two equal-budget continuations from the same activation. Ordinary outer loss and old-task preservation jointly train the same weights. Standalone core accuracy does not choose checkpoints.
- Checkpoints and thresholds are selected by **steered outer accuracy**, with correct-token negative log likelihood as a tie-break and preservation guards. Fixed-action and shuffled-score controls test whether the core's conditional choices add value.
- The inference engine keeps the shared core and an explicitly counted outer cache resident, transfers other outer modules from CPU when needed, and reuses the same core storage in both execution paths.

The new parameter organization is a fixed hypothesis for this pilot, not a demonstrated optimal arrangement. Central location is a logical parameter address; GPU residency is implemented and measured separately.

## Memory claim being tested

Sharing the core avoids storing a second independent controller model. Lower GPU residency additionally comes from streaming outer weights. The full model still occupies CPU/storage space, and transfers can increase latency. This is compatible with standard weight offloading; an accuracy benefit from control must be demonstrated independently. See the [Hugging Face offloading documentation](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling) for the underlying residency principle.

The benchmark compares identical BF16 weights, prompts, routing decisions, and computation in resident and streamed modes. It reports persistent core bytes, additional resident outer bytes, streamed cache bytes, allocator peaks, transfer volume, and synchronized wall time. The core's parameter fraction is not treated as a prediction of total VRAM. CUDA tensor allocation and allocator reservation are separate measurements, following [PyTorch's memory accounting](https://docs.pytorch.org/docs/main/notes/cuda.html).

This first engine preserves the existing one-MLP route intervention and recomputes context without a KV cache. It is not yet an optimized long-context serving system or arbitrary repeated traversal through transformer blocks. Engine correctness and reduced residency do not themselves establish a smarter model.

## Data and provenance

The initial outer weights are the unchanged v2 recovery checkpoint at update 512. Training and validation reuse the verified v2 reasoning splits and old-task replay. A newly frozen evaluation-only corpus excludes all earlier reasoning final/intermediate functions, prior patch functions, and all one/two-stage functions. It contains 128 fresh 3–4-stage programs and 128 fresh 5–6-stage programs, four counterbalanced examples each. Independent interpreters verify all 100 inputs per program. These new final results are inspected only after controller selection.

All code, cache, and checkpoint dependencies are local. The v1/v2 experiments and checkpoints remain unchanged. Use `WORKSPACE\Bezier-Think\.venv\Scripts\python.exe` for commands from this directory. Measurements and reproduction commands are recorded in [EXPERIMENT_REPORT.md](WORKSPACE/bezier_nested_llm_7b_v3/EXPERIMENT_REPORT.md).

The recovery pilot completed 512 updates. It alternates reasoning/control with replay CE/KL updates, avoiding two simultaneous outer training graphs. It has 1,024 reasoning/control example exposures and 512 replay exposures; these are sampled exposures, not distinct examples. The earlier main attempt stopped at update 128 under its old-task preservation guard and was not deployed. Selection uses validation only. The final checkpoint is `runs/controller_recovery/best_u000512`; its selected threshold is 1.0, making its deployment policy Always Stay. Improved outer-model performance must therefore be distinguished from a benefit caused by conditional steering.

## Generate with the selected controller

Run this PowerShell command from this directory. It reads the selected checkpoint directly from the completed recovery summary, applies its frozen routing threshold, and prints generated text followed by JSON token counts, timings, CUDA allocation, and cache/transfer statistics.

```powershell
$summary = Get-Content .\runs\controller_recovery\training_summary.json -Raw | ConvertFrom-Json
& WORKSPACE\Bezier-Think\.venv\Scripts\python.exe .\scripts\generate.py --model $summary.selected_checkpoint --policy-json .\runs\controller_recovery\training_summary.json --prompt "What is 17 plus 25?" --max-new-tokens 16 --mode streamed --cache-blocks 2
```

Use `--mode resident` for the same weights and policy with all weights on CUDA. This single-prompt greedy decoder stops at EOS and recomputes context without a KV cache; arbitrary prompt output is not an accuracy benchmark.
