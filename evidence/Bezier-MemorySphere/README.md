# Memory Sphere

A full-stack, from-scratch spatial recurrent token model for the requested
clustered-KV / Bezier-routing / learned-surface-exit experiment. It contains no
Qwen processing blocks, no pretrained encoder or decoder, and no inference call
to another model. The external Qwen runs are independent capability references.

The shared core has 16 addressed processing regions, a 256-dimensional state,
64-dimensional keys and four overwritten scratch KV slots. Its 1,538,295 weights
are all trained locally. Coordinates locate regions and routes; high-dimensional
states still carry the information. Input formation uses one pass through the
same spatial processing bank, followed by the configured number of reasoning
visits. These are shared operations rather than a stack of independent layers.

The experimental model is not yet a general coding assistant. Read the measured
results before drawing a quality or efficiency conclusion.

The completed comparison covers 44 conditions and 2,640 stage-runs. Adaptive
Bezier and straight routing each passed 6/120 stage-runs across two seeds;
direct routing passed 7/120. The external 14B reference passed 48/60. No sphere
condition solved the long-chain or MBPP transfer tasks or a complete project.
All three routing controls kept single-row tensor allocation flat through 128
internal steps. This bounded-memory behavior did not produce a demonstrated
coding-quality advantage. Independent regrading of 1,400 unique code/check pairs
found zero score discrepancies; the wider-batch replay produced identical outputs.

## Results and protocol

- `results/report.html`: measured success, timing, memory and fitted route plots.
- `results/summary.json`: every condition and paired comparison.
- `frozen_plan.json`: exact run plan, source hashes and environment.
- `docs/protocol.md`: mechanism, data splits, controls and limitations.
- `results/<condition>/outputs/<task>/trace.json`: every model action and score.
- `models/<routing>_s<seed>/`: standalone safetensors, config and own tokenizer.

## Run locally

From this directory in PowerShell, use the already installed local runtime:

```powershell
& '..\Bezier-Think\.venv\Scripts\python.exe' -m sphere_memory.infer --model bezier_s7 --steps adaptive
```

This produces one tool action for the first validation task without executing
it. Supply `--task path.json` for an object containing `spec`, `initial` (Python
source) and `public` (a list of assertion strings). Fixed `--steps` values can be
2, 4, 8, 16 or 32. The local runtime path shares installed Python packages only;
it does not load an older model. Inference requires torch, tokenizers and
safetensors. The checked versions are in `frozen_plan.json`.
Use `--device cpu` to run the same weights on the CPU.

```powershell
& '..\Bezier-Think\.venv\Scripts\python.exe' -m sphere_memory.verify
& '..\Bezier-Think\.venv\Scripts\python.exe' -m sphere_memory.pipeline
& '..\Bezier-Think\.venv\Scripts\python.exe' -m sphere_memory.report
```

The pipeline resumes completed fits and conditions. It refuses changed frozen
source or data. Its GPU work is serial; logs and outputs are durable between
jobs. Do not regenerate the data or rerun `freeze` over an existing experiment
when reproducing results: use the frozen artifacts and hashes. Development
checkpoint folders named `pilot*` are excluded from the final comparison.
The external-tool-schema clarification and quality-batching acceleration are
recorded in `docs/protocol_amendments.json`; earlier frozen plans are retained
beside it. Final small-model quality runs batch up to 48 independent tasks,
while all latency/memory probes use one row. The initial smaller-batch seed-7
Bezier quality runs are retained separately and replayed at the final batch size.

The terminal-style harness supports a single Python file and bounded public
tests. It is not an unrestricted shell or a real-repository coding benchmark.
Internal scratch memory stays fixed with depth; text-context memory and training
autograd memory do not. Dense evaluation of all 16 regions means this experiment
does not measure optimized sparse dispatch. Surface contact is an output address,
not proof that an answer is correct.
