# Bezier-Steer

A generative proof of concept testing whether Bezier-guided internal computation can produce **more distinct correct solutions for a given computation budget**.

The model synthesizes four-step Boolean programs from a complete specification. It learns multiple implementations, emits program tokens, and exports verified Python functions. A specification is the 16-row truth table of a function with four Boolean inputs. This finite domain lets the verifier check every possible input. The experiment measures structural variety and semantic correctness, together with complete request cost.

All new source, data, checkpoints, measurements, caches and reports live in `WORKSPACE\Bezier-Steer`. The project uses the already installed Python environment at `WORKSPACE\.venv`; its exact package snapshot is copied to `requirements-lock.txt`. The original sphere experiments remain separate. Training and inference measurements use the local NVIDIA GeForce RTX 5090.

## Results and evidence

**The prototype can steer outputs with curves, but this experiment did not demonstrate a gain in efficiency, correctness or storage.** At 64 candidate programs per task, the results below average three training seeds. Latency includes generation, all failed attempts, exhaustive verification and deduplication; these rows have the same attempt count, not identical compute cost.

| Model | Tasks solved | Distinct correct programs / task | Complete request | Inference checkpoint |
|---|---:|---:|---:|---:|
| Bezier routes | 85.29% | 3.217 | 5.81 ms | 1.278 MiB |
| Straight routes | 85.48% | 3.201 | 5.10 ms | 1.277 MiB |
| Ordinary non-spatial routing | 84.64% | 3.281 | 5.01 ms | 1.247 MiB |
| Fixed routing | 86.91% | 3.473 | 4.91 ms | 1.243 MiB |
| No expert planner | 85.74% | 3.398 | 5.05 ms | 0.736 MiB |

There is a small, observable steering effect: with the starting hidden state fixed and token decoding deterministic, trying 64 curves solved **14.71%** of tasks, compared with **10.55%** for a single fixed route. However, only **0.52%** of tasks produced two or more distinct correct structures through curve variation alone. Most variety in the full model came from ordinary token sampling. Increasing Bezier updates from 2 to 4 to 8 gave solve rates of 85.03%, 85.29% and 85.09%, while average request time increased from 4.92 to 5.81 to 7.11 ms.

Only **7.63% of raw Bezier candidates were correct** in the main comparison. The 85.29% solve rate means at least one of 64 attempts passed; it is not single-attempt accuracy. The exported solutions pass an exact verifier covering all 16 inputs. The difference from fixed routing was -1.63 percentage points, with an exploratory 95% interval of [-4.36, +1.04]; three seeds do not settle small differences. These results give no evidence for a general reasoning or knowledge-compression advantage, and do not rule out a different curve-based architecture.

- [Complete experiment report](runs/main/report.html)
- [Actual verified program alternatives and their routes](runs/main/examples.html)
- [Machine-readable results and uncertainty intervals](runs/main/summary.json)
- [Frozen plan, environment and source hashes](runs/main/manifest.json)
- [Independent artifact and prediction verification](runs/main/verification.json)
- [Prospective protocol](docs/protocol.md)
- [Curve-only intervention definition](docs/steering_intervention.md)
- [Runnable GPU demo: four verified Python implementations](runs/demo/result.json)

The main study comprises five architectures and three training seeds, with 1,970 training functions, 256 validation functions and 512 held-out functions. No function is shared between splits, even under a different implementation. Each selected function has at least four known normalized implementations. The dataset and all reference programs are saved in `data`.

## What the curve does

The specification encoder produces a 128-dimensional task representation. A sampled eight-dimensional latent code affects the initial hidden state and the route. The Bezier controller predicts surface entry/exit points and two strictly interior control points. At 2, 4 or 8 internal steps, the hidden state is updated by the two nearest of eight learned expert regions. The generator then decodes a program, with the task and resulting plan available throughout decoding.

The controls determine which computations run. They do not directly contain the program, and reaching the exit does not establish correctness. Actual task information remains in the hidden state. The conditional variational training objective has a program-conditioned posterior, which is removed from inference checkpoints. Inference receives only the specification and random draws.

Comparison models share the same encoder, decoder, latent mechanism and training settings: straight routes, ordinary non-spatial routing, fixed pairs of experts, and a smaller generator without expert planning. Training sees all three update counts. This version has no adaptive halting or branch-and-revise feedback.

## What counts as a successful alternative

Every emitted program is executed on all 16 possible inputs. A correct program must exactly match the supplied specification. Structural normalization removes unused gates, commuting/associative rearrangements and simple identities before counting distinct correct implementations. Mere renaming or dead-code padding does not earn diversity credit. Different implementations still compute the same required function.

The experiment separately measures latent variation with greedy decoding, token sampling with a fixed latent, and both together. Interventions freeze or straighten curves, bypass interior updates, and vary only the curve with the initial state fixed. This avoids assuming that every changed output came from the geometry.

Correct candidates, solved-task coverage, distinct verified programs, total latency, memory and checkpoint size are all reported. Timings count failed attempts, transfers, generation, exhaustive checks and deduplication. They exclude model loading and startup. The wall-budget comparison selects the largest tested candidate count fitting measured latency; it never chooses a candidate count using test accuracy.

## Use the model

Run these commands from `WORKSPACE\Bezier-Steer`:

```powershell
# Inspect the supplied truth specification and generate up to 64 candidates.
# The example mask here specifies (a AND b) OR (c XOR d).
..\.venv\Scripts\python.exe -m bezier_steer.synthesize --truth 0x8ff8 --candidates 64

# Vary the internal latent plan while keeping token decoding greedy.
..\.venv\Scripts\python.exe -m bezier_steer.synthesize --truth 0x8ff8 --mode latent --steps 4

# CPU inference is supported for interactive use.
..\.venv\Scripts\python.exe -m bezier_steer.synthesize --truth 0x8ff8 --device cpu
```

Bit `r` of the mask is the required output for input row `r`; bits 0, 1, 2 and 3 of that row specify `a`, `b`, `c` and `d`. Each prediction creates a new directory under `runs` with the full candidate audit, output truth tables and separate Python files for distinct verified solutions. If no candidate succeeds, the command explicitly reports that result. Successful examples are not guaranteed for every specification or seed.

## Reproduce the experiment

```powershell
# Reproduce into a fresh output directory using the saved dataset.
.\scripts\run.ps1 -Output runs\repeat -Device cuda

# Run the independent geometry, interpreter, generator and selection checks.
..\.venv\Scripts\python.exe -m pytest -q

# Re-score every candidate and regenerate each primary checkpoint's outputs.
..\.venv\Scripts\python.exe -m bezier_steer.verify runs\main --device cuda
```

The launcher trains every model before running timing, then generates the curve-only diagnostic, reports and examples, and verifies the saved results. Completed jobs can resume with an unchanged plan and scientific sources. An interrupted training job restarts from its seed. The main source snapshot is under `runs/main/source`; pilot snapshots and learning curves are preserved separately. Report and verification code can be maintained without changing scientific sources.

For a separate installation, use a local virtual environment and install the pinned dependencies with the PyTorch CUDA 12.8 wheel index available. The delivered environment is Python 3.14.3 with PyTorch 2.10.0+cu128. `pyproject.toml` lists the core dependencies; `requirements-lock.txt` records the exact shared environment used here.

## Limits

This tests small program/circuit synthesis. It does not establish natural-language coding, repository repair, design aesthetics or broad reasoning ability. Syntax is enforced by a common grammar; it is not evidence of learned linguistic coherence. The verifier guarantees accepted outputs only for this finite Boolean domain, so coverage and raw-candidate error rates remain essential. Three training seeds provide limited uncertainty estimates. Structural diversity can still include variations that a human would consider minor.

The curve-only condition is a post-training intervention; the network was originally trained with a shared latent influencing both the state and geometry. A failure there would not prove that a model trained specifically for isolated curve steering cannot work. Likewise, a positive result on this small task would not establish a general advantage of Bezier curves. Raw results, prospective decisions and measured limitations are retained so these distinctions remain reviewable.
