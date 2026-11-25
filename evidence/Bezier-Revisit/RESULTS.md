# Shared recurrence and Bézier revisits: coding results

At the prespecified eight-pass budget, Bézier routing passed 28/72 stage trials, ordinary recurrence 30/72, and non-geometric routing 28/72. This pilot provides no clear evidence that Bézier routing improves coding success over the matched controls. Allowing Bézier returns changed success by +1.39 percentage points relative to blocking returns in the fitted model. This intervention also changes the routing distribution, so it does not isolate a benefit from geometry itself.

This completed pilot compares the same stored parameters and matrix-computation budget. It does not compress the pretrained base or establish general coding-agent superiority.

## Primary comparison: eight internal LM passes

Each request uses 24 sequential shared-field applications. Scores cover 24 fresh MBPP functions and four reused three-stage projects per fitting seed.

| Method | Stages passed, seed 7 / 19 | Function trials passed | Complete projects | Fixed 32-token latency | Peak GPU allocation |
|---|---:|---:|---:|---:|---:|
| Bézier routing | 14/36 / 14/36 | 26/48 | 0/8 | 0.834 s (0.681–1.086) | 2.920 GiB |
| Non-geometric routing | 14/36 / 14/36 | 26/48 | 0/8 | 0.734 s (0.681–0.936) | 2.920 GiB |
| Ordinary recurrence | 14/36 / 16/36 | 28/48 | 0/8 | 0.713 s (0.689–0.823) | 2.920 GiB |

All primary arms: **1,545,352,240 total parameters**, **1,637,936 trainable**, including **548,400 shared-field parameters**. Per latent pass the field performs **834,944 matrix MACs**, with identical full-LM pass counts. Scalar routing overhead differs slightly and is included in elapsed timing.

Latency is the median of six batch-one probes (three per seed), followed by the observed minimum–maximum range, on an identical prompt with exactly 32 output tokens. The GPU is an RTX 5090. The probes are noisy and do not establish a general throughput ranking. GPU allocation excludes driver/context/display memory. Quality evaluation uses batches of four; its amortized time is not single-agent latency.

## Paired uncertainty

| Comparison | Stage-success difference | 95% paired interval |
|---|---:|---:|
| bezier minus ordinary | -2.78 pp | [-7.35, +0.00] |
| bezier minus nongeometric | +0.00 pp | [+0.00, +0.00] |
| bezier returns minus blocked | +1.39 pp | [+0.00, +4.69] |
| nongeometric returns minus blocked | -1.39 pp | [-4.69, +0.00] |

Intervals resample 28 whole task families, averaging the two fitting seeds. They do not estimate uncertainty across new training seeds. A [0, 0] interval from identical observed outcomes is a degenerate bootstrap, not proof of population equivalence. Returning-minus-blocked compares a fitted policy with a post-training intervention; it is not a retrained no-return baseline.

## All budgets and controls

| Method | LM passes | Returns allowed | Stage successes | Projects completed | Generated tokens | Agent actions |
|---|---:|---|---:|---:|---:|---:|
| Bézier routing | 4 | yes | 27/72 | 0/8 | 12,687 | 268 |
| Bézier routing | 8 | yes | 28/72 | 0/8 | 12,893 | 277 |
| Bézier routing | 8 | blocked | 27/72 | 0/8 | 13,218 | 276 |
| Bézier routing | 16 | yes | 26/72 | 0/8 | 13,311 | 279 |
| Zero-step trained control | 0 | not applicable | 30/72 | 0/8 | 12,667 | 290 |
| Non-geometric routing | 4 | yes | 29/72 | 0/8 | 12,983 | 276 |
| Non-geometric routing | 8 | yes | 28/72 | 0/8 | 13,563 | 287 |
| Non-geometric routing | 8 | blocked | 29/72 | 0/8 | 13,959 | 283 |
| Non-geometric routing | 16 | yes | 31/72 | 0/8 | 13,678 | 275 |
| Ordinary recurrence | 4 | not applicable | 29/72 | 0/8 | 13,550 | 268 |
| Ordinary recurrence | 8 | not applicable | 30/72 | 0/8 | 14,761 | 278 |
| Ordinary recurrence | 16 | not applicable | 28/72 | 0/8 | 13,357 | 278 |

![Success, latency and GPU memory](results/comparison.png)

## Did routes actually revisit regions?

| Method, eight passes | Actions with leave-and-return visits | Region-return events | Primary-region returns |
|---|---:|---:|---:|
| Bézier routing | 277/277 | 10051 | 3711 |
| Bézier routing (blocked) | 0/276 | 0 | 3387 |
| Non-geometric routing | 287/287 | 10245 | 3969 |
| Non-geometric routing (blocked) | 0/283 | 0 | 3311 |

A return requires the region to have been absent from the active set for at least one sample. Staying in the same region does not count. Primary-region statistics separately track the largest mixture weight; a top-four return need not change the dominant region. Conversely, the dominant label can switch back among continuously active regions even when leave-and-return visits are blocked. Frequent returns alone do not show useful reasoning: a controller may oscillate without improving its answer.

![Recorded sampled route and region sequence](results/route_example.png)

## Verification and boundaries

Re-executed saved programs for **864 coding-stage trials** and audited **3335 actions**. All six adapters passed hash, finite-weight and independent field-reload checks. Source/data hashes match the frozen protocol. No-return interventions recorded zero forbidden return events. Fixed-length probes agree in processed token positions and attention-cell budget at each pass count.

The primary eight-pass budget was selected before fitting or held-out scoring. Four and sixteen passes are secondary comparisons. Each arm receives the same 192-update fitting schedule and common 160-update protocol-training warm start. Checkpoints are selected with validation action loss only. Hidden tests never enter feedback, training or stopping.

These are small restricted Python tasks. MBPP may have occurred in pretraining, and the four authored projects were used in an earlier experiment. This adds a shared adapter and continuous latent passes to a 1.5B LM; it does not replace its transformer stack. Fitting uses only 192 updates of action imitation after the common warm start, with the base weights frozen. Dense evaluation matches arithmetic and does not establish a sparse-dispatch speedup. Two seeds and four cumulative projects are insufficient for broad claims. Geometry is computed and recorded as common instrumentation in all arms, but only the Bézier arm uses positions for selecting regions. All latency results include that instrumentation.

All three primary methods share their weights across internal steps. Their comparison isolates routing within a shared recurrent architecture; it does not measure the independent gain from weight sharing versus an untied network. Coding scores combine source correctness with following the JSON tool protocol. A changed route can change either. The complete final programs for every Bézier revisit-ablation disagreement are saved for inspection. The common warm-start zero-step controls receive less further training and compute, so their scores are contextual rather than a matched recurrence ablation.

[Machine-readable summary](results/summary.json) · [Task outcomes CSV](results/task_outcomes.csv) · [Revisit disagreement examples](results/revisit_disagreements.json) · [Independent audit](results/audit.json) · [Frozen protocol](docs/protocol.md) · [Run instructions](README.md)

Bézier routing with versus without returns: 44/56 first actions were identical on the same initial input; 57/72 final programs and 53/72 complete action trajectories were identical. Later project stages can start with different carried code.

Non-geometric routing with versus without returns: 45/56 first actions were identical on the same initial input; 61/72 final programs and 57/72 complete action trajectories were identical. Later project stages can start with different carried code.

## Recorded limitations from the final audit

The frozen trainer flushes detailed training records every ten updates. Its trace therefore contains updates 1–190, while the saved checkpoint metadata and completion record identify update 192. Those 190 recorded task/step/learning-rate entries match exactly across methods within each seed. The final two schedules are reconstructable from the seed and frozen source, but their individual loss and gradient records were not saved. The audit records this gap rather than inventing measurements.

A post-experiment standalone demo of mbpp_501 with the seed-7 eight-pass Bézier checkpoint failed, although the original batch-of-four evaluation passed. A diagnostic replay reproduced all four original first actions exactly; running the target alone twice produced the same first action on both attempts, but a different action from the batch. This establishes batch-sensitive generation in this BF16 setup. Its precise numerical cause and its prevalence across the benchmark were not isolated. The benchmark scores remain those from the prespecified batch configuration. This sensitivity further limits interpretation of a one-stage ablation gain.

[Batch diagnostic and exact actions](results/batch_sensitivity.json) · [Standalone demo log](demo.log)
