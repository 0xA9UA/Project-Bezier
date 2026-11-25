<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../../../PUBLICATION_SCOPE.md) · [Reader guide](../../../../READ_THIS_FIRST.md).

# GPU/CPU Bézier engine: measured findings

Both requested execution splits work and preserve the network's outputs and gradients. Across the nine primary model/batch settings, a split beat the faster CPU-only/GPU-only baseline in 0 cases. These are execution comparisons of the same model, not an architecture or knowledge-compression study.

The coarse CPU actor / GPU learner used 13.8% less training time than GPU acting/learning on the small RL task, with the same final success rate. GPU routing with CPU neurons also substantially reduced GPU parameter storage by keeping the expert bank in system RAM. These are separate benefits with different timing and capacity tradeoffs.

![Inference and learning timings](comparison.png)

## Inference: CPU input to CPU logits

Values are median milliseconds per whole batch over five interleaved rounds. Four warmup requests precede each repeat; the two trained model sizes use 20 requests per repeat and the untrained stress model uses five. CUDA synchronization encloses the wall-clock measurement. Transfer, allocation, routing, normalization, nonlinear functions and matrix products are included. Startup and model loading are excluded. Every row's input/output agrees with the original CPU network within the saved tolerances.

| Workload | Batch | CPU | GPU | GPU matrices / CPU activations | GPU route / CPU neurons |
|---|---|---|---|---|---|
| digits64 | 1 | 0.475 | 1.061 | 1.619 | 0.891 |
| digits64 | 16 | 0.619 | 1.184 | 1.701 | 1.031 |
| digits64 | 128 | 1.242 | 1.173 | 2.041 | 1.549 |
| emnist256 | 1 | 1.045 | 1.128 | 1.998 | 1.586 |
| emnist256 | 16 | 1.747 | 1.213 | 2.105 | 2.218 |
| emnist256 | 128 | 2.781 | 1.230 | 2.525 | 2.968 |
| stress1024 | 1 | 1.738 | 1.080 | 1.676 | 2.284 |
| stress1024 | 16 | 3.490 | 1.418 | 2.345 | 4.252 |
| stress1024 | 128 | 7.500 | 6.207 | 7.960 | 7.505 |

digits64 is the trained digit model using real held-out inputs. emnist256 uses the trained EMNIST checkpoint with synthetic timing inputs. stress1024 uses random weights and synthetic inputs; it provides no quality or large-model learning evidence. CPU thread counts and CPU expert dispatch were selected on a separate timing pilot by the geometric mean of batch-one and batch-128 latency. All raw repeats and selection data are retained. GPU matrix dispatch gathers selected matrices; CPU grouped dispatch reuses each selected expert matrix. Both implement the same sparse computation.

## Supervised training

Twelve runs: four schedules and seeds 7, 19, 42. Forty epochs, 128-example minibatches, identical initialization and shuffled example orders, AdamW, and one CPU thread. Validation every five epochs selects the saved checkpoint. Wall time includes minibatch preparation, copies, forward/backward, optimizer steps, validation and best-state copies; it excludes model construction, final test evaluation and disk output. The existing split contains 1,078 training, 359 validation and 360 test digits. Means ± sample SD describe three training seeds, not independent dataset uncertainty.

| Schedule | Training seconds | Test accuracy | Test loss |
|---|---|---|---|
| CPU only | 2.65 ± 0.05 | 96.02% | 0.1733 |
| GPU only | 2.24 ± 0.13 | 96.11% | 0.1730 |
| GPU matrices / CPU activations | 2.82 ± 0.02 | 96.11% | 0.1730 |
| GPU routing / CPU neurons | 3.41 ± 0.06 | 96.02% | 0.1733 |

Device copies remain differentiable. CPU-activation forward work produces CPU backward work and gradient transfers; GPU matrices retain their parameters and optimizer state on GPU. Moving parameter-free activation functions to CPU does not shrink the GPU weight bank.

## Reinforcement learning

Fifteen runs: the four schedules plus a CPU actor/GPU learner, with three seeds each. PPO-Clip uses 32 CPU environments, 16-step rollouts, 40 updates and 20,480 environment transitions per run. All schedules receive identical algorithmic budgets. CPU actor/GPU learner synchronizes its whole policy after each update and is synchronous/on-policy; it does not overlap independent actors with learning. Timing includes simulation, sampling, rollout buffers, advantage estimation, PPO updates and policy copying, excluding final evaluation and disk output.

| Schedule | Total seconds | Final success | Initial success | Rollout seconds | Learning seconds |
|---|---|---|---|---|---|
| CPU only | 3.49 ± 0.23 | 100.0% | 30.0% | 0.81 | 2.68 |
| GPU only | 3.43 ± 0.07 | 100.0% | 30.0% | 1.22 | 2.21 |
| GPU matrices / CPU activations | 4.33 ± 0.13 | 100.0% | 30.0% | 1.58 | 2.75 |
| GPU routing / CPU neurons | 4.72 ± 0.14 | 100.0% | 30.0% | 1.24 | 3.47 |
| CPU actor / GPU learner | 2.96 ± 0.07 | 100.0% | 30.0% | 0.81 | 2.12 |

The custom corridor has seven cells, a randomly selected left/right goal, terminal +1/-1 rewards, and an observed twelve-step deadline. Evaluation exhausts ten starting-position/goal combinations using greedy actions, including positions other than the training start. This finite synthetic task is deliberately easy and is not evidence about games, expensive simulators, robotics, or RLHF. All final policies and learning curves are saved. Differences in float32 arithmetic can change sampled trajectories, so these are paired seed/budget comparisons rather than identical trajectories.

## GPU parameter storage and memory

The stress model is shown below. Each memory measurement runs in a fresh process, preventing persistent CUDA workspaces from a previous schedule contaminating the CPU-only result. CPU/GPU columns count stored parameters, not full process RAM/VRAM. Peak CUDA allocation includes this engine's active tensors, backend workspaces and selected-matrix workspace, not the driver/context or other applications. All source weights also remain on CPU during this benchmark; that reference copy is not counted in the engine parameter columns. The original timing-run peaks remain in the raw inference file and include persistent process allocations; use these isolated measurements for the memory comparison.

| Schedule | CPU parameters MiB | GPU parameters MiB | Peak CUDA allocated MiB |
|---|---|---|---|
| CPU only | 260.698 | 0.000 | 0.00 |
| GPU only | 0.000 | 260.698 | 785.84 |
| GPU matrices / CPU activations | 0.000 | 260.698 | 785.84 |
| GPU routing / CPU neurons | 260.570 | 0.127 | 10.02 |

GPU routing can leave almost all model weights in system RAM. That is a GPU-capacity tradeoff; it does not reduce total model storage. The routes here are fixed at entry, so selections for all steps transfer together. Replanning after each activation would require additional dependent communication unless separately optimized.

## Transfer boundaries and reusable pinned buffers

Logical forward-copy counts for one digit input are below. They count tensor payloads, not PCIe transactions. Training also transfers gradients and targets, which are not included in this forward-only ledger.

| Schedule | CPU → GPU copies | GPU → CPU copies | Total payload bytes |
|---|---|---|---|
| CPU only | 0 | 0 | 0 |
| GPU only | 1 | 1 | 296 |
| GPU matrices / CPU activations | 7 | 7 | 3112 |
| GPU routing / CPU neurons | 1 | 2 | 352 |

The CPU-activation split must wait for a GPU preactivation before applying the CPU nonlinearity, then return its result for the next GPU product. Reusable pinned buffers remove some allocation work but cannot remove this dependency. The table gives blocking-time / pinned-time: above 1 means pinned was faster. Buffer lifetimes and CPU-read synchronization are checked. No cross-request pipeline is implemented.

| Workload | Split | Batch 1 | Batch 16 | Batch 128 |
|---|---|---|---|---|
| digits64 | GPU matrices / CPU activations | 0.895× | 0.907× | 1.036× |
| digits64 | GPU routing / CPU neurons | 0.974× | 0.960× | 0.974× |
| emnist256 | GPU matrices / CPU activations | 0.931× | 0.923× | 0.859× |
| emnist256 | GPU routing / CPU neurons | 1.003× | 0.961× | 0.976× |
| stress1024 | GPU matrices / CPU activations | 0.899× | 0.914× | 1.029× |
| stress1024 | GPU routing / CPU neurons | 0.950× | 0.993× | 0.975× |

## Boundaries of the finding

This is a custom scheduler built on eager PyTorch kernels, not a compiler, fused CUDA implementation or production serving engine. It tests one Intel Core Ultra 7 265KF and RTX 5090, FP32 with TF32 disabled, modest batches and a fixed four-step network. Single-host dependent execution was measured; independent-request pipelining, NUMA placement, shared-memory accelerators and larger RL actors are untested. Windows desktop scheduling introduces noise; raw repeats and GPU process snapshots are retained. Small timing differences should not be treated as established speedups.

Numerical agreement establishes correct execution. It does not establish improved accuracy from device placement. A coarse actor/learner split and a fine-grained matrix/nonlinearity split have different communication costs and should be evaluated separately.

## Reproduction and evidence

[Protocol and research sources](../../docs/protocol.md) · [Raw inference](inference.json) · [Transfer-only measurements](transfers.json) · [Classifier results](supervised/results.json) · [RL results](rl/results.json) · [Independent verification](verification.json)

The implementation follows [PyTorch's guidance on pinned transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) and [PPO-Clip equations](https://spinningup.openai.com/en/latest/algorithms/ppo.html). [IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html) is related work on separating acting and learning, not the algorithm or distributed execution system used here.
