# Reversed responsibilities: GPU chooses layers, CPU computes

The requested split works on a trained **50,915,372-parameter** model and an explicitly untrained **269,946,720-parameter** stress model. The trained model is 566 times the size of the earlier 89,968-parameter digit model. All expert weights and full hidden states stay in system RAM. Only the small layer controller resides on GPU.

CPU state → CPU eight-float summary → GPU chooses layer ID → CPU evaluates the Bézier route and two experts → CPU updates state → ask GPU again.

The CPU performs normalization, Bézier control/curve calculations, within-layer search, matrix multiplications, SiLU and decoding. Bézier geometry routes the computation; SiLU remains the expert's scalar nonlinear function. The CPU holds the catalogue and coordinates. A layer ID is a logical address rather than a transfer of expert weights. The reversed controller has 432 bytes of GPU parameters on the trained model and 1152 on the stress model. CUDA context, allocator and driver overhead are additional.

## Larger learned task

The 47-class EMNIST model achieved **77.95%** test accuracy, identical across all seven tested execution variants. All 9,400 held-out examples selected exactly the same layers and experts across CPU, GPU, both split directions, grouped CPU execution and pinned buffers. The maximum absolute logit difference was 1.14e-05.

The model changed layers during **28.97%** of predictions. Correct assigned-group selection improved from **67.64%** at entry to **77.26%** at the fourth decision. This demonstrates that the GPU responds to the CPU's evolving state. The layer groups are assigned sets of four consecutive class IDs, learned with auxiliary supervision; they are not discovered universal categories.

One seed, 12 epochs, 23,500 training examples, 4,700 validation examples, 9,400 official-test examples. The lowest validation classification loss selected epoch 7. Test labels never enter routing. Training used GPU execution; these results do not benchmark reversed-split training speed. This is a handwriting classifier, not a pretrained language model, and its accuracy is not a state-of-the-art claim.

![Examples of actual GPU layer decisions](switches.png)

Examples are the first six held-out inputs that switched layers. Errors remain visible; the selection is illustrative rather than a representative sample. The catalogue maps class IDs to characters. Geometry coordinates and all saved paths come from actual CPU execution.

## Runtime results

Milliseconds per batch from CPU input to CPU output. Five randomized rounds, ten timed requests per round after five warmups. FP32, TF32 disabled, one or eight CPU threads as shown. These numbers include feedback, synchronization and transfers. Weight loading and geometry-cache preparation are excluded. The GPU controller is deliberately small; moving that choice to GPU does not accelerate CPU expert matrix products.

| Workload | CPU threads | Batch | CPU only | GPU only | CPU chooses / GPU computes | GPU chooses / CPU computes |
|---|---|---|---|---|---|---|
| trained_51m | 1 | 1 | 1.446 | 2.402 | 2.724 | 2.244 |
| trained_51m | 1 | 16 | 5.293 | 2.654 | 3.230 | 6.331 |
| trained_51m | 1 | 128 | 16.839 | 3.950 | 5.147 | 18.280 |
| trained_51m | 8 | 1 | 1.265 | 2.437 | 2.770 | 2.031 |
| trained_51m | 8 | 16 | 3.810 | 2.709 | 3.069 | 4.642 |
| trained_51m | 8 | 128 | 9.882 | 4.508 | 5.121 | 10.777 |
| stress_270m | 1 | 1 | 2.146 | 2.802 | 2.908 | 3.066 |
| stress_270m | 1 | 16 | 18.842 | 3.093 | 3.418 | 19.910 |
| stress_270m | 1 | 128 | 86.136 | 8.512 | 9.883 | 89.552 |
| stress_270m | 8 | 1 | 1.450 | 2.383 | 2.937 | 2.355 |
| stress_270m | 8 | 16 | 9.458 | 3.082 | 3.663 | 10.357 |
| stress_270m | 8 | 128 | 36.994 | 8.687 | 9.338 | 39.734 |

Using the default grouped CPU execution, the reversed split beat matching CPU-only grouped execution in **0 of 12** median settings, and beat the fastest tested single-device baseline in **0 of 12**. These are descriptive median counts on one desktop, not significance tests. Raw repeats retain scheduling variability; no universal ordering is claimed.

![Latency across both model sizes and CPU thread settings](latency.png)

The runtime also tested CPU gather, reversed gather, and reversed grouped with pinned buffers. All 84 setting/variant combinations and their five repeats are in timing.json. Grouped dispatch reuses each selected matrix for its assigned inputs, avoiding one gathered matrix copy per input/expert pair. Both CPU-only and reversed execution receive the same option. Grouping changes execution order, not the model equations. The larger stress model is untrained and measures runtime only.

## Memory and communication

Fresh-process measurements at batch 128 and eight CPU threads; parameter bytes are exact FP32 tensor storage. MiB means 1,048,576 bytes. GPU peak means PyTorch peak allocated tensor memory during requests, excluding CUDA context and driver memory. Process working set includes Python, libraries, CPU model state and runtime allocations; it is not just weight storage.

| Workload | Mode | CPU weights MiB | GPU weights MiB | Peak GPU tensors MiB | Process working set MiB |
|---|---|---|---|---|---|
| stress_270m | cpu | 1029.765 | 0.000000 | 0.000 | 1535.2 |
| stress_270m | gpu | 0.000 | 1029.765015 | 1555.960 | 1328.1 |
| stress_270m | gpu_layers | 1029.764 | 0.001099 | 9.163 | 1829.3 |
| trained_51m | cpu | 194.227 | 0.000000 | 0.000 | 821.7 |
| trained_51m | gpu | 0.000 | 194.226730 | 334.060 | 1647.8 |
| trained_51m | gpu_layers | 194.226 | 0.000412 | 9.143 | 1111.4 |

The reversed engine needs four CPU-to-GPU summary copies and four GPU-to-CPU ID copies per four-step request. Per input this is 128 bytes of summaries plus 32 bytes of IDs: **160 logical bytes** total. Input/output already reside on CPU. Small copies still synchronize dependent decisions. Pinned buffers do not remove that dependency. No full activation or expert matrix crosses devices during inference in the reversed split.

This reduces GPU weight residency by placing weights in system RAM. It does not shrink total weights or the saved checkpoint. A CPU-only engine already provides zero GPU weight residency; this is the essential control for interpreting the GPU controller's value.

## Feedback and less frequent decisions

Adding +4/-4 to two CPU hidden-state components after the first step changed 5.01% of subsequent GPU layer decisions in a fixed 1,024-example subset. The first decision stayed identical. Entry-only controls never changed their layer sequence. This tests causality of feedback, not a beneficial perturbation.

| Reconsider every N steps | Test accuracy | Inputs switching |
|---|---|---|
| 1 | 77.95% | 28.97% |
| 2 | 77.23% | 26.28% |
| 4 | 75.40% | 0.00% |

These are post-training approximations using the same weights. Holding the layer fixed for all four steps removes within-run switching. Accuracy differences here do not establish a training advantage over a separately trained entry-only model.

## Validation and boundaries

All six previous digit checkpoints also reproduce their predictions and routes using the new reversed mode. Tests cover device placement, differentiable copies, all parameter gradients, expert membership, sphere containment, feedback, cache equality and pinned-buffer lifetimes. Independent verification checks dataset/checkpoint/source hashes, all 84 timing records, saved trajectories and six fresh-process memory records.

This is synchronous eager PyTorch on one desktop, with no fused kernels, request pipelines, model paging, LLM evaluation or reinforcement-learning experiment. GPU process checks reject known competing Python workloads, while unreadable or non-Python GPU activity and desktop scheduling remain possible interference. The model sizes test execution scaling, not a claim that arbitrary existing language-model layers can be reorganized without retraining.

[Commands and implementation](../../REVERSED.md) · [Prospective protocol and primary references](../../docs/reversed_protocol.md) · [Training results](trained/result.json) · [Raw timing](timing.json) · [Execution checks](execution_checks.json) · [Feedback tests](feedback.json) · [Layer catalogue](catalog.json) · [Verification](verification.json)
