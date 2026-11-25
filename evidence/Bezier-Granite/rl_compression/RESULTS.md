<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../../PUBLICATION_SCOPE.md) · [Reader guide](../../../READ_THIS_FIRST.md).

# Reinforcement-learning compression results

The frozen RL artifact uses **16.464 GiB**, **25.0% less** than the six-bit shared parent and 69.8% less than the original BF16 tensor payload.
It **passes** the prespecified final quality screen. Validation selection had **passed** the same screen.

| Model | Stored GiB | Test perplexity | ARC-Easy | Original-correct retained | Final screen |
|---|---:|---:|---:|---:|---|
| Original BF16 | 54.532 | 17.9369 | 82.62% | 100.00% | passed |
| Six-bit shared parent | 21.960 | 18.0199 | 80.86% | 96.22% | passed |
| Fixed 4-bit Bézier | 15.429 | 19.0809 | 82.42% | 96.45% | passed |
| RL mixed Bézier | 16.464 | 18.2305 | 80.86% | 95.74% | passed |
| Random mixed Bézier | 15.636 | 18.6695 | 81.84% | 95.51% | passed |

## Validation selection

| Candidate | Stored GiB | Validation perplexity | ARC-Easy | Original-correct retained | Screen |
|---|---:|---:|---:|---:|---|
| rl_1 | 15.890 | 22.3535 | 78.91% | 95.19% | failed |
| rl_2 | 16.464 | 21.9343 | 79.69% | 96.15% | passed |
| rl_3 | 16.464 | 21.7906 | 80.47% | 96.15% | passed |
| random_1 | 15.636 | 22.4136 | 79.69% | 94.23% | failed |
| random_2 | 15.429 | 22.8470 | 78.12% | 93.27% | failed |
| fixed3 | 12.163 | 26.6382 | 76.56% | 91.35% | failed |
| fixed4 | 15.429 | 23.1065 | 78.12% | 93.27% | failed |
| fixed6 | 21.960 | 21.4337 | 82.03% | 97.12% | passed |

Selected RL candidate: `rl_3`. Selected random candidate: `random_1` (experimental fallback after failed validation). Selection used 128 validation questions and 4,096 validation tokens. A smaller candidate that failed the screen was not substituted after seeing final-test outcomes.

Fresh test: 32,768 next tokens in 128 WikiText-2 windows and 512 ARC-Easy questions. QA scores use mean answer-token log likelihood. The original and every compressed condition use the same tokenizer, BF16 operators, explicit attention masks and layer-streaming evaluator. Screen: perplexity ratio at most 1.10, QA loss at most two percentage points, and at least 95% retention of original-correct answers. This is a limited point-estimate screen, not statistical equivalence or a general capability guarantee.

## What was preserved

All 445 compressed projection matrices retain cubic Bézier quantization tables with four learned control values and a scale per 64 weights. RL chooses 3, 4 or 6 bits for each consecutive four-layer section. All candidates are encoded directly from the original BF16 weights. Embeddings, head, norms, tokenizer and the adaptive layer-32 shared MLP are unchanged. The replacement retains 512 latent features, a rank-128 shared update, sixteen spatial groups, four selected groups per step and three activation-dependent Bézier routing steps. The old layer-32 MLP is physically absent. This preserves the existing single-block experiment; it does not convert every transformer block to shared routing.

RL section precisions, layers 0–3 through 60–63: `[4, 4, 4, 3, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6]`.

## What reinforcement learning did

A categorical compression controller received actual end-to-end training loss and storage rewards. REINFORCE made sixteen gradient updates from 64 sampled assignments, using only 2,048 training tokens. It trained precision choices, not Granite reasoning or RLHF. An independent random search used 64 assignments from the same starting distribution. Fixed 3-, 4-, and 6-bit controls were additional evaluations. Only validation data selected artifacts; selection and input hashes were frozen before final evaluation.

| RL compared with | QA change, pp (95% paired interval) | Perplexity ratio (95% paired interval) |
|---|---:|---:|
| Six-bit shared parent | +0.000 [-1.758, +1.758] | 1.0117 [1.0044, 1.0193] |
| Fixed 4-bit Bézier | -1.562 [-2.930, -0.195] | 0.9554 [0.9488, 0.9622] |
| Random mixed Bézier | -0.977 [-2.734, +0.781] | 0.9765 [0.9704, 0.9826] |

Intervals use 5,000 paired bootstrap draws over questions or text windows. They do not cover training-seed uncertainty. RL shortlisted three candidates and random search two for validation; both had the same 64-sample training budget. A smaller result than the six-bit parent does not by itself establish that RL beats fixed quantization or random search. No comparison with an optimized four-bit AWQ/GPTQ/GGUF implementation was performed.

![Search rewards, RL precision assignment and fresh-test trade-off](results.png)

## Packed inference and verification

Persistent GPU tensors: **16.454 GiB**. Measured 32-token inference/KV-check allocated peak: **17.790 GiB**; reserved peak: 18.240 GiB. Packed versus independently reconstructed dense last-token logits: maximum difference 0.0. KV-cached inference retained the same top token; maximum BF16 logit difference was 0.3828125.

All 76 manifest-listed files were hashed, and the manifest hash was frozen separately; 445 matrices were independently checked with 1335 sample bitstream decodes. Source sizes, modification times and index hash match. The routing block and static files match the parent exactly.

Generation smoke test: 23 tokens at 0.36 tokens/s; allocated peak 17.796 GiB. Raw output is in generation.json.

Same-session parent comparison on the identical prompt: 0.20 tokens/s and 23.288 GiB allocated peak. RL/parent generation-speed ratio: 1.817. This is one short eager-decoding run, not an optimized throughput benchmark. It does not establish a general speed advantage.

The same parent artifact previously generated this output at 1.13 tokens/s, versus 0.20 tokens/s in the current run. This substantial timing variation was not diagnosed; no reliable throughput conclusion is drawn. The generated code and measured memory allocation were stable.

On 256 validation activations from the RL-compressed model, a +1/−1 latent intervention after the first routing step changed 24.61% of later group selections. The first step was unchanged. This checks that feedback-dependent routing still operates; it does not establish a benefit to answer quality.

Memory measurements exclude driver/display overhead and depend on context and batch size. The custom eager PyTorch decoder reconstructs each matrix when needed; no speed advantage is claimed. Artifacts are standalone for this loader, with no original model needed for inference. They are not directly compatible with Ollama or llama.cpp.

See [README](README.md) for commands, [the frozen protocol](protocol.md) for the experiment design, [summary.json](summary.json) for numeric evidence, and search/ for every action, reward, log probability and policy update.

Research precedent: [HAQ, Wang et al., CVPR 2019](https://arxiv.org/abs/1811.08886) uses RL for mixed-precision quantization. Our controller, reward, model and Bézier representation differ; this is not a reproduction of HAQ results.
