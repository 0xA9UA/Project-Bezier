# Exact compression methods

All source matrices are taken from the pretrained checkpoint converted to float16. Fitting and reconstruction use float32; the reconstructed model uses float16. Compress every `torch.nn.Linear` weight inside `model.layers`, preserving all other parameters. Tied input/output embeddings are stored once. Every complete artifact includes those preserved parameters, codec metadata, model config, tokenizer and generic decoder sources. The shared Python, PyTorch, CUDA and Transformers installations are excluded, as with ordinary model-file size reporting.

## Direct cubic representation

For a row block of B scalar weights, use fixed positions t_j = j/(B−1) and the four cubic Bernstein basis functions. Let A[j,k] = binomial(3,k) (1−t_j)^(3−k) t_j^k. Fit c = A⁺w by ordinary least squares and store four float16 coefficients. Reconstruct w_hat = Ac, then round to float16.

With B=8, eligible weights cost 64/8 = 8 bits each. With B=16, they cost 4 bits each. Sample positions follow from the block layout and are not learned or secretly stored per weight. This is a scalar Bézier weight representation, distinct from the earlier network's sphere routing.

## Sorted cubic representation

Sort each block of 32 weights, fit the same cubic to the sorted values, and store the sorting permutation. At reconstruction, scatter the predicted sorted values back into their original positions. Four float16 coefficients cost 2 bits per original weight. The permutation is represented as a five-bit position for every weight, costing another 5 bits. Total: 7 bits per eligible weight, before container metadata. The implementation packs eight five-bit values into five bytes; it does not count an unpacked byte array as five bits.

The sorted fit is ordinary least squares. It does not impose continuous monotonicity between sampled positions. Every stored permutation and coefficient is charged to the payload.

## Grouped uniform quantization

Use a float16 scale s = max(abs(w))/q_max for each consecutive row group of 64 values. Round the scale to its actual stored precision before assigning codes. For int8, q_max=127; for int4, q_max=7. Assign q=clip(round(w/s),−q_max,q_max), then store q+q_max as an unsigned code. The four-bit method leaves one of its sixteen codes unused. A small positive scale floor handles all-zero/tiny groups.

The resulting eligible-matrix budgets are 8.25 and 4.25 bits per weight, including scales. These are simple symmetric round-to-nearest baselines, not GPTQ or AWQ.

## Curve-controlled four-bit codebook

Each matrix has sixteen reconstruction levels y_j = sum_k c_k B_k³(j/15), with endpoints c_0=−1 and c_3=1. Each group of 64 weights has a float16 absolute-maximum scale. Normalize the weights by their stored scales, select the nearest of the sixteen levels, and store one four-bit index per weight.

Fit the two interior controls by twenty alternating nearest-level assignments and weighted least-squares updates, using up to 131,072 deterministically sampled normalized weights from that matrix. A line search keeps the sixteen discrete levels ordered. All four controls are stored in float16. Assign the final indices against levels reconstructed from these rounded controls, matching deployment.

This uses 4.25 bits per eligible weight plus eight bytes per matrix for the controls. The matrix is **not** represented by four numbers alone: every weight still has an index and every group has a scale. This formulation tests whether a cubic parameterization of the quantization palette improves the size/quality tradeoff.

## Free codebook comparison

Use the same normalization, scales, samples, twenty alternating updates and four-bit indices. Fit the interior levels independently rather than restricting them to a cubic. Both endpoints remain fixed at −1 and 1, as in the Bézier codebook. Store all sixteen float16 levels, costing 32 bytes per matrix. “Free” refers to freedom from the cubic constraint; it does not mean the endpoints are unconstrained.

## Randomized low-rank comparison

For an m×n matrix and a target b of 8 or 4 bits per original element, choose rank r = floor(m n b / [16(m+n)]), bounded to a valid positive rank. Use randomized SVD with eight oversampling columns and two power iterations, with a recorded seed per matrix. Store balanced float16 factors U_r sqrt(S_r) and sqrt(S_r) V_rᵀ. Reconstruct their product in float32 and cast to float16.

This is a generic unweighted low-rank approximation with no activation calibration or recovery training. It is not the best possible low-rank language-model compressor. Its actual factor bytes and achieved rank are recorded.

## What the size and timing numbers mean

The reported reduction is reference complete-artifact bytes divided by compressed complete-artifact bytes. It includes preserved tensors and tokenization assets; an eligible-matrix bit budget is not the whole-model budget. The artifacts use safetensors for numerical payloads and JSON for metadata. No executable pickle or remote model code is needed.

At load time the codec reconstructs a dense model. Thus its steady-state GPU parameter footprint is the original model's float16 footprint. Decode timings include reading the artifact and transferring/reconstructing its tensors into a model that already exists; they are not guaranteed cold-cache disk measurements. Fresh construction is separately exercised by artifact verification. Any future claim of reduced inference memory would require a different execution path that consumes the compressed representation directly.
