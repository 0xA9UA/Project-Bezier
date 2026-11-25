# Memory Sphere: prospective full-stack experiment

The user explicitly selected replacement of the entire processing stack.
The runnable experimental model must contain no Qwen blocks, pretrained input
encoder, pretrained output decoder, or calls to another LM during inference.
It uses a new tokenizer, learned token embeddings/readout, and a single shared
spatial recurrent core. Input/output lookup and projection remain necessary
interfaces; they are trained here rather than inherited from Qwen.

This is a small model trained from scratch locally. First measure the proposed
mechanism on generated Python repository tasks with controlled dependency depth,
then separately measure transfer to unused MBPP functions. Do not present the
generated task distribution as general coding-agent or larger-model parity.

## Mechanism

Each token enters at a learned surface address. A causal, four-token local
embedding window is processed once through the same spatial region bank used by
the recurrent core. This is one common input-formation pass per real token; the
reported 2/4/8/16/32 budgets count subsequent internal reasoning visits. All wall
times and memory measurements include input formation. No independent input
encoder weights or pretrained blocks are present. The resulting states become
context keys and values through learned projections. Soft
query/key clusters are associated with three-dimensional addresses in a ball.
The same shared core repeatedly reads address-gated context and a fixed-size
scratch KV memory. Scratch slots are overwritten; internal iterations never
append tokens or context KV entries. Every selected region has learned nonlinear
parameters, used through a shared processing bank rather than a stack of
independently weighted transformer layers.

The controller predicts three cluster distributions, whose spatial barycenters
are cubic control points. A cubic segment is sampled sequentially at 1/3, 2/3,
and 1, with the state updated at each sample. The next segment is replanned from
the new state. Internal route endpoints remain in the interior. A separately
learned halting decision triggers a final segment to a surface exit; the high-
dimensional state produces the output token. Boundary contact is not treated as
proof of correctness. Maximum budgets are reported separately from learned exits.

Primary arms: cubic Bézier, straight segments, and direct learned cluster
selection. They use the same tensors, dimensions, clustered memory capacity,
training examples/order, and stopping supervision. All region operators are
evaluated before selecting a mixture, matching matrix work. This tests routing
quality rather than optimized sparse dispatch.

## Training and measurement

Train all model weights from scratch with supervised coding actions. Teach the
halting predictor using training-only action losses at different depths plus a
compute cost; never use test answers or hidden test results for an inference
stopping decision. Split generated program specifications before training the
tokenizer or model. Freeze seeds, data hashes, sizes and optimizer settings after
development checks and before final fits/test scoring.

Evaluate adaptive halting and fixed budgets 2, 4, 8, 16, 32. Extend memory/stability
probes to 64 and 128 without changing weights. Include clustering and halting
interventions that reuse fitted weights, explicitly labelled as interventions.
Measure task success, full-project completion, actual internal steps, learned
and forced exit counts, route traces, latency, inference memory, model bytes,
cluster utilization and route/exit interventions. Confidence intervals must
respect shared task families and training seeds.

Prior project files and model assets remain untouched. A matched direct-routing
model is the primary control. Existing local language models may be evaluated
as external references, but do not supply computation to the sphere model.

## Frozen local pilot settings

Six final fits: three routing arms, seeds 7 and 19, 3,000 updates each, batches of
16 examples. Every fit starts from identical per-seed random weights and uses the
same sampling order. AdamW learning rate 0.0006, 50-step warmup, cosine decay to
15%, weight decay 0.01, gradient clipping 1.0. BF16 autocast for training, FP32
weights and evaluation. Width 256, key width 64, sixteen regions of bottleneck
width 64, four overwritten scratch slots, vocabulary 2,048, context limit 1,024.
There are 1,538,295 parameters; none is pretrained.

First 100 updates cycle depths 2/4/8; subsequent updates cycle 2/4/8/16/32.
Language loss averages the available powers-of-two checkpoints in each unroll.
Halting is supervised only on 32-step training unrolls: choose the checkpoint
minimizing per-token reference NLL plus 0.012 times its step count, then supervise
the hazard through that checkpoint. Halting loss weight 0.2. Its input state is
detached so it cannot distort the language state to manufacture an easy stop.
Minimum two steps, probability threshold 0.5, adaptive cap 32. Auxiliary cluster
loss is 0.01 times marginal KL-to-uniform plus 0.005 conditional entropy. Neither
the stop target nor any hidden tests are accessible to inference.

Data: 6,000 generated training specifications plus 96 previously selected MBPP
training functions; 12,192 construction/repair/tool-continuation examples. 108
validation tasks. Final test: 24 fresh generated depth-1-to-6 programs, twelve
depth-8-to-12 programs, four three-stage update projects, and twelve MBPP test
functions unused by both prior local pilots. Tests use six tool actions, 256
new tokens per action, model-generated JSON, carried project files, and a bounded
pure-Python checker. `pytest -q` is a restricted test tool, not unrestricted shell
execution. No browser/computer-use claim is made.

Primary control is direct learned region selection with shared recurrence,
clustering and the identical halt mechanism. Uniform shared recurrence is also
tested as an intervention on fitted Bezier weights, not as a separately trained
arm. Other interventions remove memory route priors or permute selected regions.
The inference cache stores raw input embeddings, contextual embeddings and KV;
only scratch memory is constant with internal depth. Real text grows the context
up to 1,024 tokens. Gradient-training memory grows with unroll length. All sixteen
region outputs are computed before mixing, so there is no sparse-kernel speedup.

Development runs at 150 and 2,000 updates used an earlier uncontextualized input
prototype; the 2,100-update development run uses the final input and clustering
mechanism. Their weights are excluded from final comparisons. Validation showed
that token-format learning is much stronger than dependency following, so this
experiment is an initial mechanism/learning test, not a competitive coding-model
training run. The final model/data/optimizer settings are frozen before final
fits and test evaluation; there is no test-driven continuation or model selection.
