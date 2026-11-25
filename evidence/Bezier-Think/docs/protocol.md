<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../../PUBLICATION_SCOPE.md) · [Reader guide](../../../READ_THIS_FIRST.md).

# Bézier Think: variable internal computation for a local coding agent

## Scope

Run an actual instruction-tuned language model on terminal-style coding tasks.
No screenshot or GUI control is included, per the user's clarification.
The experiment tests 2, 4, 8, 16 and 32 internal latent steps for each of Bézier
routing, straight routing and ordinary shared recurrence. These are five step
counts, not five experiments: there are fifteen settings per fitting seed.

Small base: Qwen/Qwen2.5-Coder-1.5B-Instruct, pinned commit
2e1fd397ee46e1388853d2af2c993145b0f1098a, 1,543,714,304 original parameters.
Large control: Qwen/Qwen2.5-Coder-14B-Instruct, pinned commit
aedcc2d42b622764e023cf882b6652e646b95671, 14,770,033,664 original parameters.
Both use BF16 weights, the same tokenizer family, SDPA and the same tool protocol.
All computation and stored artifacts stay in this project on the local computer.

## Internal processing

The prompt is encoded once. A shared latent field transforms the last hidden
state; the resulting continuous embedding is fed back through the same language
model, updating its KV cache and producing the next hidden state. This repeats
for the requested internal count before generating a tool action. Latent steps
are not textual reasoning tokens or terminal actions. They cost actual model
forwards and their small additional KV entries are counted in memory usage.

Bézier and straight variants use the same carried state, sixteen spatial groups,
four selected groups and shared processing weights. Bézier control points depend
on the evolving activation; the straight control follows a straight segment.
Ordinary recurrence uses a dense shared update without spatial routing. Parameter
counts, active operations and routing overhead are reported, not assumed equal.
Field updates are not divided by the chosen count: increasing it means more
recurrent processing rather than merely refining the same integration interval.

Fit small adapters and rank-eight attention LoRA weights, retaining the pretrained
base. Each of the three recurrent variants trains across all five internal counts.
Two fitting seeds (7, 19) give thirty recurrent evaluation conditions. Include the
unmodified small model, the large model, and a separately trained zero-recurrence
control for each seed: thirty-four total conditions. Selection uses validation
data only. Do not choose a winning budget using test results and present it as a
preselected result.

## Agent evaluation

Use frozen disjoint training, validation and test tasks. The public MBPP source
is pinned at 4bb6404fdc6cacfda99d4ac4205087b89d32030c. Repository fixtures provide
specifications, editable Python modules and public tests. The agent can inspect
files, write an allowed module, run public tests and revise. Hidden tests never
enter prompts, stopping rules, checkpoints, training rewards or candidate choice.
Include authored cumulative coding projects in addition to function-repair tasks.
Report individual stages and complete projects separately. A small restricted
Python benchmark does not establish SWE-bench or general autonomous-agent parity.

Final dataset IDs, fixture hashes, optimizer settings, generation limits and
task counts will be frozen in runs/manifest.json before fitting or test scoring.
Development checks use explicitly separate fixtures. Do not change scientific
settings after inspecting final outcomes; document implementation corrections.

## Measurements and controls

Report task completion, public and hidden test success, invalid actions, repairs,
generated tokens, internal forwards, end-to-end agent time, model inference time,
peak CUDA allocated/reserved memory, persistent weights, and on-disk bytes.
Count failed attempts and tool execution. Distinguish initialization from request
latency and distinguish batched throughput from a single agent's latency.

Use the same outer tool/token limits for every model. Include success versus
elapsed-time curves and controlled single-request timing at each internal count.
GPU measurements run without an overlapping training/model job. Record detected
GPU activity and use interleaved repeated timing checks. Report paired task
uncertainty and seed variation; two seeds cannot establish a general scaling law.

Validate source and data hashes, latent-step counts, gradients, saved checkpoint
reloads, route geometry, action confinement and resource-limited code execution.
Keep all earlier project models and results intact.

## Research basis

- Qwen2.5-Coder model family: https://arxiv.org/abs/2409.12186
- Continuous latent reasoning: https://arxiv.org/abs/2412.06769
- Recurrent depth: https://arxiv.org/abs/2502.05171

This is a new local adaptation, not a reproduction of those papers' results.
