# Prospective protocol

## Scientific question

At the same stored parameter and matrix-computation budgets, does adaptive
Bézier routing through a shared processing bank improve verified coding success
relative to ordinary recurrence and learned non-spatial routing? Do leaving and
revisiting regions contribute to success? A positive route intervention proves
that routing influences computation, not that it improves task quality.

## Architecture and controls

Every primary arm uses the same frozen BF16 1.5B base, rank-eight Q/V LoRA tensors,
128-feature latent state, 64-feature shared input transform, sixteen 128-output
nonlinear regions, and a 130 -> 64 -> 48 controller. A common 16 -> 128 state
update consumes each controller slice. All learned tensors receive gradients.
The same weights are used at every pass and each of its three field samples.

A Bézier segment starts at the previous endpoint; its three predicted control
points are bounded to the unit ball. The true sequential sample points are
gamma(1/3), gamma(2/3), gamma(1). The state changes between samples. Four nearest
regions are selected with normalized distance-based weights. Segment control
points are replanned from the evolving state after each full language-model pass.
The first point is derived from the prompt state. Turning and revisiting are
permitted, never forced or rewarded. Region coordinates are fixed buffers.

The non-geometric controller instead outputs three sets of sixteen logits;
each set chooses four regions. Ordinary recurrence uses the arithmetic mean of
the same sixteen nonlinear outputs, plus the same controller-derived state
update. The region nonlinearities make their separately stored parameters
functional even under uniform mixing.

All arms evaluate all sixteen regions with the same dense matrix operations.
This deliberately isolates selection from sparse kernel dispatch. The budget
audit reports exactly equal parameter counts, analytic matrix MACs and measured
profiler matrix FLOPs. The profiler also records its total supported FLOPs:
small elementwise router differences are explicitly retained. Unsupported
operations, wall time and GPU memory are not inferred from the FLOP counter.

## Data and fitting

Reuse the earlier 96 MBPP training and 12 validation tasks and verified local
model assets. Select 24 additional compatible MBPP test tasks with a frozen
shuffle, excluding all earlier test IDs. Selection checks only reference
compatibility, reference length and prompt limits, never model success.
Reuse the four authored cumulative projects, which remain absent from fitting.
Their three stages are dependent observations; project-level confidence
calculations must resample whole projects.

Each seed starts every arm from the same previously trained zero-step protocol
LoRA adapter. This common preparation was 160 updates in the earlier experiment.
Train each new arm for 192 additional updates with identical task/order and
4/8/16-pass schedules within seed. AdamW, learning rate 1e-4, clipping at one,
and validation at 96 and 192 updates are fixed. Select lowest action NLL on
separate validation examples averaged over all pass counts. No routing auxiliary
reward, test feedback or hidden examples influence fitting.

## Evaluation and ablation

Greedy JSON tool actions may inspect declared files, replace solution.py, run
public tests and finish. Each stage permits six actions, at most 512 tokens per
action and 2,048 generated tokens total. Hidden tests evaluate the irrevocable
final program only. Project code carries forward. Realized work varies with
output length and stopping, so report actual tokens and forward counters as well
as identical budget ceilings. This is restricted terminal-style pure-Python
coding, not unrestricted shell execution or a general repository benchmark.

Primary budget: eight latent passes. Four and sixteen are prespecified secondary
budgets. Fit seeds: 7 and 19. Main grid: 18 conditions. The zero-step protocol
adapters are two additional context controls and are not compute-matched.

For each adaptive arm and seed, a post-training eight-pass ablation masks any
region used before but absent from the immediately preceding active set. This
permits staying, but forbids a leave-and-return visit. The current active set
always supplies four legal choices. It preserves parameters and matrix work,
while changing the learned policy distribution. Four additional conditions give
24 total. It is an intervention on a fitted policy, not a retrained architecture.

## Evidence and reporting

Record complete actions, public feedback, final code, hidden scores, region IDs,
mixture weights, actual points, return events, generated tokens, model passes,
processed token positions, an attention-cell upper bound, elapsed time and peak
CUDA allocation. Count a region return only after it has been absent for at
least one sample; consecutive selection is staying, not revisiting. Also report
returns of the single highest-weight region and count collapses/stationary paths.

Measure batch-one fixed-prompt latency with exactly 32 output tokens, three
shuffled repeats per pass count after warmup. Include routing overhead. Quality
evaluation batches four agents; amortized batch time is not individual latency.
Report paired bootstrap uncertainty over 28 task families (24 functions plus
four whole projects), averaged over seeds, and raw seed results. No test-picked
winning budget should be presented as preselected.

Independently re-score saved programs, check checkpoint/source/data hashes,
recompute route return statistics, confirm the no-return intervention and
verify matched parameters and matrix budgets. Store failure examples and all
conditions, including negative results.
