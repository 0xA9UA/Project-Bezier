# Revision 2: a fixed-budget shared reasoning and control core

The **entire central parameter region contains exactly 1,312 learned parameters**:
a 64-input, 16-hidden, 16-output feedforward network including every bias. Its
location is fixed at the origin, inside radius 0.25. Increasing the outer width
from 64 to 256 leaves the whole core unchanged. The large expert bank contains
shell experts only. There is no larger expert hidden underneath a small checker
slice.

Outer execution blends the two nearest region centers. The core's weights have
the fixed address `(0, 0, 0)`, but the outer model can select them even when its
curve is outside radius 0.25. The radius is not a hard execution gate. Path entry
into the inner sphere and selection of the core expert are measured separately.

The original implementation, frozen corpus, checkpoints and results remain in
`WORKSPACE\bezier_nested_core`. This revision has separate source,
reasoning data and output directories. It reuses that folder's verified isolated
Python 3.14.3 / PyTorch 2.10.0+cu128 environment on the RTX 5090. No environment
or previous checkpoint is modified by the revision.

## Run

```powershell
# Run all checks, then a new seven-arm, 128-update-per-arm pilot on seed 7.
& "WORKSPACE\bezier_nested_core_v2\scripts\run_experiment.ps1"

# Resume an interrupted pilot; identical sources and configuration are required.
& "WORKSPACE\bezier_nested_core_v2\scripts\run_experiment.ps1" -Resume

# Full Tier 1 protocol: seven arms, five seeds, 3,000 updates, uncapped evaluation.
& "WORKSPACE\bezier_nested_core_v2\scripts\run_experiment.ps1" -Full -Tier 1
```

The pilot evaluates 128 examples (eight complete program truth tables) per split.
It is a feasibility run, not proof of the hypothesis. Its 128 updates exercise
the snapshot refresh at update 100. Default training depths cycle through
3, 8, 16 and 32 expert visits. Tier 2 retains the same core and grows the outer
model; use `-Full -Tier 2` after reviewing Tier 1 evidence.

## One core, four uses

The same four complete parameter tensors are used in all shared arms:

1. **Outer computation:** when the route selects expert zero, the entire core
   transforms the activation. Fixed input/output mappings accommodate the outer
   width; they have no learned parameters.
2. **Initial route:** the core output sets all twelve cubic control coordinates
   through a fixed projection. There is no learned routing head outside the core.
3. **Mid-route control:** the core evaluates nearby target/strength pairs using
   live activation and geometry. Two nearby alternatives at two strengths give
   four checked redirects, plus Stay. The core chooses their ranking; fixed
   geometry routines enforce sphere containment, join continuity and fixed exit.
4. **Direct reasoning:** a parameter-free syntax adapter reads bits, opcodes,
   operand names and the queried variable. It never executes the program. Only
   the core and a fixed output projection produce the Boolean answer; the outer
   embedding, GRU and shell experts are absent from this computation.

No core rows or output heads are reserved exclusively for one role. Fixed dense
output mappings let the whole output vector contribute to each role. Tests and
audits inspect actual parameter-object identity and gradients from each role.
Elementwise gradient coverage is reported honestly: a sparse input can leave
some entries unchanged in a batch without implying a separate parameter block.

Zero added checker weights is measured against the revised clean outer model,
which already owns this same core. It does not imply zero extra computation,
buffer storage or runtime memory. The implementation pads the selected compact
core into the common expert batch shape; those temporary tensors contain no
additional learned parameters. Their cost is included in resource measurements.

## Joint training

Every arm receives the outer answer loss and the direct core reasoning loss.
Controlled arms additionally receive matched-budget gain regression and soft
ranking supervision over Stay and the four redirect proposals. The initial
weights and training examples are matched by seed. The core stays fixed in size
and location throughout training; all its parameters remain trainable.

For control supervision, the **current model** reaches a live post-region
activation with its autograd graph intact. A frozen teacher evaluates Stay and
Turn from detached copies of that exact state. In the pipelined schedule, the
unchanged lookahead visit is first performed by the live model; both teacher
continuations start from the same resulting application state. Each continuation
has identical remaining expert-call budget. Teacher labels are detached.

The current core's prediction retains the live activation graph. Control loss
therefore updates both the shared core and the outer representation that feeds
it. This directly addresses the previous version's detached-state limitation.
The additional detached-control arm removes precisely that gradient connection.
Hard routing choices themselves remain discrete; counterfactual supervision
provides the learning signal for route selection.

The scalar objective is outer answer cross entropy + core reasoning cross
entropy + counterfactual gain MSE + a soft ranking KL loss. Weights are explicit
in the tier configs. Small size is a hard architectural constraint; reasoning
and control quality are measured competence gates, not promises that training
will satisfy them. Joint gradients can conflict, so all roles are evaluated
separately as well as together.

## Reasoning tasks and leakage controls

`scripts/build_reasoning_corpus.py` creates executable Boolean programs using
AND, OR, XOR and NOT with four binary inputs. For example:

```text
a=1;b=0;c=1;d=0;e=a^b;f=e&c;f?
```

The answer is a separate ASCII `0` or `1` target. All 16 input assignments appear
for every program, and each selected program has eight answers of each class.
Two independent interpreters verify every label. The model's syntax adapter
does not use either interpreter or read the stored answer.

The corpus contains 128 training programs, 32 validation programs, 32 test
programs, and 32 longer-program tests. Both source programs **and their full
truth tables** are disjoint across splits. Thus a different spelling of a known
function cannot enter the held-out sets. Training, validation and ordinary test
programs have two to four instructions; the separate longer-program split has
five or six. Each instruction depends on its predecessor, avoiding dead-code
padding as a substitute for longer execution. Contexts fit within 64 bytes and
are padded, never truncated.

This is a constrained, checkable reasoning benchmark. Success would support
reasoning and control on these tasks, not establish general logical reasoning
or language-model capability. The first version's Python next-byte scores and
this version's Boolean answer scores are different tasks and are not directly
comparable.

## Controls and measurements

| Arm | Purpose | Additional learned checker parameters |
| --- | --- | ---: |
| clean-outer | No mid-route checker loss or exploration; core still learns reasoning and initial route | 0 |
| unshared-core | Entire controller copied into private weights, same delayed schedule | 1,312 |
| nested-core-serial | Shared core, immediate route checks | 0 |
| nested-core-concurrent | Shared core, delayed checks on a concurrent CUDA stream | 0 |
| ablation-geom-masked | Remove geometry from controller inputs | 0 |
| ablation-entry-state | Use static entry activation in controller inputs | 0 |
| ablation-control-detached | Block control gradients into the live outer representation | 0 |

The private arm is an explicit exception used as a comparison; the primary
architecture shares every core parameter. Core-only reasoning in the private
arm uses the private controller, and its additional parameter cost is reported.

Reports include full 256-way answer accuracy/NLL, restricted Boolean accuracy,
core-only accuracy/NLL, longer-program results, and local controller regret
against actual matched-budget continuations. Controller calibration currently
audits the first eligible decision; it does not claim calibration at every
later checkpoint. Validation alone selects route margins. Statistical intervals
resample whole program truth tables, pairing the same examples across arms.

Mechanism audits check the exact entire-core budget, no large central expert,
shared object identities, outer/core/control gradient participation, control
gradients into the encoder, and no encoder dependence in direct core reasoning.
Outer-only masking removes exactly the whole core expert contribution during
visits while leaving initial-path control intact. Degradation is measured, not
assumed. Continuous cubic-path core entry and exit drift are also recorded.

Concurrency tests compare the same delayed policy on one and two streams.
CUDA Graph capture supports the concurrent path; profiler traces establish
observed kernel overlap. Separate process benchmarks isolate weights, graph
pools and workspaces, including both same-checkpoint scheduling controls.
Training uses bf16 where supported; evaluation and timing use fp32. Geometry
is always fp32.

## Results and reproducibility

Read `runs/pilot/report.md` after the pilot completes. `suite.json` records the
full protocol and success gates; each arm stores `final.pt`, `train.jsonl`,
`audits.json`, per-program evaluation CSVs and selected trajectory plots.
Checkpoints include optimizer, teacher and RNG state plus source/config/corpus
hashes. Each run snapshots the exact Python source. Existing v1 checkpoints are
intentionally incompatible with this architecture.

The full study can pass only after routing improvements, core reasoning
competence, controller usefulness and sharing constraints all meet their stated
criteria. A completed pilot or successful invariant test alone is insufficient.
