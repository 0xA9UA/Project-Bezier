# V6: node-level spatial routing

**Implementation, checks, and the new pilot are complete.** Coordinates change hidden computation, with a fixed shared 15% core. Useful conditional steering beyond a fixed route remains unproven.

This experiment replaces the v5 whole-MLP address map with explicit three-dimensional coordinates for parameter-owning computational nodes. It adapts the existing Mistral model and keeps one shared GPU weight store. The central core retains the same approximately 15% parameter budget.

## Meaning of a node

A node is a computational unit that owns a disjoint set of existing parameters, not an individual scalar weight:

| Node kind | Owned parameters |
|---|---|
| Gated MLP neuron | One gate row, one up row, and one down-projection column |
| Query/output channel | One query-projection row and one output-projection column |
| Key channel | One key-projection row |
| Value channel | One value-projection row |
| Token embedding or output node | One vocabulary row |
| Normalization node | One learned scale scalar |

The coordinate atlas must account for every model parameter exactly once. Residual additions, attention softmax, rotary position arithmetic, and other parameter-free operations preserve their mathematical dependency structure. Token positions used by rotary attention are distinct from spatial node coordinates.

Coordinates are logical addresses mapped to tensor rows and columns; physical GPU memory does not become a sphere. See the [actual-coordinate figure](WORKSPACE/bezier_nested_llm_7b_v6/runs/summary/node_geometry.png).

## Execution contract

Hidden computation is selected by distance from each node to 33 samples in the forward window `t ∈ [0.125, 1]` of a cubic Bézier path, under fixed per-stage node budgets. Excluding the shared current point makes future direction changes distinguishable. Selected projection rows and columns execute through the shared tensors. Attention uses scatter operations to maintain its required tensor dimensions. Geometry therefore changes which hidden computations execute, rather than merely annotating whole transformer layers.

The engine preserves the pretrained layer dependency order. It does not permit arbitrary execution of incompatible nodes in any order. The core runs its fixed direct path through shared central weights; it scores possible redirects using the current activation and route geometry.

Use `SpatialController.route_forward(...)` or [applications/generate.py](WORKSPACE/bezier_nested_llm_7b_v6/applications/generate.py) for the node engine. The inherited generic `model(...)` call is the legacy dense entry point and does not activate this spatial architecture. The application calls the explicit node route for its `learned` and `stay` modes.

Input-token embedding rows and all output-vocabulary rows are mandatory interfaces. Their coordinates determine execution ordering, not eligibility to remove required input tokens or vocabulary choices. Consequently, "every node has coordinates" does not mean every kind of node can be omitted on every inference.

## Fixed core and memory

The core sphere is centered at `(0, 0, 0)` with radius `0.25`. Coordinates must classify precisely the shared parameter slices used by the direct core execution and its optimizer. The core budget is a parameter count, not a node-count fraction or a geometric volume fraction.

Sharing removes the need for an independent persistent core weight copy. Coordinates, indices, selected-weight gathers, activations, gradients, and optimizer state still consume memory. Loading the full model on GPU still requires storage for all full-model weights.

## Verification and pilot

The implementation passes 33 CPU tests and the selected-checkpoint GPU runtime audit. There are **987,136 unique-coordinate nodes**, owning **7,248,023,552 parameters** exactly once. The central **147,185 nodes** own **1,087,201,280 parameters**, or **14.999968918%** of the model.

Swapping selected/unselected node coordinates changes both dispatch and final logits; restoring coordinates restores logits bitwise. Full-density inference matches the dense model's greedy predictions within measured BF16 arithmetic differences. Shared GPU storage and zero additional CUDA allocation for a second core execution view are verified.

The corrected pilot completed 128 reasoning and 32 replay updates, peaking at 19.7642 GiB allocated GPU memory. All outside-core weights remain bitwise unchanged; all 48 core-bearing tensors changed.

| Held-out accuracy | Test | Longer-program OOD |
|---|---:|---:|
| Same-weight dense model | 64.06% | 57.03% |
| Spatial Stay | 59.38% | 57.03% |
| Learned steering | **66.41%** | **59.38%** |
| Validation-best fixed route | 66.02% | 59.38% |

Steering improves test accuracy over spatial Stay, but it does not establish an advantage over fixed or shuffled control. The first decision always chooses the same direction; the learned policy differs from the best fixed route's predictions on only three test examples and zero OOD examples. The point-estimate gains over the initial dense warm-start also have confidence intervals crossing zero.

The selected-checkpoint short runtime check took 23.24 ms dense, 97.24 ms spatial Stay, and 119.92 ms with control, at a 13.7171-GiB aggregate peak allocation. Full held-out deployed verification peaked at 13.9125 GiB. This implementation does not demonstrate a speed benefit or a lower total VRAM footprint than a single dense model.

Full deployed verification preserved every action but found output differences from the cached counterfactual tree: two learned test predictions and one shuffled test prediction differed. BF16 batch grouping is a plausible explanation, not an isolated cause. Final results use actual deployed records; exact cached-output/NLL equivalence did not pass. The generation smoke ran successfully but answered the simple arithmetic prompt incorrectly. See the [report](WORKSPACE/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md) for exact comparisons, uncertainty, reproduction commands, and scope.
