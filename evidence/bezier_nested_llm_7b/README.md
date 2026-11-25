# Shared-core 7B pretrained experiment

This experiment adapts **Mistral-7B-Instruct-v0.3** to a fixed central core containing **15% of the original model's parameters**. The first use case is selecting which of two small Python functions satisfies a supplied contract. Every label can be checked over the complete declared input domain without executing generated code.

## Parameter budget

- Outer model: **7,248,023,552** learned parameters.
- Core: **1,087,201,280** shared parameters (**14.9999689184%**).
- Additional learned router, adapter, embedding, or output weights: **zero**.
- Logical central address: `(0, 0, 0)`, fixed radius `0.25` inside the unit sphere. Spatial radius and fraction of parameters are separate constraints; parameter density need not be uniform.

The core contains the original token embedding, language head, final norm, the first four attention modules and their norms, three complete feed-forward modules, and 9,967 neurons from the fourth feed-forward module. The outer model uses these exact same tensors. The partial module uses views, with no copy. All interfaces needed for independent core inference count against its budget. `parameter_layout.json` and each run's parameter audit list the tensors and slices.

The base is a standard decoder Transformer, pinned to revision `c170c708c41dac9275d15a8fff4eca08d52bab71`. `base_manifest.json` records downloaded file checksums. Original files under `models/base` are immutable experiment inputs. Native architecture code is used; remote code is disabled.

## Scope of this first pretrained adaptation

Attention layers and their residual order stay intact. At one decision point the core receives a live outer activation and the candidate Bézier geometry. It estimates the conditional A/B log-probability advantage of one nearby feed-forward region over the scheduled region. A margin gates the choice; the chosen curve endpoint resolves the executed region through nearest-center lookup. A tangent-continuous tail returns to the common outer exit. Only the chosen continuation runs at inference. Training tries both continuations from the **same saved activation**, with identical remaining layer and feed-forward call budgets, before any weight update.

The control path processes two tokens: the shared BOS embedding plus fixed geometry features, and the normalized live activation. This allows its own attention query/key weights to learn. A centered fixed readout uses the shared language head and bounds the gain estimate to ±0.1 nats. Training clips only the fitting target to that range; measured true gains and regret remain unclipped. There is no additional learned readout. The geometric coordinates are logical addresses, not physical locations in GPU memory.

This is a **single feed-forward substitution experiment**, not yet an unrestricted spatial Transformer. It does not implement repeated rerouting, arbitrary attention traversal, dynamic core entry/re-entry, or a streaming generation cache. The standard outer prefix already includes the shared core layers. Core and one outer lookahead layer can execute on separate CUDA streams during inference; that is not a measured speedup claim. Training uses a deterministic serial graph.

The eager stream implementation showed no observed kernel overlap in the measured trace. Optional independent CUDA graph replay did demonstrate overlap while matching logits exactly; see `runs/graph_overlap_check_v2/graph_overlap.json`. This inference helper adds fixed-shape buffers and setup cost, but no learned weights. The measured sample does not establish a speedup.

The two candidate regions are a preset pair of neighboring layers, not a global search. Turning substitutes the next layer's MLP at the application point; that MLP is still used again at its ordinary later position. The two routes have equal MLP-call counts. At each branch the attention layers remain in their original order.

The core also independently predicts A/B through its direct four-layer path. Extracting those layers from a pretrained 32-layer model does **not** make the core a pretrained reasoning specialist. Its task accuracy and its ability to control the outer model must be measured separately.

## Verifiable task and splits

The bounded patch-review corpus has 256 training, 32 validation, 64 test and 64 longer-chain OOD programs, each with two swapped candidate presentations. Both A/B labels are balanced within each program. Correct and incorrect candidate truth functions are disjoint across splits. An independent contract interpreter and an AST whitelist interpreter verify all 100 pairs of input integers from 0 through 9. There is no Python `eval` or `exec`.

Training and ID examples contain 2–4 stages; OOD contains 5–6. Prompts have at most 294 tokens with the official tokenizer/chat template. The context limit is 384 and truncation is forbidden. Candidate length is balanced. This is a test of bounded contract-to-code consistency, not a general code-repair or deep-reasoning benchmark.

Evaluation distinguishes forced A/B accuracy, conditional A/B NLL, full-vocabulary answer NLL, unrestricted greedy answer validity, and both-swaps-correct consistency. Routing thresholds are chosen only on validation. Confidence intervals resample complete programs, keeping the swapped examples together. Test and OOD results are not used to revise training settings.

## Run locally

The checked environment is Windows, Python 3.14, PyTorch 2.10.0+cu128, Transformers 4.57.1, and an RTX 5090 with 32 GB. The existing Python executable is:

```powershell
$python = 'WORKSPACE\Bezier-Think\.venv\Scripts\python.exe'
Set-Location 'WORKSPACE\bezier_nested_llm_7b'
& $python -m pytest tests -q
& $python scripts/run_training.py --output runs/new-run --config configs/pilot.json
& $python scripts/evaluate_run.py --model runs/new-run/checkpoint_u000512 --output runs/new-evaluation --shared
```

Read `docs/use_case.md` for the single-example CLI. Training refuses to overwrite existing run logs. The final checkpoint is a standard Hugging Face model plus wrapper configuration, optimizer/RNG state, source snapshots, and a completion/checksum marker. A separate base-model process may load the trained standard weights without routing, which is useful for the steering ablation.

To continue a run, copy the pilot config and increase only `joint_updates` to the new total, then pass `--resume-from` with the completed checkpoint and keep the original run output directory. The trainer restores optimizer state, random generators and the batch sampler, checks source/corpus identity, and preserves earlier checkpoints. Include `--baseline` when continuing the baseline arm.

## Training and resource accounting

Full original weights are trainable; this is not adapter tuning or from-scratch pretraining. Joint loss combines outer answer prediction, independent core answer prediction, and supervised continuation-gain estimation. A factored Adafactor optimizer consumes a combined backward once per update, updates a leaf only after its shared uses accumulate, then releases its gradient. BF16 stochastic rounding avoids silently discarding small updates. Gradient accumulation and activation checkpoint recomputation are forbidden with these hooks.

The frozen pilot uses 512 joint updates per arm, batch size four (two complete swapped program pairs), learning rate 2e-6, 32 warmup updates, core reasoning weight 0.25 and control weight 0.1. There is no isolated core warmup: shared-core learning is paired with the outer prediction objective from the first update. Training cycles decision layers 8, 16 and 24 so geometry is not constant. The primary evaluation uses decision layer 8. Each arm sees 2,048 sampled examples with replacement; the shared arm also generates 2,048 matched counterfactual labels.

Shared weights save model storage, **not computation**. Core inference, counterfactual training, activations, optimizer state, and checkpoints have additional costs. Baseline and steered training arms match initialization, examples and update budget; the steered arm spends additional compute on its checker and counterfactual targets. Equal-budget claims concern the two alternative continuations, not total training cost between arms.

The initial discarded preflight is preserved in `runs/preflight`. Final pilot outcomes and limitations are recorded in `runs/pilot_report.json` and `EXPERIMENT_REPORT.md` after evaluation.

## Sources

- [Official Mistral model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [PyTorch optimizer-in-backward tutorial](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)

The optimizer adapts the existing local `Bezier-MemorySphere-7B-Training/sphere_train/optim.py` implementation. No earlier from-scratch 7B model is used as pretrained initialization.
