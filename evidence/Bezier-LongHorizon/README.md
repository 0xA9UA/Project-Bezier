# GPT-OSS-120B: Bézier compression and cumulative coding pilot

This directory contains an actual local conversion of the pinned
`openai/gpt-oss-120b` checkpoint and a compact, feedback-controlled replacement
for one of its mixture-of-experts blocks. All training and evaluation ran on
this PC. All four coding evaluations and four six-stage workflow runs are
complete. See [RESULTS.md](RESULTS.md) for the numerical report and charts;
completion receipts live in `runs/suite/`.

The three-bit artifact is 21.95% smaller than the native deployment. Replacing
one expert bank brings the reduction to 23.91%. The compact model before RL
matched or exceeded the native model's scores in this small pilot; controller
RL worsened them. The experiment does not establish a general long-horizon
or curvature advantage.

The experiment separates three questions: whether the Bézier codebook reduces
storage, whether a small spherical processing field can replace a large expert
bank, and whether reinforcement learning helps that replacement on coding tasks.
These are different claims. A smaller checkpoint alone does not establish better
reasoning or better performance on long tasks.

## Artifacts

| Directory | Contents |
|---|---|
| `source/gpt-oss-120b` | Original native MXFP4 checkpoint and tokenizer |
| `artifacts/bezier3` | All 36 layers, with every expert matrix requantized to three-bit Bézier codes |
| `artifacts/bezier3-distilled` | Three-bit model with layer 18's expert bank replaced by a fitted spherical field |
| `artifacts/bezier3-rl` | Same compact architecture, after eight completed controller REINFORCE episodes |
| `adaptation/{bezier,straight,frozen}` | Local fitting controls and their training records |
| `adaptation/rl` | Actual RL episodes, intermediate controller checkpoint, final training receipt |
| `runs` | Raw generations, public/hidden checks, protocols, timings, hashes, and completion receipts |

The compressed artifacts require this project's loader. They are not drop-in
Transformers, vLLM, or Ollama checkpoints. Unchanged artifact files use NTFS
hardlinks to avoid duplicating approximately 47 GiB per export. Their recorded
sizes are logical deployment sizes. Each export has its own index, manifest,
and replacement weights; model loading requires no path to its parent artifact.
Treat hardlinked weight files as immutable.

## What was changed

The original expert format already uses 4.25 bits per weight including scales.
The previous four-bit Bézier method also costs 4.25 bits per weight, so its
earlier savings against dense models cannot be repeated against this source.
The three-bit extension uses three-bit codes plus a FP16 scale per 64 weights,
or 3.25 bits per weight before small codebook and file overheads. Four cubic
controls define each matrix's codebook; every weight still has its own code.
All non-expert tensors remain bitwise unchanged in `bezier3`.
The weight-screen uniform control uses conventional symmetric quantization,
with one unused code. Its bit budget is matched, but it does not isolate a
curve-specific advantage from nonuniform level placement and code utilization.

The compact variant physically removes all 128 experts, their biases, and the
router at zero-based layer 18. A 3,098,057-parameter field replaces that MLP.
It has a 512-dimensional state, 16 spatial groups, four selected groups per
step, and three internal processing steps. Its controller proposes a cubic
Bézier segment from the current state and position. The actual midpoint selects
the processing groups; subsequent segments use the updated state. Recorded
curves trace the segment actually traversed, using de Casteljau subdivision.
Attention layers, their order, and KV-cache semantics remain intact.
The field state resets for each token. It does not add persistent cross-token
memory beyond the original attention cache.

Supervised distillation fits the compact processing weights and controller to
the three-bit parent's activations. Reinforcement learning subsequently updates
only the controller's weights from executable coding rewards. It does not
fine-tune the full 117-billion-parameter source or jointly retrain all field
weights. Physical block replacement causes the extra storage reduction; RL
changes behavior within the same compact storage budget.

## Local execution

The tested environment reuses `../Bezier-Compress/.venv/Scripts/python.exe`:
Python 3.14, PyTorch 2.10.0+cu128, Transformers 4.57.1, safetensors 0.6.2.
Exact versions and hardware are in `runs/preflight.json`. No shared environment
packages were upgraded. The GPU is an RTX 5090 with approximately 32 GiB VRAM;
the PC has approximately 32 GiB system RAM.

The loader reads expert files from SSD and holds a bounded GPU cache. The
verified CUDA path caches packed experts and decodes only the current expert;
the original reference path caches expanded BF16 experts. The full checkpoint
never has to fit in RAM or VRAM.
This is a correctness-oriented implementation; its Python dispatch, decoding,
and SSD traffic do not represent optimized serving performance.

During evaluation, a CUDA decoder was added using the NVRTC already bundled
with PyTorch. It reproduced 298,598,400 real decoded weight values bitwise,
plus every finite native exponent/nibble combination. The original native and
three-bit coding runs had already started with the reference decoder. Later
loads cache compressed bytes, use the verified kernel, and check full-model token equivalence against
a saved reference generation before measuring tasks. The GPU expert cache is
then cleared. Smoke inputs are frozen to their original date; successful checks
are reused only when model, runtime, reference, and environment fingerprints match.
This runtime change further rules out treating wall times as a
controlled architectural speed comparison. `BEZIER_REFERENCE_DECODE=1` selects
the reference implementation. The pre-change loader is preserved at
`runs/reference_decoder_model.py`; verification is in
`runs/fused_decoder_verification.json` and `runs/fused_model_equivalence_*.json`.

The CUDA implementation follows NVIDIA's [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
and [kernel-launch API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html).

From this directory, in PowerShell:

```powershell
$python = '..\Bezier-Compress\.venv\Scripts\python.exe'
& $python -m bezier_longhorizon.smoke --artifact artifacts/bezier3 --cache-gib 12 --tokens 32 --output runs/manual_smoke.json
& $python -m bezier_longhorizon.smoke --artifact artifacts/bezier3-rl --cache-gib 12 --tokens 32 --output runs/manual_rl_smoke.json
& $python -m bezier_longhorizon.status
```

These commands reuse the completed local files and do not access the network.
Choose a cache budget that leaves space for the approximately 4 GiB common
tensors, attention activations, temporary decode buffers, and other GPU users.
The matched evaluation queue uses 20 GiB; early smoke tests used 2 or 12 GiB
and are not valid speed comparisons with one another.

The source revision is `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
It was downloaded with `hf download` into this project. Only the selected
Transformers checkpoint, tokenizer, configuration, license, and usage policy
are needed. The original download is preserved.

## Reproduction and result protection

```powershell
& $python -m pytest tests -q
& $python -m bezier_longhorizon.preflight
& $python -m bezier_longhorizon.codec_pilot
& $python -m bezier_longhorizon.verify --artifact artifacts/bezier3 --output runs/bezier3_verification.json
& $python -m bezier_longhorizon.benchmark evaluate --artifact native --name native --cache-gib 20
& $python -m bezier_longhorizon.benchmark evaluate --artifact artifacts/bezier3 --name bezier3 --cache-gib 20
& $python -m bezier_longhorizon.benchmark evaluate --artifact artifacts/bezier3-distilled --name compact_distilled --cache-gib 20
& $python -m bezier_longhorizon.benchmark evaluate --artifact artifacts/bezier3-rl --name compact_rl --cache-gib 20
& $python -m bezier_longhorizon.workflow --artifact native --name native --cache-gib 20
& $python -m bezier_longhorizon.workflow --artifact artifacts/bezier3 --name bezier3 --cache-gib 20
& $python -m bezier_longhorizon.workflow --artifact artifacts/bezier3-rl --name compact_rl --cache-gib 20
& $python -m bezier_longhorizon.workflow --artifact artifacts/bezier3-distilled --name compact_distilled --cache-gib 20
```

Evaluation commands resume completed task records. A coding run refuses to mix
different model manifests or budgets under the same name. The suite stops on
any failing command and records its exit code. An existing completed RL result
or export is preserved rather than overwritten. Conversion provenance records
the codec snapshot and decoder hash; `runs/conversion_source.py` preserves the
decoder source used during conversion, before a later file-presence check fix.

For a fresh experimental reproduction, use a separate directory. The original
sequence is `preflight`, `codec_pilot`, `convert`, `capture`, `distill`,
`benchmark freeze`, `benchmark rl --field adaptation/bezier`, `export`, then
the evaluations above. Read each command's `--help` and the frozen protocols
before launching another expensive run. Do not reuse held-out results to tune
the model or protocol and then call them an untouched test set.

## Evaluation boundaries

`runs/coding_protocol.json` freezes four training tasks, two local validation
tasks, and four held-out coding tasks. The held-out pilot allows up to four
attempts of 384 generated tokens each, with public test feedback. Hidden tests
never affect candidate selection, stopping, or model feedback. Training rewards
use the separate training tasks' tests. RL uses eight episodes, at most two
attempts of 256 tokens each, and a small token-cost penalty.

`runs/workflow_protocol.json` freezes one six-stage inventory-program project:
basic stock, atomic transfers, nested transactions, snapshots, reservations,
and atomic batches. Each stage adds requirements while retaining cumulative
tests. The model receives its latest valid code, the cumulative contract, and
public feedback. This tests maintaining an evolving program with condensed
state, not remembering an entire chat or operating autonomously for hours.
A fourth, pre-RL compact workflow control is recorded in
`runs/workflow_control_protocol.json`. It was registered before any workflow
stage outcomes, after coding results began, and runs after the primary queue
with the same requirements, grader, and token budgets. This separates block
replacement effects from controller RL effects.

Both pilots use greedy decoding and low reasoning effort. Only one fitting/RL
seed is run. Four tasks and one workflow are too few for general capability or
statistical superiority claims. Straight-path and frozen-feedback controls are
fitted for local reconstruction; they have not received a matched full coding
evaluation, so coding results cannot isolate curvature as the cause.

This local adaptation uses a fixed three-step field inside one MLP. It does not
test remapping the entire transformer into one global sphere, learned halting,
or unrestricted inter-layer recurrence. Those are separate architectural
experiments that these results cannot settle.

Generated programs run in a separate Python process with a three-second timeout,
a 512 MiB Windows Job Object limit, restricted syntax, and limited built-ins.
This grader is designed for these pure-Python fixtures. It is not a sandbox for
arbitrary packages or repository-agent execution.

## Sources

- [Official model card](https://huggingface.co/openai/gpt-oss-120b)
- [Pinned source configuration](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json)
- [OpenAI reference implementation](https://github.com/openai/gpt-oss)
- [Transformers GPT-OSS documentation](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_oss)
- Local predecessor: `../Bezier-Granite/granite_bezier/codec.py`, snapshotted
  here with SHA-256 `e7e0cc1c5c1f5716c91b4d53fdf92052ee204add6245d6022054dfb49db75199`.

Source model license and usage policy are retained in the deployment artifacts.
