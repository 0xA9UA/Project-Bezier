# Bezier-Compress

A separate project testing whether compact Bézier representations can compress a pretrained language model while retaining useful behavior. All new source, downloaded models, datasets, codec artifacts, reports and caches live here. The project uses its own Python environment and reuses the parent project's installed CUDA PyTorch runtime read-only; new packages do not modify the original experiment.

The experiment compares real pretrained SmolLM2 models at 135 million and 1.7 billion advertised parameters. It includes direct cubic fits, sorted cubic fits with the permutation explicitly stored, a cubic-controlled quantization codebook, ordinary quantization, an unconstrained learned codebook, and low-rank approximation.

This is initially a model-storage experiment: compressed weights are reconstructed to the same dense float16 execution format for a controlled quality comparison. It does not claim compressed-kernel inference or reduced steady-state GPU memory.

## What the tests show

The tested Bézier representations can compress a pretrained model, but the experiments do not establish unusually high knowledge density or improved reliability. Directly replacing blocks of ordinary pretrained weights with four cubic coefficients severely damages behavior. A curve-controlled quantization codebook works much better, while retaining one four-bit index for every weight and one scale for every group of 64.

The fixed primary study completed **20 comparisons across SmolLM2-135M and SmolLM2-1.7B** on this computer's **RTX 5090**. Each case scores 65,536 held-out next tokens from WikiText-2 and the same 1,024 ARC-Easy science questions, plus a separate validation subset. Only ordinary int8 meets all three preservation targets on validation and test for both model sizes. The targets were set before outcomes: perplexity increase ≤10%, QA accuracy loss ≤2 percentage points, and retention of ≥95% of the reference model's correct answers.

Selected **1.7B primary-study** results (all complete artifact bytes counted):

| Representation | Artifact MiB | Reduction | Test perplexity ↓ | Science QA accuracy ↑ | Reference-correct retained ↑ |
|---|---:|---:|---:|---:|---:|
| FP16 reference | 3,267.52 | 1.00× | 10.706 | 72.95% | 100.00% |
| Uniform int8 | 1,779.56 | 1.84× | 10.706 | 73.05% | 99.46% |
| Uniform int4 | 1,011.56 | 3.23× | 12.753 | 71.48% | 92.64% |
| Cubic Bézier codebook | 1,011.59 | 3.23× | 16.200 | 63.18% | 81.93% |
| Free scalar codebook | 1,011.59 | 3.23× | 17.468 | 62.50% | 80.72% |

The primary cubic codebook reconstructs weights with lower relative squared error than uniform int4, yet performs worse on both tasks. Small weight error alone is not a measure of preserved knowledge. The slight int8 accuracy increase is within the paired sampling interval and is not evidence of improved reliability.

## Exploratory refinement

After seeing the primary results, we specified a separate **seven-case follow-up** anchoring a quantization level at zero. This uses another 1,024 questions and 65,536 scored text tokens, with no primary-test question IDs or text positions reused. It is a fresh subset of the same datasets, not an independent corpus. The two new fits still use weights only, with no recovery training.

Selected results on that **different follow-up subset**:

| Representation | Artifact MiB | Test perplexity ↓ | Science QA accuracy ↑ |
|---|---:|---:|---:|
| FP16 reference | 3,267.52 | 10.720 | 74.12% |
| Uniform int8 | 1,779.56 | 10.720 | 74.02% |
| Uniform int4 | 1,011.56 | 12.779 | 70.31% |
| Original Bézier codebook | 1,011.59 | 16.551 | 62.30% |
| Zero-anchored Bézier codebook | 1,011.61 | 12.271 | 70.31% |
| Zero-anchored free codebook | 1,011.62 | 12.313 | 71.00% |

The refinement improves the Bézier variant substantially and achieves slightly lower perplexity than uniform int4 at effectively the same storage and QA accuracy. It still has about **14.5% higher perplexity** and **3.81 percentage points lower QA accuracy** than FP16, exceeding the preservation limits. It retains **92.09%** of the reference's correct answers. The free zero-anchored codebook gives similar language quality and slightly higher measured QA accuracy, so the improvement is not unique to a cubic parameterization. All seven comparisons and paired intervals are included in the follow-up report.

The refined curve stores four float32 controls per matrix to reduce zero-anchor rounding; the original stores float16 controls. Its cubic constraint and control precision change together. These results do not isolate zero anchoring as the sole cause, demonstrate a sphere-routing benefit, or establish a scaling law for much larger models.

## Open the evidence

- [Primary report: all 20 comparisons](runs/main/report.html) · [text version](runs/main/report.md) · [machine-readable results and intervals](runs/main/summary.json)
- [Exploratory report: all seven comparisons](runs/anchor_followup/report.html) · [text version](runs/anchor_followup/report.md) · [machine-readable results and intervals](runs/anchor_followup/summary.json)
- [Primary protocol](docs/protocol.md) · [exact codec mathematics and byte accounting](docs/codecs.md) · [follow-up protocol](docs/anchor_followup.md)
- [Reproduction and generation commands](docs/usage.md) · [pinned asset provenance](assets_manifest.json)
- [Primary artifact verification](runs/main/verification.json) · [follow-up verification](runs/anchor_followup/verification.json)
- [Completion checks](runs/completion_checks.json) · [actual compressed-model generation output](runs/demo_zero_anchor.txt)

Validation completed: **23 unit checks passed**, all **22 distinct complete artifacts** were independently reloaded without source checkpoint weights, and sampled language predictions and QA scores reproduced the original evaluations. Source/input hashes, payload hashes, byte totals, follow-up nonoverlap and report links were checked. The generation command below was also executed successfully.

All source, original checkpoints, compressed artifacts, fixed evaluation inputs, per-example predictions, reports, manifests and caches are inside this project. A self-contained numerical artifact includes preserved weights, packed codes, scales, coefficients, metadata, model config and tokenizer. Shared Python/CUDA/library binaries are excluded from model file size, consistently for every method.

## Use a compressed model

From this project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m bezier_compress.generate --artifact runs\anchor_followup\bezier_zero_anchor4\artifact --prompt "The capital of France is" --max-new-tokens 32
```

This loads only the compressed artifact, reconstructs the original dense FP16 model and generates a completion. SmolLM2 here is a base model, not an instruction-tuned assistant. **The tested execution path does not reduce steady-state GPU parameter memory or establish faster inference.** A codec-aware kernel would be a separate implementation and experiment.

For a complete rerun of the primary study into a new directory:

```powershell
.\scripts\run_study.ps1 -Output runs\repeat
```

`scripts/run_followup.ps1` runs or resumes the exploratory study using the original primary artifacts. Scientific source and evaluation inputs are hash-frozen; changing them requires a new experiment, not silently resuming the old one. Reports and reconstruction checks are separate from that frozen fitting/evaluation code.

## Scope

This is a post-training weight-compression proof of concept on one model family at two sizes. ARC-Easy includes factual knowledge, reasoning and language comprehension; it is not a complete inventory of stored facts. Public benchmark overlap with pretraining is unknown. Quantization and SVD baselines are deliberately simple and are not GPTQ, AWQ or activation-aware low-rank compression. No compression-aware training, distillation or fine-tuning was used. A different learned representation may produce different results; the current evidence does not justify assuming that a sphere or cubic curve provides exceptional compression by itself.
