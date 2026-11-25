# Granite 30B: compact inference and a separate routing experiment

This project adapts the user's local Granite 4.2 30B model. The original model
is read from `DRIVE/AI/Models/Tensor_Models/granite-4.2-30b` and remains unchanged.
All new code, artifacts, calibration activations and results stay in this folder.

Two tracks are kept separate:

1. **Quality-preserving compression:** six-bit weight representations, comparing
   a shared cubic Bézier quantization table with symmetric uniform quantization.
   The runtime keeps packed weights on the GPU and reconstructs only the current
   matrix. Embeddings, output head and normalization weights remain BF16.
2. **Experimental architecture:** a locally distilled replacement for one MLP
   (layer 32), with shared processing weights and activation-dependent Bézier
   routing. Straight, frozen-feedback and fixed-route controls are included.

Read [RESULTS.md](RESULTS.md) for measured results and [the frozen protocol](docs/protocol.md)
for the comparison boundaries. Exact inputs, scores and uncertainty intervals live
in `data/` and `runs/summary.json`.

The subsequent [reinforcement-learning compression experiment](rl_compression/README.md)
keeps both Bézier quantization and the shared-routing block while learning mixed
3/4/6-bit precision assignments. Its separate [results](rl_compression/RESULTS.md)
include fixed-precision and random-search controls on fresh evaluation inputs.

## Use the compressed model

Run from `WORKSPACE/Bezier-Granite`. The existing sibling Python environment
is reused without installing or changing packages. This is a custom PyTorch loader;
the packed files are not directly loadable by Ollama, llama.cpp, or ordinary
`AutoModelForCausalLM.from_pretrained`.

```powershell
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.generate --artifact artifacts/bezier6 --prompt "Explain how a hash table works." --max-new-tokens 128
```

Generation preserves Granite's tokenizer, chat template, attention, RoPE, residual
scales and KV-cache implementation. The default invokes the template's non-thinking
options; the tokenizer controls their interpretation. Outputs are demonstrations,
not a general correctness guarantee. Long contexts need additional KV-cache memory.

To load from Python (working directory here):

```python
from granite_bezier.model import load_packed
model = load_packed("artifacts/bezier6")
```

To try the standalone experimental artifact:

```powershell
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.generate --artifact artifacts/bezier6-shared-bezier --prompt "Explain how a hash table works." --max-new-tokens 128
```

It physically replaces one processing block and may reduce quality. Its files are
separate from the practical artifact. Alternatively, `--experimental bezier` loads
the fitted replacement over the practical artifact in memory.

## Reproduce

```powershell
..\Bezier-Compress\.venv\Scripts\python.exe -m pytest tests -q
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.prepare
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.evaluate --name reference --capture
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.convert --method bezier --bits 6
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.convert --method uniform --bits 6
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.evaluate --artifact artifacts/bezier6 --name bezier6
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.evaluate --artifact artifacts/uniform6 --name uniform6
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.verify --artifact artifacts/bezier6
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.verify --artifact artifacts/bezier6 --live
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.experimental --steps 800
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.experimental --evaluate bezier
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.experimental --evaluate frozen
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.experimental --evaluate straight
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.experimental --evaluate fixed
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.export_experimental --mode bezier
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.verify --artifact artifacts/bezier6-shared-bezier --live
..\Bezier-Compress\.venv\Scripts\python.exe -m granite_bezier.report
```

Conversion resumes incomplete artifacts and leaves complete ones intact. Prepared
evaluation selections are frozen. Evaluation commands intentionally replace results
for their named condition; use `--name` for an additional compression comparison.
Experimental training keeps completed mode checkpoints. Use a fresh copy of this
project to run an independent full replication without replacing results.

Both exported artifacts are self-contained apart from this loader and its Python
dependencies. Neither loads the original 59 GB weights at inference. The small
training checkpoints in `experimental/` additionally support applying replacements
to the practical artifact in memory.

## Interpretation

A six-bit code and group scale still exist for every weight. Four Bézier controls
describe a shared set of quantization levels, not all the model's knowledge.
This differs from the earlier spherical architecture. Whole-model shared routing
would require broader distillation or retraining and additional validation.

The exactly integrable linear field from Bezier-Explore is not applicable to
arbitrary Granite attention/SwiGLU blocks. No analytical shortcut is claimed here.

Source model: [IBM Granite 4.2 30B](https://huggingface.co/ibm-granite/granite-4.2-30b),
Apache-2.0 according to its supplied model card. Cached public WikiText-2 and ARC-Easy
data from the prior project are reused and re-tokenized. No uploads, external training,
new model downloads, or paid services are used.
