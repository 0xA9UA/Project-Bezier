# Bezier-Compress

`Bezier-Compress` · Compression and shared fields

Compress pretrained SmolLM2 weights using cubic fits or curve-controlled codebooks while measuring language and QA preservation.

## Saved status

Completed primary and exploratory compression studies; ordinary int8 is the robust preservation result.

## Work completed or attempted

- Completed 20 primary comparisons on 135M and 1.7B SmolLM2, plus a seven-case exploratory zero-anchored follow-up.
- Compared direct/sorted cubic fitting, quantized curve/free codebooks, uniform int4/int8, and low-rank approximation; counted complete artifact bytes and independently reconstructed 22 distinct artifacts.

## Findings

**Measured.** Only ordinary int8 passed all prespecified validation/test preservation thresholds at both sizes. For 1.7B, int8 used 1779.56 MiB versus 3267.52 MiB FP16 and retained 99.46% of reference-correct answers.

[S032](../evidence/Bezier-Compress/README.md) · [S039](../evidence/Bezier-Compress/runs/main/summary.json)

**Measured.** Primary 1.7B cubic-codebook int4 used 1011.59 MiB (3.23x reduction), but perplexity rose from 10.706 to 16.200 and QA fell from 72.95% to 63.18%. Lower weight reconstruction error did not imply better model behavior.

[S032](../evidence/Bezier-Compress/README.md) · [S039](../evidence/Bezier-Compress/runs/main/summary.json)

**Measured.** On a separate follow-up subset, zero anchoring improved cubic-codebook perplexity from 16.551 to 12.271 and QA from 62.30% to 70.31%. It still missed preservation thresholds against FP16 (10.720 perplexity; 74.12% QA); a free zero-anchored codebook was similarly effective.

[S032](../evidence/Bezier-Compress/README.md) · [S040](../evidence/Bezier-Compress/runs/anchor_followup/summary.json)

**Reported.** Saved completion checks record 27 evaluation cases, 22 reconstructed artifacts, and 23 passed unit tests. The tested runtime reconstructs dense FP16 weights and does not reduce steady-state GPU parameter memory.

[S035](../evidence/Bezier-Compress/runs/completion_checks.json) · [S032](../evidence/Bezier-Compress/README.md)

## Limits and open questions

- One model family, two sizes, simple quantization baselines, no compression-aware training.
- Follow-up is a fresh subset of the same corpora, not an independent dataset; zero anchoring and control precision changed together.
- No compressed execution kernel or demonstrated inference-speed benefit.

## Recommended next steps

- Recommendation: Compare against stronger activation-aware quantizers and evaluate a packed runtime separately from artifact compression.

## Sources

- [Overview and numerical results](../evidence/Bezier-Compress/README.md)
- [Primary report](../evidence/Bezier-Compress/runs/main/report.html)
- [Zero-anchor follow-up](../evidence/Bezier-Compress/runs/anchor_followup/report.html)
- [Completion record](../evidence/Bezier-Compress/runs/completion_checks.json)
- [Codec and byte accounting](../evidence/Bezier-Compress/docs/codecs.md)

## Related projects

- [Bezier-Explore](Bezier-Explore.md)
- [Bezier-Granite](Bezier-Granite.md)
- [Bezier-LongHorizon](Bezier-LongHorizon.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
