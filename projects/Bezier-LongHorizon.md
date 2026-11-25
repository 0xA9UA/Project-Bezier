# Bezier-LongHorizon

`Bezier-LongHorizon` · Compression and shared fields

Build compressed GPT-OSS-120B artifacts and test a distilled shared-field replacement plus controller RL on bounded coding and cumulative workflows.

## Saved status

Completed compression and bounded coding pilot; controller RL regressed measured outcomes.

## Work completed or attempted

- Requantized expert weights into a three-bit Bezier codebook; built SSD-streamed execution with packed expert caching and a verified decoder.
- Replaced one expert bank using supervised distillation; ran eight controller-only REINFORCE episodes and exported standalone distilled/RL variants.
- Evaluated four authored coding tasks and one six-stage cumulative project; retained output/token IDs, hidden checks, timing, source/export hashes, and mechanism controls.

## Findings

**Measured.** Logical standalone deployment sizes were 60.794 GiB native MXFP4, 47.453 GiB Bezier three-bit, and 46.256 GiB compact variants: 21.945% and 23.913% smaller than native. Shared hardlinks mean these are not incremental physical disk totals.

[S084](../evidence/Bezier-LongHorizon/RESULTS.md) · [S085](../evidence/Bezier-LongHorizon/runs/results.json)

**Measured.** By four attempts, native and three-bit models solved 3/4 coding tasks, distilled compact 4/4, and compact+RL 3/4. The lone distinguishing task depended on finishing under a 384-token limit, so the compact win does not establish stronger general reasoning.

[S084](../evidence/Bezier-LongHorizon/RESULTS.md) · [S085](../evidence/Bezier-LongHorizon/runs/results.json)

**Measured.** Native, three-bit, and pre-RL compact passed cumulative stages 1-5 and finished with 10/12 checks; RL passed only stage 1 and finished with 8/12. No variant passed the final stage. RL introduced absent-item edge-case failures.

[S084](../evidence/Bezier-LongHorizon/RESULTS.md)

**Measured.** RL changed only 16,841 controller parameters across eight episodes; it did not compress weights or train the full base. Curved, straight, and frozen local reconstruction MSEs were similar (~0.0902), without matched coding evaluations for every geometry control.

[S084](../evidence/Bezier-LongHorizon/RESULTS.md) · [S087](../evidence/Bezier-LongHorizon/runs/rl_diagnostics.json)

## Limits and open questions

- Four tiny authored tasks, one dependent workflow, one fitting/RL seed, low reasoning effort, and tight output caps.
- Later variants used a different verified packed-cache runtime, so descriptive request times do not isolate an architectural speedup.
- Artifacts do not fit entirely in 32 GiB VRAM; SSD streaming and shared desktop variability matter.
- One expert-bank replacement, short-context prompts (max prompt+output 2681 tokens), and eight RL episodes are not general long-horizon evidence.

## Recommended next steps

- Recommendation: Keep pre-RL compact control; broaden independent workflows with matched runtime, reasoning budgets, seeds, and geometry controls.

## Sources

- [Complete results and interpretation](../evidence/Bezier-LongHorizon/RESULTS.md)
- [Machine-readable outcomes](../evidence/Bezier-LongHorizon/runs/results.json)
- [Report consistency audit](../evidence/Bezier-LongHorizon/runs/report_audit.json)
- [Controller changes and rewards](../evidence/Bezier-LongHorizon/runs/rl_diagnostics.json)
- [Artifacts and commands](../evidence/Bezier-LongHorizon/README.md)

## Related projects

- [Bezier-Granite](Bezier-Granite.md)
- [Bezier-Compress](Bezier-Compress.md)
- [Bezier-Think](Bezier-Think.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
