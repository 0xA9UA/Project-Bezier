# Project Bézier

**Informal research notes on geometric routing, shared computation, model compression, and recurrent reasoning.**

This repository shares a November 25, 2026 snapshot of an exploratory research program. It brings together **26 project folders, 137 findings, and 247 source references**, including the experiments that worked, the controls that explained apparent gains, and the approaches that failed.

The strongest results are concrete: learned routes can change which computation executes; shared fields and quantization can reduce storage; and one nested-core pilot showed a useful steering effect. Broad improvements in language-model quality, coding ability, or inference efficiency remain unestablished. These are working research notes, **not peer-reviewed findings**.

## Start reading

- [What this research is useful for](READ_THIS_FIRST.md): practical lessons, evidence strength, and reproduction limits.
- [Second-pass review](REVIEW.md): research-quality and privacy checks.

- [Research atlas](atlas/index.html): searchable project summaries, findings, limitations, and evidence links. Download the repository and open the HTML in a browser to use the interactive version.
- [Consolidated report](reports/project_bezier_consolidated.pdf): the complete narrative in PDF form.
- [Research notes](RESEARCH_NOTES.md): the main questions, interpretation, and next experiments.
- [Project notes](projects/): a separate entry for each project and revision.
- [Structured index](data/project_index.json): the consolidated record in JSON.
- [Evidence snapshot](evidence/PUBLICATION_SCOPE.md): selected cited reports, measurements, and implementation material.

The publication contains a curated evidence snapshot. It does not include the entire local workspace, pretrained model downloads, training datasets, or a complete set of checkpoints. Reproduction instructions in historical sources may refer to artifacts that are not distributed here.

## What emerged

| Question | What the saved evidence supports |
|---|---|
| Does geometric routing affect computation? | Yes. Activation and coordinate interventions changed later expert or node selections and model outputs. A useful routing mechanism still needs to outperform a matched fixed, shuffled, or ordinary-routing control. |
| Can sharing or compression reduce storage? | Yes, under the tested conditions. Shared fields helped several bounded tasks; Granite's later validation-selected mixed-precision artifact passed its fresh quality screen at **16.464 GiB**. Simpler quantization controls remain competitive. |
| Does steering ever help? | In the **nested-core v4 single-seed pilot**, enabling redirects improved the same checkpoint from **52.73% to 57.62%** on **512 answer-prefix cases from 32 programs, one seed**. The clean model reached 57.03%; the 4.89-point gain is against disabled redirects. The result needs new holdouts and additional seeds. |
| Does learned routing help the pretrained 7B models? | An input-specific benefit over fixed and shuffled routes remains unresolved. The v8 release rule retained **fixed `(2,2)`** as the default policy. |
| Do more internal visits produce a better coding agent? | The matched coding studies found no dependable gain. The from-scratch 7.41B training run completed, but its partially scored final panels achieved **0/60** and **0/40** successful tasks. |
| Does less arithmetic or GPU memory mean faster inference? | Often it did not. Transfers, dispatch, repeated passes, and allocation costs mattered; some reduced-memory designs were substantially slower. |

Evidence: [routing interventions](evidence/Bezier-Routing/README.md), [coordinate-driven execution](evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md), [shared fields](evidence/Bezier-Explore/RESULTS.md), [Granite mixed precision](evidence/Bezier-Granite/rl_compression/RESULTS.md), [nested-core v4](evidence/bezier_nested_core_v4/runs/input_preserving/report.md), [pretrained v8](evidence/bezier_nested_llm_7b_v8/REPORT.md), [coding controls](evidence/Bezier-Revisit/RESULTS.md), and [overnight training check-in](evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/checkin_2026-11-24.json).

## The research branches

| Branch | Scope |
|---|---|
| Original sphere | Digit classification, a 92-run scaling study, an untrained execution sweep, and numerical checks of restricted mathematical constructions. |
| Routing and execution | Activation feedback, Boolean-program generation, CPU/GPU placement, repeated transformer passes, and return visits. |
| Shared fields and compression | Shared processing weights, weight codecs, packed inference, compact replacements, and offline precision-policy optimization. |
| MemorySphere | Small and billion-parameter models trained from scratch with bounded internal scratch memory. |
| Nested cores, v1-v4 | Shared parameter slices, whole-core budgets, scaling, and input-preserving steering. |
| Pretrained 7B, v1-v8 | Shared execution over a pretrained Mistral model, distillation, residency, node routing, recurrent traversal, and controller learning. |

These folders are not 26 independent replications. Some contain multiple experiments; others inherit data, code, checkpoints, or evaluation choices from earlier revisions. The from-scratch MemorySphere models and pretrained Mistral adaptations are distinct lines of work.

## How to interpret the results

The project notes distinguish saved measurements, narrative reports, implemented mechanisms, and hypotheses. The publication consolidates existing evidence; its assembly did not rerun training, inference, or the historical test suites.

Comparisons use different datasets, sample counts, seeds, answer formats, compute budgets, and runtime paths. A percentage from one branch should not be ranked against a percentage from another. Several later revisions are adaptive pilots, and paired-program confidence intervals do not capture uncertainty across training seeds. Untrained stress tests establish execution behavior only.

The main open experiment is to replicate nested-core v4 with additional seeds and a newly sequestered holdout. For the pretrained 7B line, the priority is a reliable gain over validation-selected fixed and shuffled policies. For from-scratch coding, valid tool actions and completed workflows need to improve before longer training is treated as progress.

Corrections, independent replications, and stronger matched controls are welcome. When discussing a result, please identify its project, revision, evaluation panel, and source artifact so the claim remains traceable.
