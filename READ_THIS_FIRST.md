# What this research is useful for

This is an exploratory lab notebook for people investigating conditional computation, weight sharing, and geometric routing. Its main value is the record of controls and failure modes, plus a small number of leads worth testing again. It does not establish a generally better neural-network architecture.

## Useful lessons

| Lesson | Evidence and boundary |
|---|---|
| **Test a routing mechanism separately from its benefit.** Moving activations or node coordinates can change later computation without improving answer quality. | [Routing interventions](evidence/Bezier-Routing/README.md) and [pretrained spatial routing](evidence/bezier_nested_llm_7b_v6/EXPERIMENT_REPORT.md). Fixed and shuffled routes are essential controls. |
| **Compare against a compact ordinary model.** Skipping experts can beat a model that executes every stored expert, yet lose to a smaller residual model. | [Original scaling study](evidence/runs/scaling/report.md). Useful for choosing honest storage and latency baselines. |
| **Keep storage, GPU allocation, and latency separate.** Sharing and quantization can save bytes; transfers and dispatch can consume the arithmetic saving. | [Shared fields](evidence/Bezier-Explore/RESULTS.md), [Granite compression](evidence/Bezier-Granite/rl_compression/RESULTS.md), and [streamed core](evidence/bezier_nested_llm_7b_v3/EXPERIMENT_REPORT.md). These are local, implementation-specific measurements. |
| **The input-preserving v4 result is a replication lead.** Redirects improved the same checkpoint from 52.73% to 57.62%. | [V4 report](evidence/bezier_nested_core_v4/runs/input_preserving/report.md): one seed, 512 answer-prefix cases across 32 program files, reused holdouts. The separately trained clean model reached 57.03%; the 4.89-point gain is against disabled redirects, not that clean model. |
| **More internal computation needs an outcome test.** Repeated visits and controller training did not reliably improve the measured coding workflows or input-specific routing. | [Matched coding study](evidence/Bezier-Revisit/RESULTS.md) and [pretrained v8](evidence/bezier_nested_llm_7b_v8/REPORT.md). They help avoid treating elaborate routes, lower token loss, or successful training as proof of useful reasoning. |

## What readers can and cannot do

The summaries, saved measurements, and some per-example records allow inspection of the reported comparisons. For example, v8's published test records support recounting 490 learned, 485 fixed, 481 shuffled, and 495 dense correct answers out of 768 prompts from 192 program groups.

This is **not a runnable release of all 26 projects**. Only selected source material is included. Training datasets, model-weight payloads, complete environments, and much of the implementation are absent. Historical reports can mention files available in the original workspace that are not distributed here; see the [publication scope](evidence/PUBLICATION_SCOPE.md) and [source manifest](data/source_manifest.json).

The most useful next step is independent replication of the v4 lead with fresh programs and additional seeds. Larger claims require matched quality/resource comparisons, stronger external baselines, and evidence beyond these small or synthetic tasks. Publishing these records does not independently verify the experiments or establish novelty.
