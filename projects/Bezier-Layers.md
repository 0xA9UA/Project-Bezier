# Bezier-Layers

`Bezier-Layers` · Routing and reasoning

Test content-associated layer selection across CPU/GPU boundaries, then reverse the split to keep the expert bank in system RAM.

## Saved status

Completed both split directions; placement and feedback verified, speed advantage unestablished.

## Work completed or attempted

- Implemented CPU-selected layers/GPU experts and GPU-selected layers/CPU experts, catalogues, grouped dispatch, cached geometry, pinned transfers, and decision-cadence controls.
- Completed six original digit checkpoints, cached/uncached timing studies, a trained 50.9M-parameter EMNIST model, and an untrained 269.9M execution stress model.

## Findings

**Measured.** Original digit feedback switched layers on 6.20% of predictions and improved assigned-family selection from 91.48% to 95.83%; final accuracy was 95.46% versus 95.56% for separately trained entry-only selection, without a demonstrated overall improvement.

[S058](../evidence/Bezier-Layers/README.md)

**Measured.** The 50,915,372-parameter model achieved 77.95% on 9400 EMNIST test images, identical across seven execution variants. Layer switching occurred on 28.97%; maximum logit difference was 1.14e-5.

[S061](../evidence/Bezier-Layers/runs/reversed_large/report.md)

**Measured.** Reversed grouped dispatch beat matching CPU-only execution in 0/12 median settings and the fastest single-device baseline in 0/12. The 269,946,720-parameter stress model was untrained.

[S061](../evidence/Bezier-Layers/runs/reversed_large/report.md)

**Measured.** At batch 128, reversed placement reduced GPU tensor peaks to about 9.14 MiB for the trained model and 9.16 MiB for the stress model, with expert weights in system RAM. Four decisions transfer 160 logical bytes per input; total model bytes are not compressed.

[S061](../evidence/Bezier-Layers/runs/reversed_large/report.md)

## Limits and open questions

- Original study has three seeds; larger EMNIST study has one seed and assigned class groups taught with auxiliary labels.
- Synchronous device decisions add dependent communication; CPU-only already has zero GPU weight residency.
- No LLM, model paging, RL, fused-kernel, or reversed-training-speed result.

## Recommended next steps

- Recommendation: Treat RAM placement as a deployment tradeoff; compare to CPU-only and optimize larger workloads before claiming speed gains.

## Sources

- [Original layer experiment](../evidence/Bezier-Layers/README.md)
- [Reversed mode and commands](../evidence/Bezier-Layers/REVERSED.md)
- [Original report](../evidence/Bezier-Layers/runs/main/report.html)
- [Larger-model measured results](../evidence/Bezier-Layers/runs/reversed_large/report.md)
- [Saved verification](../evidence/Bezier-Layers/runs/reversed_large/verification.json)

## Related projects

- [Bezier-Routing](Bezier-Routing.md)
- [Bezier-Engine](Bezier-Engine.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
