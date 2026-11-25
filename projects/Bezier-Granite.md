# Bezier-Granite

`Bezier-Granite` · Compression and shared fields

Adapt a local Granite 4.2 30B checkpoint to a 32 GB GPU using six-bit packed weights and test a compact feedback-routed replacement block.

## Saved status

Completed six-bit deployment, single-block adaptation, and later mixed-precision RL compression; smallest validation-selected screened artifact is 16.464 GiB.

## Work completed or attempted

- Built uniform6 and Bezier6 standalone artifacts, frozen language/QA screens, streamed BF16 reference evaluation, and a packed on-GPU runtime.
- Distilled one layer-32 MLP into a compact shared field; compared curved, straight, frozen-feedback, and fixed controls; exported the replacement independently.
- Completed a separate offline mixed-precision RL compression study: sixteen REINFORCE updates over 64 sampled section assignments, matched random search, fixed-bit controls, frozen validation selection, and a fresh final test.

## Findings

**Measured.** Six-bit artifacts occupy 22.245 GiB versus 54.532 GiB reference payload. On 16,384 WikiText-2 tokens and 256 ARC-Easy questions, Bezier6 had perplexity 20.0582, QA 82.03%, and 98.58% reference-correct retention, passing the project's screens. Uniform6 also passed.

[S041](../evidence/Bezier-Granite/RESULTS.md) · [S042](../evidence/Bezier-Granite/runs/summary.json)

**Measured.** Packed Bezier6 retained 22.235 GiB persistent tensors, with 23.567 GiB peak allocation; separately reconstructed dense last-token logits matched exactly. One 32-token prefill took 0.958 seconds.

[S041](../evidence/Bezier-Granite/RESULTS.md) · [S042](../evidence/Bezier-Granite/runs/summary.json)

**Measured.** The single-block standalone replacement used 21.960 GiB, saving 291.7 MiB versus the practical compressed artifact. All four fitted controls scored 81.25% QA; frozen feedback had the lowest replacement perplexity (20.2454 versus Bezier 20.2729). No added quality gain from curvature or feedback was established.

[S041](../evidence/Bezier-Granite/RESULTS.md) · [S042](../evidence/Bezier-Granite/runs/summary.json)

**Measured.** A latent perturbation changed 25.20% of later group choices in the Bezier replacement versus 0% for frozen feedback. This demonstrates adaptive routing, not better reasoning; local reconstruction relative MSE remained about 0.8021.

[S041](../evidence/Bezier-Granite/RESULTS.md) · [S042](../evidence/Bezier-Granite/runs/summary.json)

**Measured.** The later RL mixed-precision artifact used 16.464 GiB, 25.0% below the 21.960 GiB six-bit shared parent, and passed the prespecified validation and final quality screens. On the fresh 32768-token/512-question test it scored perplexity 18.2305, QA 80.86%, and original-correct retention 95.74%.

[S044](../evidence/Bezier-Granite/rl_compression/RESULTS.md) · [S046](../evidence/Bezier-Granite/rl_compression/test_results.json)

**Measured.** The RL policy chose 3/4/6-bit precision by four-layer section; it did not train inference-time reasoning. Fixed four-bit (15.429 GiB) and random mixed precision (15.636 GiB) also passed final test despite failing validation. RL had lower perplexity but occupied more storage; it did not establish dominance over these alternatives.

[S044](../evidence/Bezier-Granite/rl_compression/RESULTS.md)

**Measured.** The mixed-precision artifact retained 16.454 GiB persistent GPU tensors and peaked at 17.790 GiB for the recorded inference/KV check. Packed versus reconstructed dense last-token logits matched exactly. Large same-artifact timing variation prevented a reliable generation-speed conclusion.

[S044](../evidence/Bezier-Granite/rl_compression/RESULTS.md) · [S045](../evidence/Bezier-Granite/rl_compression/summary.json)

## Limits and open questions

- Limited quality screens and one block/fitting seed; no whole-model architectural conversion or broad reliability claim.
- Eager current-matrix reconstruction targets residency, not fused low-bit speed; KV cache and driver/display memory are additional.
- Reference size counts original tensor payload, while compressed size includes all artifact files.
- The RL compression follow-up uses different fresh evaluation subsets from the earlier six-bit study; their perplexity and QA values must not be pooled. Its screen is point-estimate preservation, not statistical equivalence.

## Recommended next steps

- Recommendation: Retain uniform6 as a matched control; broaden tasks/context lengths and isolate replacement training from route geometry.

## Sources

- [Results and boundaries](../evidence/Bezier-Granite/RESULTS.md)
- [Machine-readable quality and runtime data](../evidence/Bezier-Granite/runs/summary.json)
- [Deployment and reproduction](../evidence/Bezier-Granite/README.md)
- [Later RL precision-policy compression results](../evidence/Bezier-Granite/rl_compression/RESULTS.md)
- [Mixed-precision search and final metrics](../evidence/Bezier-Granite/rl_compression/summary.json)
- [Fresh final-test quality screen](../evidence/Bezier-Granite/rl_compression/test_results.json)
- [Frozen pre-test selection](../evidence/Bezier-Granite/rl_compression/pre_test_freeze.json)

## Related projects

- [Bezier-Compress](Bezier-Compress.md)
- [Bezier-Routing](Bezier-Routing.md)
- [Bezier-Explore](Bezier-Explore.md)
- [Bezier-LongHorizon](Bezier-LongHorizon.md)

Saved local evidence reviewed for the 2026-11-25 consolidation; historical tests and measurements were not rerun.

[Back to the project directory](README.md)
