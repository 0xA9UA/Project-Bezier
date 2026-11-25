<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../PUBLICATION_SCOPE.md) · [Reader guide](../../READ_THIS_FIRST.md).

# Mathematical specification

This project implements the interior-ball interpretation from the referenced conversation. It is a geometric mixture of experts with a fixed number of residual state updates. It is not an implementation of a general-purpose language model, an exact ODE solver, or a claim of architectural novelty.

## Entry, route, and exit

An image x is encoded into h₀ = SiLU(Wₑx + bₑ), with h₀ in Rᵈ. A small router maps h₀ to twelve real numbers, grouped into four three-dimensional vectors uₐ, uᵦ, v₁, v₂. They define:

    a = uₐ / ‖uₐ‖                  b = uᵦ / ‖uᵦ‖
    Pᵢ = 0.95 vᵢ / sqrt(1 + ‖vᵢ‖²)
    γ(t) = (1−t)³a + 3(1−t)²tP₁ + 3(1−t)t²P₂ + t³b

The normalization function returns (1,0,0) for a near-zero raw surface vector. This defines a valid surface point in that exceptional case, with no gradient through the fallback. Entry and exit start near opposite poles and are learned independently. The implementation does not enforce a minimum separation between the two ports.

The cubic Bernstein coefficients are nonnegative and sum to one on [0,1]. Since a and b have unit norm and the interior controls have norm below 0.95, γ(0)=a, γ(1)=b and ‖γ(t)‖<1 for 0<t<1, in exact arithmetic. The extra radius margin helps preserve this numerically. Computation is sampled strictly between the endpoints. Geometry tests check endpoints, containment, reversal, and the derivative independently with finite differences.

Position is an address, not the complete information state. The hidden vector has d=64 features by default; the route has only three coordinates. The readout D(hₛ) produces ten class logits at the exit. The exit position is not itself interpreted as a class or answer.

## Spatial processing field

There are M=8 shared experts, located at learned centers:

    qₘ = 0.85 zₘ / sqrt(1 + ‖zₘ‖²)
    scoreₘ(r) = −‖r − qₘ‖² / σ²           σ = 0.6
    A(r) = indices of the k largest scores  k = 2
    αₘ(r) = exp(scoreₘ(r)) / Σⱼ∈A(r) exp(scoreⱼ(r))
    gₘ(h) = W₂ₘ SiLU(W₁ₘ LayerNorm(h) + b₁ₘ) + b₂ₘ

LayerNorm is shared across experts. Each expert is a 64→32→64 bottleneck. We evaluate S=4 residual updates:

    tₛ = (s + 1/2) / S,                 s = 0,…,S−1
    hₛ₊₁ = hₛ + (1/S) Σₘ∈A(γ(tₛ)) αₘ(γ(tₛ)) gₘ(hₛ)

This is a fixed-depth network motivated by dh/dt=F(h,γ(t),t). It uses a first-order explicit hidden-state update with the prescribed spatial route sampled at interval midpoints. It is **not** the second-order midpoint ODE method: it does not compute an intermediate hidden state. No numerical-error tolerance, adaptive solver, or exact continuous-flow guarantee is claimed. Changing the number of steps changes the discrete architecture and generally calls for retraining.

Task gradients flow through the selected scores, their softmax weights, centers, and route controls. Top-k membership is discrete and piecewise constant; this is not a globally smooth field. The classification gradient through a renormalized top-1 gate is zero almost everywhere. Use k≥2 for learning task-dependent geometry in this implementation. A small soft load-balancing term uses all routing probabilities during training:

    loss = cross_entropy + 0.01 (M Σₘ mean_batch,step[pₘ]² − 1)

This encourages balanced soft importance; it does not guarantee equal hard-selection counts. Both soft routing and actual selection counts can differ. The report includes the actual counts.

## Actual sparse execution

The sparse dispatcher gathers selected expert matrices into batched tensors and executes batched matrix products for those selections only. It computes B×S×k expert transformations, versus B×S×M in the diagnostic all-expert implementation. Unselected expert matrices remain stored, and all M routing distances are still computed. Gathering repeats weights across examples and can create considerable memory traffic. This is a transparent PyTorch proof of concept rather than a fused, optimized expert kernel.

The `dense_reference` path evaluates every expert, then selects the same outputs and weights. It should produce the same logits and gradients. Tests compare both paths, count evaluated expert rows, and verify that unselected expert parameters receive zero task gradient within the bank. This reference measures savings relative to evaluating the **same** sparse model inefficiently; it is not an additional trained architecture.

## Optional Bézier transfer curves

The `bezier_transfer` variant replaces each bottleneck SiLU with a trainable scalar cubic:

    t = sigmoid(z)
    φₘⱼ(z) = Σᵢ₌₀³ cₘⱼᵢ Bᵢ³(t)

Coefficients initialize to [−1, −1/3, 1/3, 1], so the initial response is 2 sigmoid(z) − 1. There are four coefficients per expert bottleneck channel. This is a per-channel activation, **not** a full four-coefficient function on every input/output edge. It tests a compact version of nonlinear Bézier responses while keeping the main routing comparison separate. This variant has more parameters and a different activation; improvements cannot be attributed solely to spatial routing.

## Comparators and interventions

| Model | Route | Interior response | Default expert transformations per input |
|---|---|---|---:|
| bezier | Learned cubic with learned surface ports | SiLU | 4×2=8 |
| straight | Learned surface ports joined by a chord | SiLU | 4×2=8 |
| moe | Linear gate from current hidden state; no spatial domain | SiLU | 4×2=8 |
| residual | All eight distinct blocks in sequence | SiLU | 8 |
| bezier_transfer | Learned cubic | Learned cubic scalar activations | 4×2=8 |

The straight model constructs the cubic control points at one-third and two-thirds of the chord, which gives exactly a straight line. Its intermediate points are strictly inside the ball when its surface ports are distinct; coincident ports are a degenerate boundary path. The Bézier model's strict interior guarantee holds even with coincident ports because its inner controls are strictly interior.

All models share encoder/decoder dimensions, expert dimensions, optimizer, number of epochs and training split. The expert bank initialization is aligned across matching seeds. Router size, execution depth, residual scaling and total parameter count differ and are disclosed. This is a controlled starting point, not exhaustive matching of optimized accuracy/latency frontiers.

For the trained spatial models, we also replace the controls with a chord, shuffle routes among inputs, and bypass the interior field. Those interventions show whether a particular trained network uses its geometry and interior computation. They are not retrained or tuned alternatives. The encoder and decoder can solve part of the task on their own, so bypass accuracy is reported rather than assuming all predictive power comes from geometry.

## Scope

Paths can point in arbitrary spatial directions. Reversing the control-point order reverses the geometry; it does not invert the neural computation. Routes that cross do not exchange information. This prototype has no adaptive halting, branching, token attention, recurrent state across samples, enforced rotation equivariance, or biological interpretation. Those would be separate experiments.

The test is small and single-domain. Lower expert arithmetic is not equivalent to lower latency, memory traffic, or energy. The report measures eager forward latency and CUDA allocator peaks, and reports partial linear MAC counts. It does not measure physical memory traffic or energy use. General inference-efficiency claims would require broader tasks, independently tuned baselines, larger batches/model widths, and optimized dispatch kernels.

## Sources and provenance

- User-provided discussion: `referenced_conversation.txt`, retrieved from conversation `6b042eb6-5f0c-83ed-b62c-81046fb07428`. The mathematical structure above is an implementation choice grounded in that discussion.
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366): learned hidden-state dynamics. This prototype uses fixed unrolled updates, not the paper's adjoint solver.
- [Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538): conditional expert computation. The geometric routing here is an experimental design choice.
- [scikit-learn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html): bundled 8×8 handwritten digits, 1,797 examples and ten labels. The saved dataset uses these bundled records and a local stratified split; it is not MNIST and not an official writer-disjoint evaluation.
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) and [PyTorch 2.10 release](https://pytorch.org/blog/pytorch-2-10-release-blog/): pinned CUDA 12.8 wheels and Python 3.14 support. The installed wheel's actual CUDA execution was verified on the local RTX 5090.
