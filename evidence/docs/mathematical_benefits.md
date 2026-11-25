# Mathematical benefits of Bézier sphere networks

Research and algebra checks completed on 23 November 2026.

The structure can be more efficient when it captures relationships that the task actually has. The strongest possibilities are compact routing, shared processing weights, and a restricted field whose complete evolution can be evaluated analytically. A sphere and a curve do not by themselves guarantee a smaller or better neural network.

The existing prototype implements geometric sparse routing through independently stored nonlinear experts. Its measurements establish some speed benefits against a network executing every stored block, but no overall storage or reliability advantage. Shared weight fields and exact integration are proposed changes, not features demonstrated by the 92 trained comparisons. The derivations below are deductions for specified models; the linked papers establish related mechanisms rather than validate this particular architecture.

## What sparse execution already saves

Let \(M\) be the number of stored experts, \(d\) the hidden width, \(b\) the expert bottleneck width, \(S\) the number of processing steps, and \(k\) the experts selected at each step. Ignoring biases, the expert bank stores

\[
P_{\mathrm{bank}}=2Mdb
\]

weights. Evaluating the selected experts takes approximately

\[
C_{\mathrm{experts}}=2Skdb
\]

multiply-accumulates per input, excluding routing, nonlinearities and memory operations. A comparator executing all \(M\) blocks once costs \(2Mdb\). Thus its expert-arithmetic ratio to this routed model is \(M/(Sk)\). At \(M=64,S=4,k=2\), that ratio is eight. This comparison assumes the same expert dimensions; it is not a matched-accuracy speedup theorem.

Increasing stored experts can therefore add capacity without proportional expert arithmetic. This is the established conditional-computation principle in [Shazeer et al., Sparsely-Gated Mixture-of-Experts (2017)](https://arxiv.org/abs/1701.06538). It applies to ordinary routing too. All \(M\) expert weights still occupy storage.

## Bézier routing has an exact compact factorization

For a fixed input, write its cubic route as

\[
\gamma(t)=\sum_{i=0}^{3}B_i^3(t)P_i.
\]

The current router scores an expert center \(c_e\) using

\[
s_e(t)=-\|\gamma(t)-c_e\|^2/\sigma^2.
\]

Expanding the square gives

\[
s_e(t)=
\frac{2c_e^\top\gamma(t)-\|c_e\|^2}{\sigma^2}
-\frac{\|\gamma(t)\|^2}{\sigma^2}.
\]

The last term is the same for every expert at a given time. Removing it preserves top-k membership and softmax weights. The equivalent scores are

\[
\widetilde s_e(t)
=\sum_{i=0}^{3}B_i^3(t)
\underbrace{\frac{2c_e^\top P_i-\|c_e\|^2}{\sigma^2}}_{A_{ie}}.
\]

At any number of sampled times, the equivalent score matrix is \(BA\), with \(B\) having four columns. Its rank is at most four. This bound concerns logits after removing the common term, not the rank of the softmax probabilities or the neural network's output.

There are several consequences:

- A route has four three-dimensional control points: twelve scalar coordinates regardless of how densely it is sampled. Explicitly storing \(S\) unrelated spatial points uses \(3S\) coordinates. At the current \(S=4\), there is no coordinate-count saving; the distinction becomes useful for longer routes.
- Routing scores can be computed from a compact position rather than a full hidden vector. An ordinary linear gate uses \((d+1)M\) parameters. The current Bézier router, with router width \(u\), uses \((d+1)u+12(u+1)+3M\), including expert centers. At \(d=1024,u=32,M=64\), these are 65,600 and 33,388 parameters respectively. The savings affect the router; they are tiny relative to the approximately 68-million-parameter expert model.
- The score difference between any two experts is a polynomial of degree at most three in time. Unless it is identically zero, those two experts exchange order at most three times on a route. This limits arbitrary switching. A straight path has the stronger linear restriction, so curvature supplies flexibility at some extra cost.

These restrictions could help when useful expert schedules vary smoothly along an input-dependent trajectory. They also prevent some schedules a less constrained router could express. A generic low-rank router can capture related savings without Bézier curves.

Our numerical check reconstructs the equivalent scores to \(2.7\times10^{-15}\) maximum absolute error and obtains rank four across 257 route samples and 64 experts. This checks the algebra, not a speed gain.

Low-dimensional geometric routing has relevant precedent: [Chi et al., On the Representation Collapse of Sparse Mixture of Experts (2022)](https://arxiv.org/abs/2204.09179) project routing representations onto a lower-dimensional hypersphere and report improved routing behavior and downstream results. Their construction is not a three-dimensional Bézier path through a ball; its empirical benefits do not transfer automatically to this model.

## Substantial storage savings require shared processing weights

Suppose a family of \(M\) matrix transformations really follows a cubic relation:

\[
W_e=\sum_{i=0}^{3}B_i^3(t_e)C_i,
\qquad C_i\in\mathbb R^{d\times d}.
\]

Then storing four control matrices requires \(4d^2\) scalar weights, instead of \(Md^2\) independent weights. With 64 matrices, that is a 16-fold reduction in matrix coefficients. Addresses \(t_e\), any router, biases, encoder and decoder must also be counted when they require storage.

This is a constructive example of a real compression opportunity. It is exact only for a cubic family, and approximate when the transformations are close to that family. Reconstructing or applying a weighted sum of four dense matrices also has a compute cost; storage compression alone does not establish faster inference.

In a deliberately constructed check, 64 matrices of size \(64\times64\) were generated from four control matrices at known uniform addresses. The equivalent float32 tensor budgets are 1 MiB and 64 KiB. Reconstruction has relative Frobenius error \(7.4\times10^{-16}\) in float64 and \(5.1\times10^{-8}\) with float32 controls and evaluation. Fitting the same four-control representation to 64 independent random matrices leaves relative error 0.968. The example demonstrates the benefit and its structural assumption together; it is not compression of the trained checkpoints.

A more general candidate inside the sphere is a shared field

\[
F(h,r)=U\left[a(r)\odot\phi(V^\top h)\right],
\qquad U,V\in\mathbb R^{d\times q}.
\]

Here \(q\) is a shared feature width and \(a(r)\) gives position-dependent coefficients. If each coefficient uses \(J\) spatial basis functions, the main field stores about \(2dq+Jq\) weights. Its main application costs about \(2dq\) multiply-accumulates per step plus coefficient evaluation. The current independent expert bank instead stores approximately \(2Mdb\).

A tensor-product cubic Bézier coefficient volume on a cube containing the ball uses \(J=64\) basis terms. A polynomial field of total degree at most three uses \(J=20\). These are different function classes and must not be assigned the same integration-degree bound. Both can describe shared spatial variation; whether a small shared width preserves task performance is an empirical question.

This is closely related to weight sharing and [HyperNetworks, Ha, Dai and Le (2016)](https://arxiv.org/abs/1609.09106), where a smaller learned generator produces weights of another network. [COIN, Dupont et al. (2021)](https://arxiv.org/abs/2103.03123) provides a separate example of representing structured data through a compact coordinate-dependent function. Its evidence concerns lossy image compression, not arbitrary factual knowledge.

The current prototype stores independent expert matrices. Merely changing their spatial positions leaves that storage essentially intact. To test knowledge compression, shared structure must enter the weights or functions carrying the knowledge.

## A restricted field permits exact evolution along the route

There is a precise case where the interior computation can be evaluated without repeatedly updating the hidden state.

Let \(U\in\mathbb R^{d\times q}\) have orthonormal columns, so \(U^\top U=I\). Let the input determine a route once, before evolution, and define

\[
\frac{dh}{dt}
=U\,\operatorname{diag}(a(\gamma_x(t)))\,U^\top h.
\]

The basis \(U\) is shared across positions and inputs. The position-dependent rates \(a(r)\) do not depend on the evolving hidden state.

Set \(y=U^\top h\). Then each coordinate obeys the scalar equation

\[
\frac{dy_j}{dt}=a_j(\gamma_x(t))y_j.
\]

The component orthogonal to \(U\) remains unchanged. Therefore, with

\[
I_x=\int_0^1a(\gamma_x(t))\,dt,
\]

the exact endpoint is

\[
\boxed{
h(1)=h(0)+U\left[
\big(\exp(I_x)-1\big)\odot U^\top h(0)
\right].
}
\]

The exponential is elementwise. This needs two shared basis projections and scalar operations, after computing \(I_x\).

If each spatial rate \(a_j(r)\) is a polynomial of total degree at most three, composing it with a cubic route produces degree at most nine in \(t\). A five-node Gauss–Legendre rule integrates it exactly in real arithmetic, because an n-node Gaussian rule integrates polynomials through degree \(2n-1\). See [NIST DLMF, Gaussian quadrature](https://dlmf.nist.gov/3.5#v). These are five small coefficient-field evaluations, not five full hidden-state updates. A general tensor-product cubic coefficient volume can reach degree 27 along the route and would instead need fourteen Gaussian nodes.

For \(J=20\), the main cost is approximately \(2dq+5Jq\), plus route construction and scalar exponentials. It does not scale with a numerical integration step count. Our constructed float64 example agrees with 1,024-step Runge–Kutta integration to relative error \(3.3\times10^{-14}\).

The decisive restriction is a common eigenbasis: all the position-dependent matrices commute. General nonlinear experts, or matrices that do not commute, cannot use this formula. A separate two-dimensional noncommuting counterexample in the check gives about 10.2% relative error if one incorrectly exponentiates the integrated matrix and ignores ordering.

The encoder and input-dependent router can still make the overall input/output mapping nonlinear, but this interior field is less expressive than arbitrary nonlinear dynamics. Its outcome depends on integrated rates rather than the full order of transformations. Affine spatial rates depend only on the mean of the four route controls; higher-degree rates are needed to retain more curve-shape information. Even then, distinct paths can have identical integrated rates.

This proposed variant fits the continuous-dynamics viewpoint of [Neural Ordinary Differential Equations, Chen et al. (2018)](https://arxiv.org/abs/1806.07366). The exact formula here follows from the stated special structure; it is not a general result of that paper or a property of the existing discrete prototype.

## Geometry gives bounded route perturbations

Because the Bernstein weights are nonnegative and sum to one,

\[
\sup_{t\in[0,1]}
\|\gamma(t)-\widehat\gamma(t)\|
\leq \max_i\|P_i-\widehat P_i\|.
\]

Thus perturbing every control point by at most \(\varepsilon\) moves the route by at most \(\varepsilon\). This is useful for control-point quantization and regularization. It does not by itself bound the error in predictions.

For unchanged expert centers with norm at most \(R\), perturbing a position by at most \(\varepsilon\) changes any pairwise score difference by at most

\[
\frac{4R\varepsilon}{\sigma^2}.
\]

To see this, subtract the two distance scores: the position-dependent term is \(2r^\top(c_e-c_f)/\sigma^2\), whose change is bounded by \(2\varepsilon\|c_e-c_f\|/\sigma^2\).

Consequently, the top-k set is preserved wherever the gap between the kth and next score exceeds \(4R\varepsilon/\sigma^2\). This is a local membership certificate. The check certifies 252 of 257 sampled positions for the chosen perturbation and observes no violations among those certified positions.

At a boundary, the current hard top-k gate can replace one expert with another having very different outputs. A continuous geometric curve can therefore produce discontinuous computation. A smooth shared field or compatible expert outputs could reduce that problem.

For a genuinely smooth field with Lipschitz constants \(L_h\) in state and \(L_r\) in position, two trajectories satisfy the standard differential-inequality bound

\[
\|\Delta h(1)\|
\leq e^{L_h}\|\Delta h(0)\|
+L_r\varepsilon\frac{e^{L_h}-1}{L_h},
\]

with the fraction interpreted as one when \(L_h=0\). This shows why controlling the field's sensitivity matters as much as controlling the path. It assumes the required global bounds on the compared trajectories. The present hard-selection field does not meet that assumption globally. Normalizing a nearly zero port vector also need not be stable with respect to the original input.

For the analytic field above, constraining all scalar rates to be nonpositive prevents state amplification along a fixed route. It still does not certify classification accuracy or truthful factual answers.

## What the sphere contributes

A ball is convex, so bounded Bézier control points give interior containment without repeatedly projecting sampled positions. It also has no preferred global orientation: rotating both a route and the expert centers preserves their distances and routing decisions.

That last property is an internal coordinate symmetry. Rotating an input image or physical object does not automatically rotate the learned route appropriately. To obtain data efficiency from physical rotational symmetry, the encoder and field must respect that symmetry explicitly. [Tensor Field Networks, Thomas et al. (2018)](https://arxiv.org/abs/1802.08219) provide a relevant example of such designed equivariance for three-dimensional data.

Fixed expert centers could additionally support a spatial search index at inference. This may reduce candidate search when there are many experts, but a balanced tree has only average-case search advantages, and GPU branch/dispatch overhead can outweigh them. The present code scans all centers. Neither a sphere nor a curve eliminates that search automatically.

## Hardware efficiency remains a separate condition

The local 68-million-parameter untrained benchmark measured Bézier/full-residual latency of 0.969/3.012 ms at batch one and 5.767/3.578 ms at batch 128. Thus there is already a measured speed benefit in one setting, alongside a reversal in another. Those timings have no matched-accuracy interpretation.

One large temporary tensor explains part of the concern: gathering one \(1024\times512\) float32 expert matrix for two selections across 128 inputs materializes

\[
128\cdot2\cdot1024\cdot512\cdot4
=536{,}870{,}912\text{ bytes}=512\text{ MiB}.
\]

This is an allocation size, not a measurement of physical memory traffic. Grouping inputs by expert can reuse matrices across examples and avoid this particular per-input matrix duplication. [MegaBlocks, Gale et al. (2022)](https://arxiv.org/abs/2211.15841) demonstrates why routing and block-sparse GPU execution need to be designed together. Its reported performance is not a prediction for this Windows implementation.

An optimized dispatcher should be applied to geometric and ordinary expert routing alike. Improving it could reveal a better practical speed frontier without establishing a uniquely Bézier benefit.

## Limits on compressing knowledge

Independent random facts have little reusable structure. For \(N\) arbitrary labels from \(K\) possibilities on known keys, there are \(K^N\) possible tables. A representation able to encode every such table exactly needs at least \(N\log_2 K\) bits. A model with \(P\) parameters stored at \(b\) bits has at most \(2^{Pb}\) parameter states, giving the necessary condition \(Pb\geq N\log_2K\), before accounting for keys or other stored information.

This is a counting argument about arbitrary tables, not a claim that every particular dataset is incompressible. Real knowledge can contain reusable rules and relationships, so a useful inductive bias can represent it much more economically. The original independent-fact experiment intentionally tested retention rather than that kind of semantic structure.

A continuous curve has infinitely many mathematical points, but a finite-precision description does not encode infinitely many independent facts. Replacing every scalar weight with four unrelated Bézier coefficients also increases the local parameter count unless it allows fewer connections or greater sharing elsewhere. Research such as [KAN, Liu et al. (2024)](https://arxiv.org/abs/2404.19756) motivates learned scalar functions for structured problems, while also showing that parameter efficiency and runtime efficiency are different questions.

A curved route is never shorter than the straight chord connecting fixed endpoints. Geometric travel distance therefore does not establish computational savings on a GPU. Reversing a route likewise does not invert arbitrary neural transformations.

## The strongest next experiments

| Hypothesis | Concrete change | Evidence needed |
|---|---|---|
| Related transformations share a compact field | Replace independent expert matrices with shared \(U,V\) and a small spatial coefficient field | Better accuracy per total stored byte than equally sized residual, ordinary expert and shared-field baselines |
| Some tasks tolerate an integrable interior | Use the commuting field and exact endpoint formula above | Competitive accuracy and measured end-to-end speed after including encoder, router and coefficient evaluation |
| Smooth routing improves stability | Bound route/field sensitivity and measure route margins or replace hard selection with a continuous field | Better perturbation accuracy and calibration at comparable clean accuracy |
| The current dispatcher hides sparse-compute gains | Group examples by expert and optimize the shared implementation | Better latency and allocation at matched quality across both geometric and ordinary routing |

For the shared-field and analytic variants, compare curved routes, straight routes, unconstrained position samples, and direct low-rank gating. Include tasks with plausible shared structure, such as related transformations or physical dynamics, together with ordinary held-out learning tasks and independent random-fact controls. A benefit should survive comparisons that isolate the curve from the shared parameterization.

My strongest working hypothesis is that **a compact, smooth field shared across input-dependent routes could improve useful computation per stored byte on structured tasks**. It is more specific and testable than a claim that spherical arrangement alone must improve every efficiency measure.

## Reproduce the mathematical checks

The [check script](../scripts/check_theoretical_benefits.py) uses small constructed arrays and leaves the trained models unchanged:

    .\.venv\Scripts\python.exe scripts\check_theoretical_benefits.py

Outputs include [the numerical results](../runs/theory_checks/results.json), the script hash and the figure below. These checks verify identities and examples; they are not a newly trained model benchmark.

![Conditional mathematical benefits](../runs/theory_checks/conditional_benefits.png)

The existing [trained study](../runs/scaling/report.html) and [larger execution study](../runs/scaling_execution/report.html) remain the measured evidence for the current prototype.
