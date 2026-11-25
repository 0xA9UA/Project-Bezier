# What is actually being compressed or accelerated?

Let hidden width be d, shared rank q, number of processing steps S, independent
expert count M, and experts selected per step k. The 3D ball is an address space;
the hidden information vector has d components. All models include an input
encoder and an output decoder, which must be counted in storage and runtime.

## Shared nonlinear field

At each of S midpoint samples on a route, the model applies

    h_next = h + D( SiLU(V LayerNorm(h) + b_v) * g(position) ) / S

Here V is q×d and D includes a d×q matrix and output bias. One pair of matrices
is reused at every step and for every spatial location. The gate is

    g(r) = 1 + phi(r) C_eff
    C_eff[:, j] = C[:, j] / (1 + sum_i |C[i, j]|)

where phi contains all 20 monomials of total spatial degree at most three.
For points inside the unit ball, each monomial has absolute value at most one,
so every gate lies between 0 and 2. Input-dependent route control points come
from a d→32→12 MLP. Its endpoints are normalized onto the sphere; its two inner
controls are smoothly constrained to the open ball.

The leading processing-matrix parameter count changes from **2Mdq** to **2dq**,
with another 20q polynomial coefficients and small routing/bias terms. In our
M=16 experiment, this removes fifteen independent matrix pairs. It also
restricts the model: all positions now share a feature basis instead of having
independent expert transformations. An arbitrary bank of unrelated expert
matrices generally cannot be represented exactly by this shared field.

The leading hidden projection work changes from **2Skdq** multiply-accumulates
to **2Sdq**, since k=2 selected experts become one shared update. Routing,
normalization, activations, gate evaluation and memory movement add overhead.
The original implementation also gathers parameter matrices per input. The
shared model uses ordinary batch matrix multiplications, avoiding those large
temporary matrix copies. Optimized grouped expert kernels could reduce the
original implementation's overhead; our speed results describe the tested code.

The matched residual control uses four distinct ordinary bottleneck blocks,
choosing their bottleneck width to put its total parameter count within 1% of
the shared Bézier model. The constant and direct-gate controls also reuse the
same processing matrices. These are necessary controls for attributing a gain
to the curve instead of to parameter sharing.

## Exactly integrable commuting field

The restricted model fixes one orthonormal basis U, d×q, across each trajectory:

    h'(t) = U diag(a(gamma_x(t))) U^T h(t)
    U^T U = I

Rates a use the same 20 spatial monomials, with coefficient scale 2 instead of 1.
Consequently |a_j|<2. All state-transition generators commute. In coordinates
z=U^T h, each component satisfies z'_j=a_j(t)z_j and the orthogonal component
is constant. Integration gives

    I_x = integral_0^1 a(gamma_x(t)) dt
    h(1) = h(0) + U [expm1(I_x) * (U^T h(0))].

A cubic path substituted into a total-degree-three spatial polynomial has
degree at most nine in t. Five-point Gauss-Legendre quadrature therefore gives
the exact integral in real arithmetic. Floating-point errors remain. A general
64-term tensor-product cubic volume would require up to fourteen Gauss nodes;
it is not the basis used here.

Hidden projection work is now **2dq**, independent of a numerical step count.
There are five evaluations of the small polynomial field, not five evaluations
of a full hidden-state update. Stored processing weights are dq basis entries
plus 20q field coefficients and routing. The basis is kept orthonormal by a QR
retraction after each optimizer update. QR is a training cost; it is absent at
inference, and no extra cached copy of the basis is hidden from storage counts.

The numerical baseline integrates this same trained field with RK4, already
factored into U coordinates. It precomputes all spatial rates once and performs
only diagonal updates inside the RK4 loop. Comparing against this baseline
avoids exaggerating the analytic gain by repeatedly projecting full hidden
vectors in a numerical solver.

A straight path only yields a degree-three time polynomial, so two Gauss nodes
would suffice. The frozen training models use the same five-node implementation
for both path types. A post-training two-node check is included separately to
expose that additional optimization opportunity. Directly predicting the rate
integral removes spatial integration altogether and is another necessary
efficiency baseline.

## Restrictions matter

For a fixed route, the analytic transition's operator norm is at most exp(2).
That bound does not establish a global Lipschitz bound for the whole predictor:
the encoder and decoder are learned, and the route itself depends on the input.
Normalizing raw endpoint vectors also has an unbounded derivative as their norm
approaches zero. Empirical corruption and calibration tests are therefore needed;
bounded coordinates alone do not prove reliability.

Reversing the order of an analytic model's route leaves its rate integral
unchanged. Its final output is thus direction-order invariant in exact arithmetic.
That is a useful symmetry for some tasks and a restriction for others. A general
nonlinear field or a noncommuting linear field does not have that property.
The four-step nonlinear models can depend on the order in which positions occur.
The post-training route intervention tests quantify dependence on routes and
check the reversal property without retraining or selecting checkpoints.

Finally, fixed-precision curve parameters cannot encode arbitrarily many
independent facts without error. Shared structure permits compression; unrelated
associations test how much capacity is lost. The synthetic recall tables use a
seeded pseudorandom generator and are reproducible from their generator and seed.
The neural learners are given key-label examples, not the generator or seed.
They are a controlled memorization test, not literally incompressible files or a
test of natural-language knowledge.
