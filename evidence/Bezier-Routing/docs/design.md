<!-- PUBLICATION-SCOPE -->
> **Historical source in a curated publication.** This document describes the original workspace. Claims that all code, data or weights are available locally do not describe this public snapshot. Some original links, figures and reproduction commands refer to files not distributed here. [Publication scope](../../PUBLICATION_SCOPE.md) · [Reader guide](../../../READ_THIS_FIRST.md).

# Bézier-Routing: activation-dependent routing

The original geometric prototype calculated its full curve from the entry activation. It activated multiple expert groups, but intermediate processing could not change that curve. Its ordinary MoE comparator did make new non-geometric routing decisions from the updated hidden state. The new question is whether feeding intermediate state back into **geometric** routing helps.

This project implements all three requested granularities:

1. **Neurons:** choose individual neurons from the whole sphere.
2. **Groups:** choose groups and execute all neurons in those selected groups.
3. **Hierarchy:** choose groups, then a small subset of neurons within each chosen group. Repeat the decision after processing, so subsequent hops can stay in a group or switch groups.

Each artificial neuron computes one scalar `a_i = SiLU(v_i · LayerNorm(h) + b_i)` and has outgoing synaptic weights `u_i`. The carried activation is a feature vector `h`, updated by a weighted sum of the selected neurons' scalar outputs times their outgoing vectors. This is individual-neuron computation, but not a simulation of independent scalar particles, spikes or physical wire propagation.

At hop `s`, the router reads `(h_s, r_s, s/S)` and proposes a cubic curve from current position `r_s` to a surface destination. The curve's midpoint determines spatial routing scores. Group centers have norm below 0.60; neuron offsets have norm below 0.23, so every neuron lies strictly inside the ball. The hierarchy first scores group centers, then scores only neurons in the chosen groups. Top-two soft mixtures keep task gradients through the selected routing weights; discrete membership changes remain non-smooth.

Each selected branch is a cubic from `r_s` to that neuron's position. For mixture weights `w_i`, the combined curve is `C_s(t) = sum_i w_i C_si(t)`, and the next junction is `r_(s+1) = C_s(1)`. Convexity keeps all branches and their mixture inside the ball. The next curve starts at that junction, giving position continuity. Tangent continuity is not imposed. Branches split and their contributions recombine; the displayed average route is not a claim that one scalar traverses every branch sequentially.

The update is `h_(s+1) = h_s + sum_i w_i u_i a_i / sqrt(S)`. The updated activation and position then determine the next route. After six internal hops, a final cubic reaches the surface and the classifier reads the final feature vector. A bounded hop count prevents unlimited loops. There is no learned adaptive stopping in this first version. The final exit segment performs no additional neuron computation.

## Comparisons fixed before training

Use the existing 8×8 digit dataset, copied into this project, with the original 1,078/359/360 train/validation/test split. Train each configuration for 80 epochs, batch 128, AdamW learning rate 0.003 and weight decay 0.0001; use cosine decay to 0.00015, gradient clipping at 5, and a 0.01 soft-importance penalty. Select checkpoints by validation loss. Use training seeds 7, 19 and 42.

Each of the three modes has an otherwise-identical comparison that always gives the planner the **entry activation**. Position/time still change, and the number of planner calls is unchanged. This isolates intermediate-state feedback more closely than comparing two entirely different architectures. Additional controls use dynamic hierarchical straight segments and dynamic direct non-geometric neuron selection. These give 24 planned runs. Width 64, eight groups, eight neurons per group, six hops, two selected groups and two selected neurons per group. Group-only routing executes 16 neurons per hop; flat-neuron and hierarchical routing execute four. Between-mode compute budgets therefore differ; each feedback/frozen pair has the same budget and parameter count.

Causal checks perturb the activation after a selected hop while keeping earlier processing fixed, then check later curves and selections. Disabling feedback on a trained model is a separate diagnostic, not a replacement for the independently trained comparison. Save actual routes, neuron/group selections, prediction changes, and model checkpoints. Report the ability to reroute separately from any accuracy benefit. This initial task cannot establish knowledge compression or large-scale efficiency. Other project work may share the GPU, so this experiment makes no latency claims.

## Why it might help, and what remains uncertain

An intermediate result may reveal which computation is needed next, allowing conditional composition and reuse of neurons. Repeated routing also has overhead, and jointly learning neuron functions and route choices can be difficult. Existing recursive routing-network research supports studying this hypothesis; it does not establish an advantage for Bézier geometry or this implementation.

- Rosenbaum, Klinger & Riemer, [Routing Networks: Adaptive Selection of Non-linear Functions for Multi-Task Learning](https://arxiv.org/abs/1711.01239).
- Rosenbaum et al., [Routing Networks and the Challenges of Modular and Compositional Computation](https://arxiv.org/abs/1904.12774).
