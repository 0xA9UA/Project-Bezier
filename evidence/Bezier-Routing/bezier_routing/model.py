"""State-dependent routing through scalar neurons and optional groups.

Each neuron computes one scalar activation and has a learned outgoing vector.
The carried feature vector is shared across the selected branches; this is not
a simulation of independent spiking particles or of physical signal travel.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F


def inside(x, radius=.95):
    return radius*x/torch.sqrt(1+x.square().sum(-1, keepdim=True))


def surface(x):
    norm = x.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(x)
    fallback[..., 0] = 1
    return torch.where(norm > 1e-7, x/norm.clamp_min(1e-7), fallback)


def evaluate_curve(control, t):
    t = torch.as_tensor(t, device=control.device, dtype=control.dtype)
    basis = torch.stack(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3), -1)
    return torch.einsum("...cd,tc->...td", control, basis)


@dataclass
class Config:
    input_dim: int = 64
    classes: int = 10
    width: int = 64
    groups: int = 8
    neurons_per_group: int = 8
    group_topk: int = 2
    neuron_topk: int = 2
    hops: int = 6
    router_width: int = 32
    bandwidth: float = .45

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not 2 <= self.group_topk <= self.groups:
            raise ValueError("group_topk must be at least two and no larger than groups")
        if not 2 <= self.neuron_topk <= self.neurons_per_group:
            raise ValueError("neuron_topk must be at least two and no larger than neurons_per_group")


class RoutingModel(nn.Module):
    def __init__(self, config=None, mode="hierarchical", feedback=True, curved=True):
        super().__init__()
        if mode not in ("neurons", "groups", "hierarchical", "direct"):
            raise ValueError(mode)
        self.cfg = c = config or Config()
        self.mode, self.feedback, self.curved = mode, feedback, curved
        self.encoder = nn.Sequential(nn.Linear(c.input_dim, c.width), nn.SiLU())
        self.norm = nn.LayerNorm(c.width)
        n = c.groups*c.neurons_per_group
        self.read = nn.Parameter(torch.randn(n, c.width)/math.sqrt(c.width))
        self.bias = nn.Parameter(torch.zeros(n))
        self.write = nn.Parameter(torch.randn(n, c.width)*.25)
        self.decoder = nn.Linear(c.width, c.classes)
        self.raw_groups = nn.Parameter(torch.randn(c.groups, 3))
        self.raw_neurons = nn.Parameter(torch.randn(c.groups, c.neurons_per_group, 3))
        self.entry = nn.Linear(c.width, 3)
        self.planner = nn.Sequential(nn.Linear(c.width+4, c.router_width), nn.SiLU(),
                                     nn.Linear(c.router_width, 9))
        nn.init.normal_(self.planner[-1].weight, std=.16)
        with torch.no_grad():
            self.planner[-1].bias.copy_(torch.tensor([-.25, .3, 0., .25, -.3, 0., 1., 0., 0.]))
        # Direct routing is an additional learned, non-geometric control. Its
        # extra parameter cost is reported; it is not a parameter-matched pair.
        self.direct_gate = nn.Linear(c.width, n) if mode == "direct" else None

    @property
    def group_positions(self):
        return inside(self.raw_groups, .60)

    @property
    def neuron_positions(self):
        return self.group_positions[:, None] + inside(self.raw_neurons, .23)

    def plan(self, state, position, phase):
        time = state.new_full((len(state), 1), phase)
        raw = self.planner(torch.cat((self.norm(state), position, time), -1)).reshape(-1, 3, 3)
        end = surface(raw[:, 2])
        if self.curved:
            q1, q2 = inside(raw[:, 0]), inside(raw[:, 1])
        else:
            q1, q2 = (2*position+end)/3, (position+2*end)/3
        return torch.stack((position, q1, q2, end), 1)

    def select(self, state, proposal):
        c, batch = self.cfg, len(state)
        centers = self.neuron_positions
        scores = -(proposal[:, None]-self.group_positions).square().sum(-1)/c.bandwidth**2
        group_values, groups = scores.topk(c.group_topk, dim=-1)
        group_weights = group_values.softmax(-1)
        group_balance = c.groups*scores.softmax(-1).mean(0).square().sum()-1
        if self.mode in ("neurons", "direct"):
            flat_scores = (self.direct_gate(self.norm(state)) if self.mode == "direct" else
                           -(proposal[:, None]-centers.flatten(0, 1)).square().sum(-1)/c.bandwidth**2)
            values, indices = flat_scores.topk(c.group_topk*c.neuron_topk, dim=-1)
            weights = values.softmax(-1)
            probabilities = flat_scores.softmax(-1).mean(0)
            balance = probabilities.square().sum()*centers.shape[0]*centers.shape[1]-1
            return indices, weights, indices//c.neurons_per_group, balance
        local_centers = centers[groups]
        if self.mode == "groups":
            local = torch.arange(c.neurons_per_group, device=state.device).expand(batch, c.group_topk, -1)
            weights = group_weights[..., None].expand(-1, -1, c.neurons_per_group)/c.neurons_per_group
            balance = group_balance
        else:
            local_scores = -(proposal[:, None, None]-local_centers).square().sum(-1)/(.5*c.bandwidth)**2
            values, local = local_scores.topk(c.neuron_topk, dim=-1)
            weights = group_weights[..., None]*values.softmax(-1)
            # This is a soft-importance regularizer, not guaranteed hard load balancing.
            importance = local_scores.softmax(-1).mean((0, 1))
            balance = group_balance + c.neurons_per_group*importance.square().sum()-1
        indices = groups[..., None]*c.neurons_per_group + local
        return indices.flatten(1), weights.flatten(1), groups, balance

    def apply_neurons(self, state, indices, weights, dense_reference=False):
        z = self.norm(state)
        if dense_reference:
            activations = F.silu(F.linear(z, self.read, self.bias)).gather(1, indices)
        else:
            activations = F.silu((z[:, None]*self.read[indices]).sum(-1)+self.bias[indices])
        return (weights[..., None]*activations[..., None]*self.write[indices]).sum(1)

    def forward(self, x, *, trace=False, return_aux=False, intervention=None,
                freeze_feedback=False, bypass=False, dense_reference=False):
        """intervention=(completed_hop_index, additive_state_delta) is diagnostic.

        A frozen-feedback comparison keeps using the entry activation in the
        planner. It still recomputes geometry from the current position and time,
        so its planner calls and stored parameters match the feedback variant.
        """
        c = self.cfg
        initial = h = self.encoder(x)
        position = surface(self.entry(initial))
        points, plans, branches, controls = [position], [], [], []
        selections, weights_log, groups_log, states, balances = [], [], [], [h], []
        centers = self.neuron_positions.flatten(0, 1)
        for hop in range(c.hops):
            route_state = h if self.feedback and not freeze_feedback else initial
            plan = self.plan(route_state, position, hop/c.hops)
            proposal = evaluate_curve(plan, [.5])[:, 0]
            indices, weights, groups, balance = self.select(route_state, proposal)
            # Each realized branch ends exactly at a selected neuron. Their
            # weighted mean is another cubic, starting at the same junction.
            branch = plan[:, None].expand(-1, indices.shape[1], -1, -1).clone()
            branch[:, :, 3] = centers[indices]
            if not self.curved:
                branch[:, :, 1] = (2*position[:, None]+branch[:, :, 3])/3
                branch[:, :, 2] = (position[:, None]+2*branch[:, :, 3])/3
            combined = (branch*weights[:, :, None, None]).sum(1)
            position = combined[:, 3]
            if not bypass:
                h = h + self.apply_neurons(h, indices, weights, dense_reference)/math.sqrt(c.hops)
            if intervention is not None and hop == intervention[0]:
                h = h + intervention[1]
            if return_aux:
                balances.append(balance)
            if trace:
                plans.append(plan)
                branches.append(branch)
                controls.append(combined)
                selections.append(indices)
                weights_log.append(weights)
                groups_log.append(groups)
                points.append(position)
                states.append(h)
        exit_state = h if self.feedback and not freeze_feedback else initial
        exit_curve = self.plan(exit_state, position, 1.)
        logits = self.decoder(h)
        if not (trace or return_aux):
            return logits
        aux = {}
        if return_aux:
            aux["balance_loss"] = torch.stack(balances).mean()
        if trace:
            aux.update(positions=torch.stack(points, 1), proposals=torch.stack(plans, 1),
                       controls=torch.stack(controls, 1), branches=torch.stack(branches, 1),
                       neurons=torch.stack(selections, 1), weights=torch.stack(weights_log, 1),
                       groups=torch.stack(groups_log, 1), states=torch.stack(states, 1),
                       exit_curve=exit_curve, exit=exit_curve[:, 3])
        return logits, aux

    def description(self):
        c = self.cfg
        active = c.group_topk*(c.neurons_per_group if self.mode == "groups" else c.neuron_topk)
        return {"mode": self.mode, "feedback": self.feedback, "curved": self.curved,
                "config": asdict(c), "parameters": sum(p.numel() for p in self.parameters()),
                "active_neurons_per_hop": active, "neuron_calls_per_input": active*c.hops,
                "stored_neurons": c.groups*c.neurons_per_group}
